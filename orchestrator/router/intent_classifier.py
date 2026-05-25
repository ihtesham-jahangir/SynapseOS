"""
Intent classification engine — v3.5.

Three-stage pipeline:
  Stage 1 – fast keyword/regex matching (microseconds, no model call)
             Returns scored matches for ALL intents (not just the top hit)
  Stage 2 – per-exemplar semantic similarity via BGE embeddings
             Triggered only when stage-1 max confidence < KEYWORD_THRESHOLD
             Compares query to each individual exemplar, not mean-pooled centroids
  Stage 3 – confidence fusion
             When both stages agree on the top intent, boost confidence
             When they disagree, blend scores weighted by stage reliability

Additional capabilities (v3.5):
  - Multi-intent detection: returns secondary intents above MIN_SECONDARY_CONFIDENCE
  - Subtype extraction: language hints, translation targets, data tool detection
  - Context-aware classification: last 2 turns shift prior probabilities
  - Adaptive confidence: keyword density × semantic agreement multiplicative boost
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from orchestrator.core.base import BaseIntentClassifier
from orchestrator.core.types import Intent, IntentScore, IntentType
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import AsyncLRUCache

log = get_logger(__name__)

# ─── Keyword rules (Stage 1) ─────────────────────────────────────────────────
# More patterns per intent + tighter regexes to reduce false positives.

_KEYWORD_RULES: Dict[IntentType, List[str]] = {
    IntentType.CODING: [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bscript\b", r"\bdebug\b",
        r"\bprogramm", r"\bimport\b", r"\bdef \w+\(", r"\bsyntax\b",
        r"\bbug\b", r"\berror\b.*line", r"\bimplementi?\b", r"\brefactor\b",
        r"\bgithub\b", r"\bapi endpoint", r"\bunit test", r"\bcompile\b",
        r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\bjava\b",
        r"\bc\+\+\b", r"\brust\b", r"\bgo\b.*lang", r"\bkotlin\b",
        r"\bsql query", r"\bdocker", r"\bkubernetes", r"\bcicd",
        r"\bpull request", r"\brepo\b", r"\bcommit\b", r"\bbranch\b",
        r"\bpackage\b", r"\bdependenc", r"\bframework\b", r"\blibrary\b",
        r"\balgorithm\b", r"\bdata struct", r"\brecursi", r"\biterat",
    ],
    IntentType.MATH: [
        r"\bsolve\b", r"\bcalculate\b", r"\bequation\b", r"\bintegral\b",
        r"\bderivative\b", r"\bmatrix\b", r"\bprob(?:ability)?\b", r"\bstatistic",
        r"\bformula\b", r"\balgebra\b", r"\bcalculus\b", r"\bcompute\b",
        r"\d+\s*[\+\-\*\/\^]\s*\d+", r"\bsum of\b", r"\baverage\b",
        r"\bgeometr", r"\btrigonometr", r"\bvector\b", r"\blinear\b.*algebra",
        r"\bdifferential\b", r"\boptimiz", r"\bminimiz", r"\bmaximiz",
        r"\beigen", r"\bdeterminant\b", r"\bpolynomial\b", r"\bfunction\b.*graph",
        r"\bpercentage\b", r"\bratio\b", r"\bfraction\b", r"\bfactorial\b",
    ],
    IntentType.TRANSLATION: [
        r"\btranslat", r"\bin french\b", r"\bin spanish\b", r"\bin german\b",
        r"\bin japanese\b", r"\bin chinese\b", r"\bin arabic\b", r"\bin portuguese\b",
        r"\bin italian\b", r"\bin korean\b", r"\bin russian\b", r"\bin hindi\b",
        r"\bto english\b", r"\bfrom english\b", r"\blanguage\b",
        r"\bsay .* in\b", r"\bhow do you say\b", r"\bword for\b",
    ],
    IntentType.SUMMARIZATION: [
        r"\bsummar", r"\bbriefly\b", r"\btldr\b", r"\bshorten\b",
        r"\bkey points\b", r"\bmain idea", r"\bcondense\b", r"\bsynopsis\b",
        r"\boverview\b", r"\bgist\b", r"\bdigest\b", r"\bbreakdown\b",
        r"\bhighlight", r"\bkeyword", r"\bextract\b.*point",
    ],
    IntentType.RETRIEVAL: [
        r"\bwhat is\b", r"\bwho is\b", r"\bwhen did\b", r"\bwhere is\b",
        r"\bhow does\b", r"\btell me about\b", r"\bexplain\b",
        r"\bdefine\b", r"\bmeaning of\b", r"\bwhat are\b", r"\blist of\b",
        r"\bdescribe\b", r"\blook up\b", r"\bfind me\b", r"\bcan you tell\b",
        r"\bwhat do you know about\b", r"\bgive me info",
    ],
    IntentType.REASONING: [
        r"\bwhy\b", r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b",
        r"\bpros and cons\b", r"\bargument\b", r"\blogic\b", r"\bdeduce\b",
        r"\binfer\b", r"\bcritique\b", r"\bassess\b", r"\bthink\b.*step",
        r"\bshould i\b", r"\bwhich is better\b", r"\bdifference between\b",
        r"\badvantage", r"\bdisadvantage", r"\btradeoff\b", r"\bimpact\b",
        r"\bcause\b.*effect", r"\bhypothes", r"\bconclusion\b",
    ],
    IntentType.VOICE: [
        r"\btranscrib", r"\bspeak\b", r"\baudio\b", r"\bspeech\b",
        r"\brecogniz", r"\bvoice\b", r"\bwav\b", r"\bmp3\b",
        r"\bpodcast\b", r"\btext.to.speech\b", r"\bspeech.to.text\b",
    ],
    # ── v3.5 new domains ──────────────────────────────────────────────────────
    IntentType.WRITING: [
        r"\bwrite\b.*(?:essay|article|email|letter|report|blog|post|paragraph|story|poem)",
        r"\bdraft\b.*(?:message|email|proposal|report)",
        r"\bemail\b.*(?:write|compose|draft)", r"\bcompose\b",
        r"\bproofread\b", r"\bedit\b.*(?:text|essay|article|paragraph)",
        r"\bgrammar\b", r"\bspelling\b", r"\bpunctuati",
        r"\bcovering? letter\b", r"\bresume\b", r"\bcv\b",
        r"\bprofessional tone\b", r"\bformal\b.*writing",
        r"\bcreative writing\b", r"\bnarrative\b",
    ],
    IntentType.DATA_ANALYSIS: [
        r"\bdata\b.*(?:analysis|analytics|pipeline|processing)", r"\bsql\b",
        r"\bselect\b.*from\b", r"\bjoin\b.*table", r"\bpandas\b",
        r"\bdataframe\b", r"\bnumpy\b", r"\bmatplotlib\b", r"\bseaborn\b",
        r"\bvisuali[sz]", r"\bchart\b", r"\bgraph\b.*data", r"\bplot\b",
        r"\bcsv\b", r"\bjson\b.*parse", r"\bexcel\b", r"\bspreadsheet\b",
        r"\bdashboard\b", r"\bmetric", r"\bkpi\b", r"\btrend\b.*data",
        r"\bcluster", r"\bregression\b", r"\bclassif", r"\bprediction\b",
        r"\bmachine learning\b", r"\bmodel\b.*train", r"\bfeature\b.*engineer",
    ],
    IntentType.CREATIVE: [
        r"\bbrainstorm\b", r"\bidea[s]?\b.*(?:for|about)", r"\bcreate\b.*(?:concept|idea)",
        r"\bimagine\b", r"\bstory\b.*(?:about|with|idea)", r"\bplot\b.*story",
        r"\bcharacter\b.*(?:design|create|develop)", r"\bworld.?build",
        r"\bpoem\b", r"\blyric", r"\bsong\b.*write", r"\bfiction\b",
        r"\bscenario\b", r"\bwhat if\b", r"\bhypothetical\b",
        r"\binvent\b", r"\bcreative\b.*idea", r"\binnovate\b",
        r"\boriginal\b.*concept", r"\bgame\b.*design", r"\bcharacter\b.*backstory",
    ],
    IntentType.SECURITY: [
        r"\bsecurity\b", r"\bvulnerabilit", r"\bexploit\b", r"\bpenetration\b",
        r"\bpentest\b", r"\bhack\b", r"\bcve\b", r"\bowasp\b",
        r"\binjection\b", r"\bxss\b", r"\bsqli\b", r"\bcsrf\b",
        r"\bauth(?:entication|orization)\b", r"\bencrypt\b", r"\bdecrypt\b",
        r"\bssl\b", r"\btls\b", r"\bcertificate\b", r"\bfirewall\b",
        r"\bmalware\b", r"\bphishing\b", r"\bsocial engineering\b",
        r"\bsecure code review\b", r"\bsanitize\b.*input", r"\bpassword\b.*hash",
        r"\btoken\b.*(?:jwt|oauth|csrf)", r"\bprivileged?\b.*access",
    ],
    IntentType.PLANNING: [
        r"\bplan\b.*(?:project|task|sprint|roadmap|schedule)",
        r"\bproject\b.*(?:plan|timeline|milestone)", r"\broadmap\b",
        r"\bsprint\b", r"\bbacklog\b", r"\bprioritiz", r"\bschedule\b",
        r"\btimeline\b", r"\bdeadline\b", r"\bsteps? to\b", r"\bhow to\b.*(?:start|begin|build)",
        r"\bworkflow\b", r"\bprocess\b.*(?:design|map)", r"\bstrategy\b",
        r"\bgoal\b.*(?:set|achieve|reach)", r"\bokr\b", r"\bkpi\b.*plan",
        r"\bbreak(?:down| down)\b.*task", r"\bsubtask\b",
        r"\bdo i need to\b", r"\bwhat do i need\b", r"\bstep by step\b",
    ],
    IntentType.EDUCATION: [
        r"\bteach\b.*(?:me|how)", r"\bexplain\b.*(?:concept|like|simply|beginner)",
        r"\bhow does\b.*work\b", r"\blearn\b.*(?:about|how to)",
        r"\btutorial\b", r"\bcourse\b", r"\blesson\b", r"\bunderstand\b",
        r"\bfor beginner", r"\bfrom scratch\b", r"\bbasics? of\b",
        r"\bwhat is\b.*(?:and how|meaning|definition|example)",
        r"\bconcept\b.*(?:explain|understand)", r"\bsimpl(?:y|ify)\b",
        r"\banalog(?:y|ies)\b", r"\bexample\b.*(?:of|to understand)",
        r"\bstep.by.step\b.*(?:learn|understand|explain)",
        r"\belaborate\b", r"\bcan you explain\b",
    ],
}

# ─── Exemplar sentences (Stage 2) ────────────────────────────────────────────
# More exemplars per intent for better discriminative power.

_INTENT_EXEMPLARS: Dict[IntentType, List[str]] = {
    IntentType.CODING: [
        "Write a Python function to sort a list",
        "Fix this JavaScript bug in my code",
        "How do I implement a binary search tree in Java?",
        "Generate unit tests for this function",
        "Refactor this code to use async/await",
        "What is the time complexity of quicksort?",
        "Create a REST API endpoint using FastAPI",
        "Debug this null pointer exception",
    ],
    IntentType.MATH: [
        "Solve this quadratic equation: 2x² + 5x - 3 = 0",
        "What is the integral of x squared?",
        "Calculate the probability of drawing two aces",
        "Find the eigenvalues of this matrix",
        "Prove that the square root of 2 is irrational",
        "What is the derivative of sin(x)cos(x)?",
        "Compute the mean and variance of this dataset",
    ],
    IntentType.TRANSLATION: [
        "Translate this paragraph to French",
        "How do you say hello in Japanese?",
        "Convert this English text to Spanish",
        "What is the German word for freedom?",
        "Translate: Bonjour, comment ça va?",
    ],
    IntentType.SUMMARIZATION: [
        "Summarize this article in three bullet points",
        "Give me the key takeaways from this document",
        "Create a brief overview of this content",
        "What are the main points of this research paper?",
        "Condense this 5-page report into one paragraph",
    ],
    IntentType.RETRIEVAL: [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What are the symptoms of diabetes?",
        "When was the Eiffel Tower built?",
        "What is photosynthesis?",
        "Tell me about black holes",
    ],
    IntentType.REASONING: [
        "What are the pros and cons of remote work?",
        "Analyze the causes of the 2008 financial crisis",
        "Compare object-oriented and functional programming paradigms",
        "Should I use PostgreSQL or MongoDB for this use case?",
        "Why did the Roman Empire fall?",
        "Evaluate the ethical implications of AI in hiring",
    ],
    IntentType.CONVERSATION: [
        "Hello, how are you?",
        "Tell me a joke",
        "What can you help me with?",
        "Thanks for your help!",
        "That's a great point",
        "I disagree with that approach",
    ],
    IntentType.VOICE: [
        "Transcribe this audio file",
        "Convert my speech to text",
        "Generate audio from this text",
        "What does the speaker say in this recording?",
    ],
    IntentType.WRITING: [
        "Write a professional email declining a meeting",
        "Draft a cover letter for a software engineering role",
        "Write a blog post about machine learning trends",
        "Help me improve the tone of this paragraph",
        "Proofread and fix grammar in this essay",
        "Write an executive summary of this report",
        "Compose a formal complaint letter",
    ],
    IntentType.DATA_ANALYSIS: [
        "Write a SQL query to find the top 10 customers by revenue",
        "Analyze this dataset with pandas and show trends",
        "Create a matplotlib chart from this data",
        "How do I join two tables in SQL?",
        "Build a simple linear regression model in scikit-learn",
        "Parse this JSON file and extract user data",
        "What does this correlation matrix tell me?",
    ],
    IntentType.CREATIVE: [
        "Brainstorm 10 startup ideas in the fintech space",
        "Write a short story about a robot discovering emotions",
        "Come up with a creative name for my app",
        "Help me design a game mechanic for a puzzle game",
        "Write a poem about the ocean at night",
        "What if humans could photosynthesize? Explore the implications",
        "Invent a new cocktail recipe and give it a name",
    ],
    IntentType.SECURITY: [
        "Review this code for SQL injection vulnerabilities",
        "Explain how cross-site scripting attacks work",
        "How do I implement proper JWT authentication?",
        "What are common OWASP Top 10 vulnerabilities?",
        "How should I hash passwords securely?",
        "Analyze this login form for security issues",
        "Explain the difference between authentication and authorization",
    ],
    IntentType.PLANNING: [
        "Create a project plan for building a mobile app in 3 months",
        "Break down the task of launching a new product",
        "Help me prioritize these 20 backlog items",
        "What are the steps to migrate from PostgreSQL to MongoDB?",
        "Design a sprint plan for our two-week iteration",
        "Create a roadmap for learning machine learning from scratch",
        "What do I need to start a freelance business?",
    ],
    IntentType.EDUCATION: [
        "Explain recursion to a 10-year-old",
        "Teach me how neural networks work from scratch",
        "What is the difference between supervised and unsupervised learning?",
        "I don't understand how async/await works, can you explain?",
        "Give me a beginner's guide to Docker",
        "Explain the concept of entropy in information theory",
        "How does the internet actually work?",
    ],
}

# ─── Subtype extraction patterns ─────────────────────────────────────────────

_CODING_LANGUAGE_PATTERNS = [
    (r"\bpython\b", "python"), (r"\bjavascript\b|\bjs\b", "javascript"),
    (r"\btypescript\b|\bts\b", "typescript"), (r"\bjava\b", "java"),
    (r"\bc\+\+\b|\bcpp\b", "cpp"), (r"\brust\b", "rust"),
    (r"\bgo\b.*lang|\bgolang\b", "go"), (r"\bkotlin\b", "kotlin"),
    (r"\bswift\b", "swift"), (r"\bruby\b", "ruby"),
    (r"\bphp\b", "php"), (r"\bcsharp\b|\bc#\b", "csharp"),
    (r"\bsql\b", "sql"), (r"\bbash\b|\bshell\b", "bash"),
]

_TRANSLATION_TARGET_PATTERNS = [
    (r"\b(?:to |in )french\b", "french"), (r"\b(?:to |in )spanish\b", "spanish"),
    (r"\b(?:to |in )german\b", "german"), (r"\b(?:to |in )japanese\b", "japanese"),
    (r"\b(?:to |in )chinese\b", "chinese"), (r"\b(?:to |in )arabic\b", "arabic"),
    (r"\b(?:to |in )portuguese\b", "portuguese"), (r"\b(?:to |in )italian\b", "italian"),
    (r"\b(?:to |in )korean\b", "korean"), (r"\b(?:to |in )russian\b", "russian"),
    (r"\b(?:to |in )hindi\b", "hindi"), (r"\bto english\b", "english"),
]

_DATA_TOOL_PATTERNS = [
    (r"\bpandas\b", "pandas"), (r"\bsql\b|\bselect\b.*from", "sql"),
    (r"\bspark\b", "spark"), (r"\bmatplotlib\b|\bseaborn\b|\bplotly\b", "visualization"),
    (r"\bsklearn\b|\bscikit", "sklearn"), (r"\btensorflow\b|\bkeras\b", "tensorflow"),
    (r"\bpytorch\b", "pytorch"), (r"\bnumpy\b", "numpy"),
]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _extract_subtype(text: str, intent_type: IntentType) -> Optional[str]:
    """Extract fine-grained subtype from the query text."""
    lower = text.lower()
    if intent_type == IntentType.CODING:
        for pattern, lang in _CODING_LANGUAGE_PATTERNS:
            if re.search(pattern, lower):
                return lang
    elif intent_type == IntentType.TRANSLATION:
        for pattern, lang in _TRANSLATION_TARGET_PATTERNS:
            if re.search(pattern, lower):
                return lang
    elif intent_type == IntentType.DATA_ANALYSIS:
        for pattern, tool in _DATA_TOOL_PATTERNS:
            if re.search(pattern, lower):
                return tool
    return None


class HybridIntentClassifier(BaseIntentClassifier):
    """
    Three-stage intent classifier (v3.5):
      1. Keyword fast-path — scored per intent (not winner-take-all)
      2. Per-exemplar semantic similarity — individual comparisons beat mean-pooled
      3. Confidence fusion — agreement boost, context prior, subtype extraction
    """

    KEYWORD_THRESHOLD = 0.72      # min confidence for keyword path to win outright
    KEYWORD_CONFIDENCE = 0.72     # alias kept for backward compatibility
    SEMANTIC_THRESHOLD = 0.45     # min cosine similarity to assign any semantic intent
    MIN_SECONDARY_CONFIDENCE = 0.40  # threshold for reporting secondary intents
    AGREEMENT_BOOST = 0.08        # added when keyword and semantic agree on top intent
    CONTEXT_WEIGHT = 0.12         # how much prior context shifts confidence

    def __init__(self, embedder=None) -> None:
        self._embedder = embedder
        # Per-exemplar embeddings: intent → list of (text, embedding) pairs
        self._exemplar_embeddings: Optional[Dict[IntentType, List[np.ndarray]]] = None
        self._cache: AsyncLRUCache = AsyncLRUCache(maxsize=1024)
        self._patterns: Dict[IntentType, List[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in _KEYWORD_RULES.items()
        }

    async def _ensure_exemplars(self) -> None:
        if self._exemplar_embeddings is not None or self._embedder is None:
            return

        all_texts: List[str] = []
        all_intents: List[IntentType] = []
        for intent, examples in _INTENT_EXEMPLARS.items():
            all_texts.extend(examples)
            all_intents.extend([intent] * len(examples))

        embeddings = await self._embedder.embed(all_texts)
        arr = np.array(embeddings, dtype=np.float32)

        self._exemplar_embeddings = {}
        idx = 0
        for intent, examples in _INTENT_EXEMPLARS.items():
            n = len(examples)
            self._exemplar_embeddings[intent] = [arr[idx + i] for i in range(n)]
            idx += n

        log.info("Intent exemplar embeddings loaded (per-exemplar)", intents=len(self._exemplar_embeddings))

    def _keyword_score_all(self, text: str) -> Dict[IntentType, float]:
        """
        Score ALL intents via keyword matching.
        Returns raw scores (hit counts) for every matched intent.
        """
        scores: Dict[IntentType, float] = {}
        for intent, patterns in self._patterns.items():
            hits = sum(1 for p in patterns if p.search(text))
            if hits > 0:
                # Confidence = base + scaled hit rate (caps at KEYWORD_THRESHOLD)
                confidence = min(self.KEYWORD_THRESHOLD, 0.60 + hits * 0.06)
                scores[intent] = confidence
        return scores

    async def _semantic_score_all(self, text: str) -> Dict[IntentType, float]:
        """
        Score ALL intents via per-exemplar cosine similarity.
        Uses max-similarity (best matching exemplar) instead of mean-pool.
        """
        if self._embedder is None:
            return {}
        await self._ensure_exemplars()
        if not self._exemplar_embeddings:
            return {}

        query_emb = np.array(
            await self._embedder.embed_query(text), dtype=np.float32
        )

        scores: Dict[IntentType, float] = {}
        for intent, exemplar_embs in self._exemplar_embeddings.items():
            sims = [_cosine_sim(query_emb, e) for e in exemplar_embs]
            # Use 80th-percentile similarity: more robust than pure max but
            # more discriminative than mean
            sims.sort(reverse=True)
            top_k = max(1, len(sims) // 5 + 1)
            score = sum(sims[:top_k]) / top_k
            if score >= self.SEMANTIC_THRESHOLD:
                scores[intent] = score

        return scores

    def _apply_context_prior(
        self,
        scores: Dict[IntentType, float],
        context: Optional[str],
    ) -> Dict[IntentType, float]:
        """
        Shift scores based on the intent visible in recent conversation context.
        The top keyword-matched intent from the context gets a small boost.
        """
        if not context:
            return scores
        ctx_scores = self._keyword_score_all(context)
        if not ctx_scores:
            return scores

        ctx_top = max(ctx_scores, key=ctx_scores.__getitem__)
        result = dict(scores)
        if ctx_top in result:
            result[ctx_top] = min(1.0, result[ctx_top] + self.CONTEXT_WEIGHT)
        elif ctx_scores[ctx_top] > 0.5:
            result[ctx_top] = ctx_scores[ctx_top] * self.CONTEXT_WEIGHT
        return result

    def _keyword_classify(self, text: str) -> Optional[Tuple[IntentType, float]]:
        """Backward-compat wrapper over _keyword_score_all. Returns (intent, confidence) or None."""
        scores = self._keyword_score_all(text)
        if not scores:
            return None
        best = max(scores, key=scores.__getitem__)
        return best, scores[best]

    async def classify(self, text: str, context: Optional[str] = None) -> Intent:
        cache_key = f"{text[:200]}||{(context or '')[:100]}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Stage 1: keyword scoring for all intents
        kw_scores = self._keyword_score_all(text)
        kw_top_intent = max(kw_scores, key=kw_scores.__getitem__) if kw_scores else None
        kw_top_conf = kw_scores.get(kw_top_intent, 0.0) if kw_top_intent else 0.0

        stage = "keyword"
        fused_scores: Dict[IntentType, float] = dict(kw_scores)

        if kw_top_conf < self.KEYWORD_THRESHOLD:
            # Stage 2: semantic scoring
            stage = "semantic"
            try:
                sem_scores = await self._semantic_score_all(text)
            except Exception:
                sem_scores = {}

            if sem_scores:
                # Merge: weighted blend where semantic gets more weight when
                # keyword confidence is low
                kw_weight = 0.35
                sem_weight = 0.65
                all_intents = set(kw_scores) | set(sem_scores)
                fused_scores = {
                    intent: (
                        kw_weight * kw_scores.get(intent, 0.0)
                        + sem_weight * sem_scores.get(intent, 0.0)
                    )
                    for intent in all_intents
                }

                # Agreement boost: if both stages pick the same top intent
                sem_top = max(sem_scores, key=sem_scores.__getitem__) if sem_scores else None
                if sem_top and sem_top == kw_top_intent:
                    fused_scores[sem_top] = min(1.0, fused_scores[sem_top] + self.AGREEMENT_BOOST)
                    stage = "fused"

        # Apply context-aware prior shift
        if context:
            fused_scores = self._apply_context_prior(fused_scores, context)

        if not fused_scores:
            intent_type, confidence = IntentType.CONVERSATION, 0.50
        else:
            intent_type = max(fused_scores, key=fused_scores.__getitem__)
            confidence = fused_scores[intent_type]

        # Build secondary intents (other high-confidence intents ≠ primary)
        secondary = [
            IntentScore(intent_type=it, confidence=round(sc, 3))
            for it, sc in sorted(fused_scores.items(), key=lambda x: -x[1])
            if it != intent_type and sc >= self.MIN_SECONDARY_CONFIDENCE
        ][:3]  # cap at 3 secondary intents

        # Subtype extraction
        subtype = _extract_subtype(text, intent_type)

        # Determine routing flags
        _EXPERT_INTENTS = {
            IntentType.CODING, IntentType.MATH, IntentType.TRANSLATION,
            IntentType.WRITING, IntentType.DATA_ANALYSIS, IntentType.SECURITY,
            IntentType.PLANNING, IntentType.EDUCATION, IntentType.CREATIVE,
        }
        _RAG_INTENTS = {
            IntentType.RETRIEVAL, IntentType.REASONING, IntentType.SUMMARIZATION,
            IntentType.DATA_ANALYSIS,
        }

        intent = Intent(
            intent_type=intent_type,
            confidence=round(confidence, 3),
            subtype=subtype,
            keywords=[],
            requires_expert=intent_type in _EXPERT_INTENTS,
            requires_rag=intent_type in _RAG_INTENTS,
            secondary_intents=secondary,
            classification_stage=stage,
        )

        await self._cache.set(cache_key, intent)
        log.debug(
            "Intent classified",
            intent=intent_type.value,
            confidence=f"{confidence:.2f}",
            stage=stage,
            subtype=subtype,
            secondary=[s.intent_type.value for s in secondary],
        )
        return intent
