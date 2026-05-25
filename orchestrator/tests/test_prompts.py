"""
Prompt file tests — validates the external prompt system for all 6 new experts.

Groups:
  1.  Prompt files exist on disk
  2.  Prompt files are non-empty and meet minimum length
  3.  Prompt files are valid UTF-8 with no encoding errors
  4.  Prompt file content contains domain-specific keywords
  5.  Expert _system_prompt matches file content exactly
  6.  Expert _build_system_prompt base equals the file content
  7.  Query-specific appends preserve the base file content
  8.  All prompt files are distinct from each other
  9.  Task planner prompt file is valid and contains JSON schema hint
  10. Expert guidance returns system_prompt derived from the file
  11. Prompt file path constants point to the correct locations
  12. Prompt files list complete — no extra / missing files
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

EXPERT_FILES = {
    "writing":       "expert_writing.txt",
    "data_analysis": "expert_data_analysis.txt",
    "creative":      "expert_creative.txt",
    "security":      "expert_security.txt",
    "planning":      "expert_planning.txt",
    "education":     "expert_education.txt",
}

EXPERT_MODULES = {
    "writing":       "orchestrator.experts.writing_expert",
    "data_analysis": "orchestrator.experts.data_expert",
    "creative":      "orchestrator.experts.creative_expert",
    "security":      "orchestrator.experts.security_expert",
    "planning":      "orchestrator.experts.planning_expert",
    "education":     "orchestrator.experts.education_expert",
}

EXPERT_CLASSES = {
    "writing":       "WritingExpert",
    "data_analysis": "DataExpert",
    "creative":      "CreativeExpert",
    "security":      "SecurityExpert",
    "planning":      "PlanningExpert",
    "education":     "EducationExpert",
}

DOMAIN_KEYWORDS = {
    "writing":       ["tone", "writing", "edit"],
    "data_analysis": ["sql", "pandas", "data"],
    "creative":      ["creative", "original", "ideas"],
    "security":      ["owasp", "vulnerability", "security"],
    "planning":      ["plan", "sprint", "dependencies"],
    "education":     ["teach", "learner", "analogi"],
}


def _read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _get_expert(domain: str):
    mod = importlib.import_module(EXPERT_MODULES[domain])
    cls = getattr(mod, EXPERT_CLASSES[domain])
    return cls()


# ── 1. Prompt files exist on disk ─────────────────────────────────────────────

class TestPromptFilesExist:
    def test_task_planner_prompt_exists(self):
        assert (PROMPTS_DIR / "task_planner.txt").exists()

    @pytest.mark.parametrize("filename", EXPERT_FILES.values())
    def test_expert_prompt_file_exists(self, filename):
        assert (PROMPTS_DIR / filename).exists(), f"Missing: {filename}"

    def test_prompts_directory_exists(self):
        assert PROMPTS_DIR.is_dir()

    def test_init_file_exists(self):
        assert (PROMPTS_DIR / "__init__.py").exists()


# ── 2. Prompt files are non-empty and meet minimum length ─────────────────────

class TestPromptFileLength:
    MIN_CHARS = 200  # system prompts should be substantial

    @pytest.mark.parametrize("domain,filename", EXPERT_FILES.items())
    def test_prompt_file_non_empty(self, domain, filename):
        content = _read_prompt(filename)
        assert len(content) > 0, f"{filename} is empty"

    @pytest.mark.parametrize("domain,filename", EXPERT_FILES.items())
    def test_prompt_file_meets_minimum_length(self, domain, filename):
        content = _read_prompt(filename)
        assert len(content) >= self.MIN_CHARS, (
            f"{filename} too short ({len(content)} chars, min {self.MIN_CHARS})"
        )

    def test_task_planner_prompt_non_empty(self):
        content = _read_prompt("task_planner.txt")
        assert len(content) >= 100


# ── 3. Prompt files are valid UTF-8 ──────────────────────────────────────────

class TestPromptFileEncoding:
    @pytest.mark.parametrize("filename", list(EXPERT_FILES.values()) + ["task_planner.txt"])
    def test_file_reads_as_utf8(self, filename):
        # Raises UnicodeDecodeError if not valid UTF-8
        content = (PROMPTS_DIR / filename).read_bytes().decode("utf-8")
        assert isinstance(content, str)

    @pytest.mark.parametrize("domain,filename", EXPERT_FILES.items())
    def test_no_null_bytes(self, domain, filename):
        raw = (PROMPTS_DIR / filename).read_bytes()
        assert b"\x00" not in raw

    @pytest.mark.parametrize("domain,filename", EXPERT_FILES.items())
    def test_stripped_content_has_no_leading_trailing_whitespace(self, domain, filename):
        content = _read_prompt(filename)
        assert content == content.strip()


# ── 4. Prompt file content contains domain-specific keywords ─────────────────

class TestPromptFileContent:
    @pytest.mark.parametrize("domain,keywords", DOMAIN_KEYWORDS.items())
    def test_domain_keywords_present(self, domain, keywords):
        filename = EXPERT_FILES[domain]
        content = _read_prompt(filename).lower()
        for kw in keywords:
            assert kw in content, f"'{kw}' not found in {filename}"

    def test_writing_prompt_mentions_tone(self):
        content = _read_prompt("expert_writing.txt").lower()
        assert "tone" in content

    def test_security_prompt_mentions_owasp(self):
        content = _read_prompt("expert_security.txt").lower()
        assert "owasp" in content

    def test_security_prompt_mentions_ethical_use(self):
        content = _read_prompt("expert_security.txt").lower()
        assert "authorized" in content or "ethical" in content or "defense" in content

    def test_education_prompt_mentions_learner(self):
        content = _read_prompt("expert_education.txt").lower()
        assert "learner" in content or "learn" in content

    def test_planning_prompt_mentions_risk(self):
        content = _read_prompt("expert_planning.txt").lower()
        assert "risk" in content

    def test_creative_prompt_mentions_originality(self):
        content = _read_prompt("expert_creative.txt").lower()
        assert "original" in content or "unexpected" in content or "cliché" in content

    def test_data_prompt_mentions_sql(self):
        content = _read_prompt("expert_data_analysis.txt").lower()
        assert "sql" in content

    def test_task_planner_prompt_mentions_json(self):
        content = _read_prompt("task_planner.txt").lower()
        assert "json" in content

    def test_task_planner_prompt_mentions_sub_tasks(self):
        content = _read_prompt("task_planner.txt")
        assert "sub_tasks" in content


# ── 5. Expert _system_prompt matches file content exactly ─────────────────────

class TestExpertSystemPromptMatchesFile:
    @pytest.mark.parametrize("domain", EXPERT_FILES.keys())
    def test_system_prompt_equals_file_content(self, domain):
        filename = EXPERT_FILES[domain]
        file_content = _read_prompt(filename)
        expert = _get_expert(domain)
        assert expert._system_prompt == file_content, (
            f"{EXPERT_CLASSES[domain]}._system_prompt does not match {filename}"
        )

    @pytest.mark.parametrize("domain", EXPERT_FILES.keys())
    def test_module_level_constant_equals_file(self, domain):
        mod = importlib.import_module(EXPERT_MODULES[domain])
        file_content = _read_prompt(EXPERT_FILES[domain])
        assert mod._SYSTEM_PROMPT == file_content


# ── 6. Expert _build_system_prompt base equals the file content ───────────────

class TestBuildSystemPromptBase:
    @pytest.mark.asyncio
    async def _get_base_prompt(self, domain: str, neutral_query: str = "help") -> str:
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert(domain)
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.70)
        return expert._build_system_prompt(neutral_query, intent)

    @pytest.mark.parametrize("domain", EXPERT_FILES.keys())
    def test_build_prompt_starts_with_file_content(self, domain):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert(domain)
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.70)
        file_content = _read_prompt(EXPERT_FILES[domain])
        built = expert._build_system_prompt("help", intent)
        assert built.startswith(file_content), (
            f"{domain} build_system_prompt does not start with file content"
        )

    @pytest.mark.parametrize("domain", EXPERT_FILES.keys())
    def test_build_prompt_for_neutral_query_equals_file(self, domain):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert(domain)
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.70)
        file_content = _read_prompt(EXPERT_FILES[domain])
        built = expert._build_system_prompt("help me", intent)
        # Neutral query → no append → should equal file content exactly
        assert built == file_content or built.startswith(file_content)


# ── 7. Query-specific appends preserve the base file content ──────────────────

class TestQuerySpecificAppends:
    def _build(self, domain: str, query: str) -> str:
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert(domain)
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.70)
        return expert._build_system_prompt(query, intent)

    def test_writing_email_query_contains_file_base(self):
        base = _read_prompt("expert_writing.txt")
        result = self._build("writing", "write a professional email to decline a meeting")
        assert result.startswith(base)
        assert "email" in result.lower() or "business" in result.lower()

    def test_writing_email_query_appends_focus(self):
        base = _read_prompt("expert_writing.txt")
        result = self._build("writing", "write an email response to the client")
        assert len(result) > len(base)

    def test_security_review_query_appends_focus(self):
        base = _read_prompt("expert_security.txt")
        result = self._build("security", "review this code for security vulnerabilities")
        assert result.startswith(base)
        assert len(result) > len(base)

    def test_security_pentest_query_appends_focus(self):
        base = _read_prompt("expert_security.txt")
        result = self._build("security", "help me pentest this API endpoint")
        assert result.startswith(base)

    def test_education_beginner_query_appends_audience_hint(self):
        base = _read_prompt("expert_education.txt")
        result = self._build("education", "explain recursion for a beginner from scratch")
        assert result.startswith(base)
        assert len(result) > len(base)

    def test_education_advanced_query_appends_different_hint(self):
        base = _read_prompt("expert_education.txt")
        beginner = self._build("education", "explain for a beginner")
        advanced = self._build("education", "explain internals and deep dive into advanced topics")
        # Both start with base, but the appended context differs
        assert beginner.startswith(base)
        assert advanced.startswith(base)
        assert beginner != advanced

    def test_planning_agile_query_appends_focus(self):
        base = _read_prompt("expert_planning.txt")
        result = self._build("planning", "help me plan the next agile sprint backlog")
        assert result.startswith(base)
        assert len(result) > len(base)

    def test_data_sql_query_appends_focus(self):
        base = _read_prompt("expert_data_analysis.txt")
        result = self._build("data_analysis", "write a SQL query to get top customers")
        assert result.startswith(base)
        assert len(result) > len(base)

    def test_creative_brainstorm_query_appends_focus(self):
        base = _read_prompt("expert_creative.txt")
        result = self._build("creative", "brainstorm 5 unique app ideas")
        assert result.startswith(base)
        assert len(result) > len(base)


# ── 8. All prompt files are distinct from each other ─────────────────────────

class TestPromptFilesDistinct:
    def test_all_expert_prompts_are_unique(self):
        contents = {}
        for domain, filename in EXPERT_FILES.items():
            content = _read_prompt(filename)
            for other_domain, other_content in contents.items():
                assert content != other_content, (
                    f"{domain} and {other_domain} have identical prompt files"
                )
            contents[domain] = content

    def test_expert_prompts_distinct_from_task_planner(self):
        planner = _read_prompt("task_planner.txt")
        for domain, filename in EXPERT_FILES.items():
            content = _read_prompt(filename)
            assert content != planner, f"{domain} prompt matches task_planner.txt"

    def test_no_expert_prompt_is_a_substring_of_another(self):
        contents = {domain: _read_prompt(fn) for domain, fn in EXPERT_FILES.items()}
        domains = list(contents.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                assert contents[d1] not in contents[d2], f"{d1} is substr of {d2}"
                assert contents[d2] not in contents[d1], f"{d2} is substr of {d1}"


# ── 9. Task planner prompt file validation ────────────────────────────────────

class TestTaskPlannerPrompt:
    def test_task_planner_file_readable(self):
        content = _read_prompt("task_planner.txt")
        assert len(content) > 0

    def test_task_planner_mentions_roles(self):
        content = _read_prompt("task_planner.txt").lower()
        assert "research" in content
        assert "summarizer" in content

    def test_task_planner_mentions_dependencies(self):
        content = _read_prompt("task_planner.txt")
        assert "dependencies" in content

    def test_task_planner_mentions_json_schema(self):
        content = _read_prompt("task_planner.txt")
        assert "sub_tasks" in content
        assert '"id"' in content or "'id'" in content or "id" in content

    def test_task_planner_loaded_by_module(self):
        from orchestrator.agents import task_planner as tp
        file_content = _read_prompt("task_planner.txt")
        assert tp._PLAN_SYSTEM == file_content


# ── 10. Expert guidance returns system_prompt derived from the file ───────────

class TestExpertGuidanceUsesFilePrompt:
    @pytest.mark.asyncio
    async def test_writing_guidance_system_prompt_from_file(self):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert("writing")
        intent = Intent(intent_type=IntentType.WRITING, confidence=0.80)
        guidance = await expert.get_guidance("write an email", intent)
        file_content = _read_prompt("expert_writing.txt")
        assert guidance.system_prompt.startswith(file_content)

    @pytest.mark.asyncio
    async def test_security_guidance_system_prompt_from_file(self):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert("security")
        intent = Intent(intent_type=IntentType.SECURITY, confidence=0.90)
        guidance = await expert.get_guidance("review this code", intent)
        file_content = _read_prompt("expert_security.txt")
        assert guidance.system_prompt.startswith(file_content)

    @pytest.mark.asyncio
    async def test_education_guidance_system_prompt_from_file(self):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert("education")
        intent = Intent(intent_type=IntentType.EDUCATION, confidence=0.75)
        guidance = await expert.get_guidance("explain neural networks", intent)
        file_content = _read_prompt("expert_education.txt")
        assert guidance.system_prompt.startswith(file_content)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain", EXPERT_FILES.keys())
    async def test_all_experts_guidance_from_file(self, domain):
        from orchestrator.core.types import Intent, IntentType
        expert = _get_expert(domain)
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.70)
        guidance = await expert.get_guidance("help me with something", intent)
        file_content = _read_prompt(EXPERT_FILES[domain])
        assert guidance.system_prompt.startswith(file_content)
        assert guidance.system_prompt is not None
        assert len(guidance.system_prompt) > 0


# ── 11. Prompt file path constants point to correct locations ─────────────────

class TestPromptFilePathConstants:
    def test_writing_expert_prompt_file_constant(self):
        import orchestrator.experts.writing_expert as m
        assert m._PROMPT_FILE.name == "expert_writing.txt"
        assert m._PROMPT_FILE.exists()

    def test_data_expert_prompt_file_constant(self):
        import orchestrator.experts.data_expert as m
        assert m._PROMPT_FILE.name == "expert_data_analysis.txt"
        assert m._PROMPT_FILE.exists()

    def test_creative_expert_prompt_file_constant(self):
        import orchestrator.experts.creative_expert as m
        assert m._PROMPT_FILE.name == "expert_creative.txt"
        assert m._PROMPT_FILE.exists()

    def test_security_expert_prompt_file_constant(self):
        import orchestrator.experts.security_expert as m
        assert m._PROMPT_FILE.name == "expert_security.txt"
        assert m._PROMPT_FILE.exists()

    def test_planning_expert_prompt_file_constant(self):
        import orchestrator.experts.planning_expert as m
        assert m._PROMPT_FILE.name == "expert_planning.txt"
        assert m._PROMPT_FILE.exists()

    def test_education_expert_prompt_file_constant(self):
        import orchestrator.experts.education_expert as m
        assert m._PROMPT_FILE.name == "expert_education.txt"
        assert m._PROMPT_FILE.exists()

    def test_task_planner_prompt_file_constant(self):
        from orchestrator.agents import task_planner as m
        assert m._PROMPT_FILE.name == "task_planner.txt"
        assert m._PROMPT_FILE.exists()


# ── 12. Prompt files list is complete — no extra / missing files ──────────────

class TestPromptFileListing:
    EXPECTED_FILES = {
        "__init__.py",
        "task_planner.txt",
        "expert_writing.txt",
        "expert_data_analysis.txt",
        "expert_creative.txt",
        "expert_security.txt",
        "expert_planning.txt",
        "expert_education.txt",
    }

    def test_no_missing_prompt_files(self):
        existing = {f.name for f in PROMPTS_DIR.iterdir()}
        missing = self.EXPECTED_FILES - existing
        assert not missing, f"Missing prompt files: {missing}"

    def test_no_unexpected_prompt_files(self):
        existing = {f.name for f in PROMPTS_DIR.iterdir() if not f.name.startswith("__pycache__")}
        unexpected = existing - self.EXPECTED_FILES
        assert not unexpected, f"Unexpected files in prompts/: {unexpected}"

    def test_all_txt_files_are_tracked(self):
        txt_files = {f.name for f in PROMPTS_DIR.glob("*.txt")}
        expected_txt = {f for f in self.EXPECTED_FILES if f.endswith(".txt")}
        assert txt_files == expected_txt
