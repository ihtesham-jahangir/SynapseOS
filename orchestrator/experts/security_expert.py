"""Security expert – vulnerability analysis, secure code review, and defense guidance."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import Intent
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_security.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class SecurityExpert(ExpertBase):
    expert_type = "security"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Review this login code for vulnerabilities:\n```python\nquery = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n```",
            "assistant": """**Critical: SQL Injection (CWE-89)**

**Vulnerability:** String interpolation allows an attacker to break out of the SQL string.

**Attack example:** `username = "admin'--"` produces:
```sql
SELECT * FROM users WHERE username='admin'--' AND password='...'
```
The `--` comments out the password check — attacker logs in as admin with any password.

**Remediation:**
```python
# Use parameterized queries (never string interpolation)
cursor.execute(
    "SELECT * FROM users WHERE username = %s AND password = %s",
    (username, hashed_password)
)
```

**Additional controls:**
- Hash passwords with bcrypt/argon2 (never store plaintext or MD5)
- Add rate limiting (≤5 attempts/min per IP)
- Log failed authentication attempts""",
        },
        {
            "user": "How do I implement JWT authentication securely?",
            "assistant": """**Secure JWT Implementation Checklist**

**1. Algorithm:** Use RS256 (asymmetric) or HS256 with a 256-bit random secret. Never use `alg: none`.
```python
import jwt, secrets
SECRET = secrets.token_hex(32)  # Generate once, store in env var

# Issue token
token = jwt.encode(
    {"sub": user_id, "exp": datetime.utcnow() + timedelta(hours=1)},
    SECRET, algorithm="HS256"
)

# Verify (always specify allowed algorithms)
payload = jwt.decode(token, SECRET, algorithms=["HS256"])
```

**2. Claims to always include:**
- `exp` — expiry (short: 15min for access tokens)
- `iat` — issued-at (detect clock-skew attacks)
- `sub` — subject (user ID, not username)

**3. Storage:** `httpOnly` cookie (not localStorage — prevents XSS theft)

**4. Refresh tokens:** Long-lived (7d), stored in DB, revocable. Rotate on use.

**5. Do NOT:** Store sensitive data in payload (it's base64, not encrypted).""",
        },
        {
            "user": "What are the OWASP Top 10 vulnerabilities I should check for?",
            "assistant": """**OWASP Top 10 (2021) — Quick Reference**

| # | Vulnerability | CWE | Quick Test |
|---|---|---|---|
| A01 | Broken Access Control | CWE-200 | Can user A access user B's data? |
| A02 | Cryptographic Failures | CWE-327 | Is data encrypted in transit AND at rest? |
| A03 | Injection (SQLi, XSS, etc.) | CWE-89 | Are all inputs parameterized/escaped? |
| A04 | Insecure Design | CWE-209 | Is threat modeling done at design time? |
| A05 | Security Misconfiguration | CWE-16 | Default creds? Debug mode in prod? |
| A06 | Vulnerable Components | CWE-1035 | Are dependencies audited (`pip audit`, `npm audit`)? |
| A07 | Auth & Session Failures | CWE-287 | Brute-force protection? Session fixation? |
| A08 | Software Integrity Failures | CWE-494 | CI/CD pipeline integrity? Supply chain? |
| A09 | Logging & Monitoring Failures | CWE-778 | Are auth failures, SQLi attempts logged? |
| A10 | SSRF | CWE-918 | Do user-supplied URLs get fetched server-side? |

**Start with A01 and A03** — highest exploitability in most web apps.""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "review" in q or "audit" in q or "check" in q:
            prompt += "\n\nFocus: Code security review. Triage by severity (Critical/High/Medium/Low). Provide line-specific findings."
        elif "implement" in q or "how to" in q or "best practice" in q:
            prompt += "\n\nFocus: Secure implementation guidance. Show correct code alongside the explanation."
        elif "pentest" in q or "exploit" in q or "test" in q:
            prompt += "\n\nFocus: Authorized testing methodology. Describe the test approach and what evidence to collect."
        return prompt
