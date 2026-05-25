"""Data analysis expert – SQL, pandas, visualization, and ML pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import Intent
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_data_analysis.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class DataExpert(ExpertBase):
    expert_type = "data_analysis"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "SQL: find the top 5 customers by total revenue in the last 30 days",
            "assistant": """```sql
SELECT
    c.customer_id,
    c.name,
    SUM(o.amount) AS total_revenue
FROM orders o
JOIN customers c USING (customer_id)
WHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY c.customer_id, c.name
ORDER BY total_revenue DESC
LIMIT 5;
```
*Uses USING for cleaner join syntax; INTERVAL for portable date arithmetic.*""",
        },
        {
            "user": "How do I calculate a 7-day rolling average in pandas?",
            "assistant": """```python
import pandas as pd

df['rolling_7d_avg'] = (
    df['value']
    .rolling(window=7, min_periods=1)
    .mean()
)
```
- `min_periods=1` returns a value even for the first 6 rows (partial window).
- Sort by date first: `df = df.sort_values('date')` if not already ordered.""",
        },
        {
            "user": "Train a simple classifier on the iris dataset",
            "assistant": """```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
```
**Baseline accuracy to beat:** 33% (random) → RF typically achieves 96%+.""",
        },
        {
            "user": "Find duplicate rows in a pandas DataFrame",
            "assistant": """```python
# View duplicates (all occurrences)
dupes = df[df.duplicated(keep=False)]

# Count per group
print(df.duplicated().sum(), "duplicate rows found")

# Remove duplicates, keep first occurrence
df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
```
Pass `subset=['col1', 'col2']` to check duplicates on specific columns only.""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "sql" in q or "select" in q or "query" in q:
            prompt += "\n\nFocus: SQL. Write ANSI-compatible SQL with a brief explanation of the query logic."
        elif "pandas" in q or "dataframe" in q or "csv" in q:
            prompt += "\n\nFocus: pandas/Python. Use modern pandas patterns (method chaining where readable)."
        elif "visuali" in q or "chart" in q or "plot" in q or "graph" in q:
            prompt += "\n\nFocus: Data visualization. Include complete, runnable matplotlib/seaborn code."
        elif "machine learning" in q or "model" in q or "train" in q or "predict" in q:
            prompt += "\n\nFocus: ML. Include data prep, training, evaluation, and interpretation steps."
        return prompt
