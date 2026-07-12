# Policy Intelligence Engine

The Policy Intelligence Engine analyzes enterprise policy documents and returns normalized policies, extracted obligations, and deterministic policy findings.

## Public API

```python
from engine.pipeline import analyze_policy_documents

result = analyze_policy_documents(documents)