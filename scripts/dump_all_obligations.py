import sys
from pathlib import Path

# Resolve repo root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.pipeline import analyze_policy_documents
from engine.candidates.candidate_generator import cosine_similarity

_POLICY_DIR = _REPO_ROOT / "sample_data" / "problem_11" / "policies"

def main():
    policy_paths = sorted(_POLICY_DIR.glob("*.md"))
    result = analyze_policy_documents(policy_paths, llm_provider=None)
    obligations = result.obligations
    policies = result.policies

    # Group obligations by policy filename
    policy_id_to_file = {p.policy_id: Path(p.source_file).name for p in policies}
    
    # Compute corpus frequencies
    import re
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    normalized_texts = {obl.obligation_id: normalize_text(obl.sentence_text) for obl in obligations}
    obl_to_freq = {}
    for obl_a in obligations:
        matching_policies = {obl_a.policy_id}
        norm_a = normalized_texts[obl_a.obligation_id]
        emb_a = obl_a.embedding
        for obl_b in obligations:
            if obl_b.policy_id == obl_a.policy_id:
                continue
            if obl_b.policy_id in matching_policies:
                continue
            norm_b = normalized_texts[obl_b.obligation_id]
            if norm_a == norm_b:
                matching_policies.add(obl_b.policy_id)
                continue
            if emb_a and obl_b.embedding:
                sim = cosine_similarity(emb_a, obl_b.embedding)
                if sim >= 0.95:
                    matching_policies.add(obl_b.policy_id)
        obl_to_freq[obl_a.obligation_id] = len(matching_policies)

    print(f"Total obligations: {len(obligations)}")
    print(f"{'Policy':<15} | {'Topic':<15} | {'Freq':<4} | {'Action':<12} | {'Object':<20} | Text")
    print("-" * 120)

    for p in policies:
        fname = Path(p.source_file).name
        obls = [o for o in obligations if o.policy_id == p.policy_id]
        for o in obls:
            freq = obl_to_freq[o.obligation_id]
            action = o.action or ""
            obj = o.object or ""
            print(f"{fname:<15} | {str(o.topic):<15} | {freq:<4} | {action:<12} | {obj:<20} | {o.sentence_text}")

if __name__ == "__main__":
    main()
