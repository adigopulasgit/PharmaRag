# app/llm/context_builders.py

def format_admet(preds):
    return "ADMET Predictions:\n" + "\n".join([f"- {k}: {v:.3f}" for k, v in preds.items()])

def format_corpus_hits(hits):
    out = ["Retrieved Evidence:"]
    for h in hits:
        out.append(f"[{h.get('doc_id', '?')}] {h.get('text', str(h))[:300]}... (source: {h.get('source','')})")
    return "\n".join(out)
