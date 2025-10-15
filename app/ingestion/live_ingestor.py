from app.ingestion import web_ingestor as wi
from app.ingestion.cache_manager import get_cache, set_cache

def get_live_results(query: str, max_items: int = 25):
    cached = get_cache(query)
    if cached: 
        return cached

    results = []
    results.extend(wi.query_pubchem_by_name(query, 10))
    results.extend(wi.query_bindingdb_by_uniprot(wi.find_uniprot_id(query) or "", 10))
    results.extend(wi.query_chembl_target(wi.find_chembl_id(query) or ""))
    if any(k in query.lower() for k in wi.query_tdc_summary.__annotations__):
        results.extend(wi.query_tdc_summary(query))
    
    dedup = {r["text"]: r for r in results}.values()
    final = list(dedup)[:max_items]
    set_cache(query, final)
    return final
