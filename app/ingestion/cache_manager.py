"""
Cache Manager for PharmaRAG — JSON file cache
"""
import os, json, hashlib, time

CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TTL = 86400  # 24 hours

def _path(query: str) -> str:
    key = hashlib.sha1(query.lower().encode()).hexdigest()
    return os.path.join(CACHE_DIR, key + ".json")

def get_cache(query: str):
    p = _path(query)
    if not os.path.exists(p): return None
    if (time.time() - os.path.getmtime(p)) > TTL: return None
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return None

def set_cache(query: str, data):
    try:
        with open(_path(query), "w") as f: json.dump(data, f)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")
