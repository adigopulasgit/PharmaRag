import os, json, hashlib, time

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TTL = 86400  # 24h

def _key(query): 
    return hashlib.sha1(query.encode()).hexdigest() + ".json"

def get_cache(query):
    path = os.path.join(CACHE_DIR, _key(query))
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < TTL:
        with open(path) as f: 
            return json.load(f)
    return None

def set_cache(query, data):
    path = os.path.join(CACHE_DIR, _key(query))
    with open(path, "w") as f: 
        json.dump(data, f)
