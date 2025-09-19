import requests

from app.llm_server import MODEL


def complete(messages, stream=False):
    r = requests.post("http://localhost:11434/api/chat",
                      json={"model": MODEL, "messages": messages, "stream": stream},
                      timeout=600, stream=stream)
    if not stream:
        return r.json()["message"]["content"]
    else:
        for line in r.iter_lines():
            if line:
                yield line.decode("utf-8")
