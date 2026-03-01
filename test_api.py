"""Quick test of Moonshot Kimi K2.5 API via NVIDIA and Moonshot endpoints."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("OPENAI_API_KEY", "")
print(f"Key: {key[:15]}...")
print()

# Test 1: NVIDIA hosted API (where nvapi- keys work)
print("=== Test 1: NVIDIA integrate.api.nvidia.com ===")
try:
    resp = httpx.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "moonshotai/kimi-k2.5",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 256,
        },
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        d = resp.json()
        c = d["choices"][0]["message"]
        print(f"Content: {c.get('content')!r}")
        print(f"Reasoning: {(c.get('reasoning_content') or c.get('reasoning', ''))[:150]!r}")
        print(f"Usage: {d.get('usage')}")
    else:
        print(resp.text[:400])
except Exception as e:
    print(f"Error: {e}")

print()

# Test 2: Moonshot native API (where sk- keys work)
print("=== Test 2: Moonshot api.moonshot.cn ===")
try:
    resp2 = httpx.post(
        "https://api.moonshot.cn/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "kimi-k2.5",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 256,
        },
        timeout=30,
    )
    print(f"Status: {resp2.status_code}")
    if resp2.status_code == 200:
        d2 = resp2.json()
        c2 = d2["choices"][0]["message"]
        print(f"Content: {c2.get('content')!r}")
    else:
        print(resp2.text[:400])
except Exception as e:
    print(f"Error: {e}")
