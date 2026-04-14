"""
ingest.py — Ingest Reddit threads into Actian VectorAI DB

Fetches threads from configured subreddits via snoowrap, embeds them using
the local embedding server, and stores them in VectorAI DB for semantic search.

Usage:
    pip install actian-vectorai snoowrap praw
    python ingest.py

Environment:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    VECTORAI_DB_URL (default http://localhost:27832)
    EMBEDDING_SERVER_URL (default http://localhost:11434)
"""

import os
import json
import time
import requests
from praw import Reddit

# ─── Config ──────────────────────────────────────────────────────────

VECTORAI_URL = os.environ.get("VECTORAI_DB_URL", "http://localhost:27832")
EMBEDDING_URL = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:11434")
COLLECTION = os.environ.get("VECTORAI_COLLECTION", "reddit_threads")
SUBREDDITS = os.environ.get("SUBREDDITS", "smallbusiness,startups,Entrepreneur,SaaS").split(",")
LIMIT = int(os.environ.get("INGEST_LIMIT", "100"))

# ─── Reddit Client ───────────────────────────────────────────────────

reddit = Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT", "semantiq-ingest/1.0"),
)

# ─── Functions ───────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using the local embedding server."""
    response = requests.post(
        f"{EMBEDDING_URL}/api/embed-batch",
        json={"texts": texts},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def init_collection():
    """Create the VectorAI DB collection if it doesn't exist."""
    try:
        response = requests.post(
            f"{VECTORAI_URL}/api/collections",
            json={
                "name": COLLECTION,
                "dimension": 384,
                "metadata_schema": {
                    "subreddit": "string",
                    "title": "string",
                    "author": "string",
                    "score": "integer",
                    "num_comments": "integer",
                    "created_at_reddit": "integer",
                    "url": "string",
                },
            },
        )
        if response.status_code != 409:
            response.raise_for_status()
        print(f"Collection '{COLLECTION}' ready.")
    except Exception as e:
        print(f"Collection init warning: {e}")


def fetch_threads() -> list[dict]:
    """Fetch threads from configured subreddits."""
    threads = []
    for sub in SUBREDDITS:
        try:
            for submission in reddit.subreddit(sub).new(limit=LIMIT):
                threads.append({
                    "id": submission.id,
                    "subreddit": str(submission.subreddit),
                    "title": submission.title,
                    "selftext": (submission.selftext or "")[:5000],
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_at_reddit": int(submission.created_utc),
                    "url": submission.url,
                })
            print(f"  r/{sub}: fetched {LIMIT} threads")
        except Exception as e:
            print(f"  r/{sub}: error - {e}")
    return threads


def upsert_to_vectorai(threads: list[dict], embeddings: list[list[float]]):
    """Upsert threads into VectorAI DB."""
    records = []
    for thread, embedding in zip(threads, embeddings):
        if not embedding:
            continue
        records.append({
            "id": thread["id"],
            "vector": embedding,
            "metadata": {
                "subreddit": thread["subreddit"],
                "title": thread["title"],
                "author": thread["author"],
                "score": thread["score"],
                "num_comments": thread["num_comments"],
                "created_at_reddit": thread["created_at_reddit"],
                "url": thread["url"],
            },
        })

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        response = requests.post(
            f"{VECTORAI_URL}/api/collections/{COLLECTION}/upsert",
            json={
                "ids": [r["id"] for r in batch],
                "vectors": [r["vector"] for r in batch],
                "metadata": [r["metadata"] for r in batch],
            },
            timeout=30,
        )
        response.raise_for_status()
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} threads)")

    return len(records)


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Test semantic search against VectorAI DB."""
    # Embed query
    response = requests.post(
        f"{EMBEDDING_URL}/api/embed",
        json={"text": query},
        timeout=30,
    )
    response.raise_for_status()
    query_vector = response.json()["embedding"]

    # Search
    response = requests.post(
        f"{VECTORAI_URL}/api/collections/{COLLECTION}/search",
        json={
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=== SemantiQ Ingestion Pipeline ===")
    print(f"Subreddits: {SUBREDDITS}")
    print(f"Limit: {LIMIT} per subreddit")
    print(f"VectorAI DB: {VECTORAI_URL}")
    print(f"Embedding server: {EMBEDDING_URL}")
    print()

    # Init
    init_collection()

    # Fetch
    print("Fetching threads from Reddit...")
    start = time.time()
    threads = fetch_threads()
    print(f"Fetched {len(threads)} total threads in {time.time() - start:.1f}s")
    print()

    # Embed
    print("Embedding threads...")
    start = time.time()
    texts = [f"{t['title']} {t['selftext']}" for t in threads]
    embeddings = embed_texts(texts)
    print(f"Embedded {len(embeddings)} threads in {time.time() - start:.1f}s")
    print()

    # Upsert
    print("Upserting into VectorAI DB...")
    start = time.time()
    upserted = upsert_to_vectorai(threads, embeddings)
    print(f"Upserted {upserted} threads in {time.time() - start:.1f}s")
    print()

    # Test search
    print("Testing semantic search...")
    results = semantic_search("best tool for managing client invoices", top_k=5)
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results):
        title = r.get("metadata", {}).get("title", "N/A")
        sim = r.get("score", r.get("distance", "N/A"))
        print(f"  {i+1}. [{sim}] {title[:80]}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
