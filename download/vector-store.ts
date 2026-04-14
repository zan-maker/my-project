/**
 * vector-store.ts — Actian VectorAI DB client wrapper
 *
 * Provides a clean abstraction over Actian VectorAI DB for:
 *   - Upserting thread vectors with metadata
 *   - Semantic similarity search with metadata filters
 *   - Hybrid fusion search (vector + engagement scoring)
 *
 * Setup:
 *   1. Run VectorAI DB: docker compose up vectorai-db
 *   2. Install Python client: pip install actian-vectorai
 *   3. Install embeddings model: pip install sentence-transformers
 *   4. Start embedding server: python src/lib/embedding-server.py
 */

import type { RedditPost } from './reddit';

// ─── Types ───────────────────────────────────────────────────────────

export interface ThreadVectorRecord {
  id: string;               // Reddit post ID
  vector: number[];         // Embedding
  subreddit: string;
  title: string;
  selftext: string;
  author: string;
  score: number;            // Reddit upvotes
  numComments: number;
  createdAtReddit: number;  // Unix timestamp
  url: string;
}

export interface SemanticSearchResult {
  id: string;
  title: string;
  subreddit: string;
  selftext: string;
  author: string;
  score: number;
  numComments: number;
  createdAtReddit: number;
  url: string;
  similarity: number;       // 0–1 semantic similarity score
}

export interface VectorSearchOptions {
  queryText: string;                  // Text to embed and search with
  topK?: number;                      // Max results (default 20)
  subredditFilter?: string[];         // Restrict to these subreddits
  minAgeHours?: number;               // Only threads newer than this
  maxAgeHours?: number;               // Only threads older than this
  minComments?: number;               // Minimum engagement threshold
  minScore?: number;                  // Minimum upvote threshold
}

export interface HybridSearchOptions extends VectorSearchOptions {
  engagementWeight?: number;   // Weight for engagement score in fused ranking (0–1, default 0.3)
}

// ─── Configuration ──────────────────────────────────────────────────

const VECTORAI_CONFIG = {
  baseUrl: process.env.VECTORAI_DB_URL || 'http://localhost:27832',
  collectionName: process.env.VECTORAI_COLLECTION || 'reddit_threads',
  embeddingServerUrl: process.env.EMBEDDING_SERVER_URL || 'http://localhost:11434',
};

// ─── Embedding via local model ──────────────────────────────────────

/**
 * Generate embedding vector for text using the local embedding server.
 * The embedding server runs sentence-transformers/all-MiniLM-L6-v2.
 */
export async function embedText(text: string): Promise<number[]> {
  const response = await fetch(`${VECTORAI_CONFIG.embeddingServerUrl}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Embedding failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Batch embed multiple texts in one call (faster than one-by-one).
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const response = await fetch(`${VECTORAI_CONFIG.embeddingServerUrl}/api/embed-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts }),
  });

  if (!response.ok) {
    throw new Error(`Batch embedding failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return data.embeddings;
}

// ─── VectorAI DB Operations ─────────────────────────────────────────

/**
 * Initialize a collection in VectorAI DB with proper schema.
 */
export async function initCollection(): Promise<void> {
  const response = await fetch(`${VECTORAI_CONFIG.baseUrl}/api/collections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: VECTORAI_CONFIG.collectionName,
      dimension: 384, // all-MiniLM-L6-v2 output dimension
      metadata_schema: {
        subreddit: 'string',
        title: 'string',
        author: 'string',
        score: 'integer',
        num_comments: 'integer',
        created_at_reddit: 'integer',
        url: 'string',
      },
    }),
  });

  // Collection might already exist — that's fine
  if (response.status === 409) return;
  if (!response.ok) {
    throw new Error(`Failed to init collection: ${response.status} ${await response.text()}`);
  }
}

/**
 * Upsert thread vectors into VectorAI DB.
 * Uses bulk upsert for efficiency.
 */
export async function upsertThreads(threads: ThreadVectorRecord[]): Promise<void> {
  if (threads.length === 0) return;

  const response = await fetch(
    `${VECTORAI_CONFIG.baseUrl}/api/collections/${VECTORAI_CONFIG.collectionName}/upsert`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ids: threads.map(t => t.id),
        vectors: threads.map(t => t.vector),
        metadata: threads.map(t => ({
          subreddit: t.subreddit,
          title: t.title,
          author: t.author,
          score: t.score,
          num_comments: t.numComments,
          created_at_reddit: t.createdAtReddit,
          url: t.url,
        })),
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Upsert failed: ${response.status} ${await response.text()}`);
  }
}

/**
 * Semantic similarity search in VectorAI DB with metadata filters.
 * This is the "Filtered Search" advanced feature from the hackathon.
 */
export async function semanticSearch(
  options: VectorSearchOptions
): Promise<SemanticSearchResult[]> {
  const {
    queryText,
    topK = 20,
    subredditFilter,
    minAgeHours,
    maxAgeHours,
    minComments = 0,
    minScore = 0,
  } = options;

  // Embed the query text
  const queryVector = await embedText(queryText);

  // Build metadata filters
  const now = Math.floor(Date.now() / 1000);
  const filters: Record<string, any> = {};

  if (subredditFilter && subredditFilter.length > 0) {
    filters.subreddit = { $in: subredditFilter };
  }

  if (maxAgeHours) {
    const minTimestamp = now - (maxAgeHours * 3600);
    filters.created_at_reddit = { $gte: minTimestamp };
  }

  if (minAgeHours) {
    const maxTimestamp = now - (minAgeHours * 3600);
    filters.created_at_reddit = { ...filters.created_at_reddit, $lte: maxTimestamp };
  }

  if (minComments > 0) {
    filters.num_comments = { $gte: minComments };
  }

  if (minScore > 0) {
    filters.score = { $gte: minScore };
  }

  const response = await fetch(
    `${VECTORAI_CONFIG.baseUrl}/api/collections/${VECTORAI_CONFIG.collectionName}/search`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vector: queryVector,
        top_k: topK,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
        include_metadata: true,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Search failed: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();

  return (data.results || []).map((r: any) => ({
    id: r.id,
    title: r.metadata?.title || '',
    subreddit: r.metadata?.subreddit || '',
    selftext: '',
    author: r.metadata?.author || '',
    score: r.metadata?.score || 0,
    numComments: r.metadata?.num_comments || 0,
    createdAtReddit: r.metadata?.created_at_reddit || 0,
    url: r.metadata?.url || '',
    similarity: r.score || r.distance || 0,
  }));
}

/**
 * Hybrid fusion search — combines semantic similarity with engagement metrics.
 * This is the "Hybrid Fusion" advanced feature from the hackathon.
 *
 * Uses Reciprocal Rank Fusion (RRF) to merge:
 *   - Semantic similarity ranking from VectorAI DB
 *   - Engagement score ranking (comments + upvotes + recency)
 */
export async function hybridSearch(
  options: HybridSearchOptions
): Promise<(SemanticSearchResult & { fusedScore: number })[]> {
  const {
    queryText,
    topK = 20,
    engagementWeight = 0.3,
    ...filterOpts
  } = options;

  // Get semantic results (fetch more for better fusion)
  const semanticResults = await semanticSearch({
    ...filterOpts,
    queryText,
    topK: topK * 3,
  });

  // Calculate engagement score for each result
  const now = Math.floor(Date.now() / 1000);
  const engagementResults = semanticResults.map(result => {
    const ageInHours = (now - result.createdAtReddit) / 3600;
    const commentScore = Math.min(Math.log(result.numComments + 1) * 10, 100);
    const upvoteScore = Math.min(Math.log(result.score + 1) * 8, 100);
    const recencyScore = Math.max(0, 100 - (ageInHours / 168) * 100);

    return {
      ...result,
      engagementScore: (commentScore + upvoteScore + recencyScore) / 3,
    };
  });

  // Reciprocal Rank Fusion (RRF)
  // Rank each result by both signals, then fuse
  const k = 60; // RRF constant (standard value)
  const semanticRanked = [...semanticResults].sort((a, b) => b.similarity - a.similarity);
  const engagementRanked = [...engagementResults].sort((a, b) => b.engagementScore - a.engagementScore);

  const semanticRanks = new Map(semanticRanked.map((r, i) => [r.id, i + 1]));
  const engagementRanks = new Map(engagementRanked.map((r, i) => [r.id, i + 1]));

  const fused = engagementResults.map(result => {
    const sRank = semanticRanks.get(result.id) || topK * 3;
    const eRank = engagementRanks.get(result.id) || topK * 3;
    const rrfScore = 1 / (k + sRank) + 1 / (k + eRank);

    return {
      ...result,
      fusedScore: rrfScore,
    };
  });

  // Sort by fused score and return top K
  return fused
    .sort((a, b) => b.fusedScore - a.fusedScore)
    .slice(0, topK);
}

/**
 * Delete vectors from VectorAI DB by IDs.
 */
export async function deleteThreads(ids: string[]): Promise<void> {
  if (ids.length === 0) return;

  const response = await fetch(
    `${VECTORAI_CONFIG.baseUrl}/api/collections/${VECTORAI_CONFIG.collectionName}/delete`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    }
  );

  if (!response.ok) {
    throw new Error(`Delete failed: ${response.status} ${await response.text()}`);
  }
}

/**
 * Get collection stats from VectorAI DB.
 */
export async function getCollectionStats(): Promise<{ count: number; collection: string }> {
  const response = await fetch(
    `${VECTORAI_CONFIG.baseUrl}/api/collections/${VECTORAI_CONFIG.collectionName}/stats`
  );

  if (!response.ok) {
    throw new Error(`Stats failed: ${response.status}`);
  }

  const data = await response.json();
  return {
    count: data.count || 0,
    collection: VECTORAI_CONFIG.collectionName,
  };
}
