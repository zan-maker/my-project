/**
 * vector-scorer.ts — Semantic thread scorer powered by Actian VectorAI DB
 *
 * Replaces the keyword/regex-based scorer.ts with semantic similarity search.
 * Uses hybrid fusion (RRF) to combine vector similarity with engagement metrics.
 *
 * Key improvements over keyword scorer:
 *   - Catches threads with no keyword overlap but matching intent
 *   - Metadata filters for subreddit, recency, and engagement thresholds
 *   - Hybrid fusion ranking (semantic + engagement via RRF)
 *   - Configurable engagement vs semantic weight
 */

import type { RedditPost } from './reddit';
import {
  embedText,
  hybridSearch,
  type SemanticSearchResult,
  type ThreadVectorRecord,
} from './vector-store';

// ─── Types ───────────────────────────────────────────────────────────

export interface VectorScoringConfig {
  subreddits: string[];
  engagementWeight: number;   // 0–1, how much to weight engagement vs semantic (default 0.3)
  topK: number;               // Max threads to return (default 20)
  maxAgeHours: number;        // Only threads newer than this (default 168 = 7 days)
  minComments: number;        // Minimum comment count (default 0)
  minScore: number;           // Minimum upvote score (default 0)
}

export interface VectorScoredThread {
  post: RedditPost;
  semanticScore: number;       // 0–1 similarity from VectorAI DB
  engagementScore: number;     // 0–100 engagement composite
  fusedScore: number;          // RRF fused score
  subreddit: string;
  similarity: number;
}

// ─── Scoring ────────────────────────────────────────────────────────

/**
 * Build a semantic query from a business profile.
 * This replaces the keyword list — one embedding captures the full domain.
 */
export function buildSemanticQuery(profile: {
  description: string;
  valueProposition: string;
  painPoints: string;
  buyerPersona: string;
}): string {
  return [
    profile.valueProposition,
    profile.description,
    profile.painPoints,
    profile.buyerPersona,
  ]
    .filter(Boolean)
    .join('. ');
}

/**
 * Calculate engagement score for a Reddit post (same logic as original scorer).
 */
function calculateEngagementScore(post: RedditPost): number {
  const commentScore = Math.min(Math.log(post.num_comments + 1) * 10, 100);
  const upvoteScore = Math.min(Math.log(post.score + 1) * 8, 100);
  const ageInHours = (Date.now() / 1000 - post.created_utc) / 3600;
  const recencyScore = Math.max(0, 100 - (ageInHours / 168) * 100);
  return Math.round((commentScore + upvoteScore + recencyScore) / 3);
}

/**
 * Score threads using Actian VectorAI DB hybrid fusion search.
 *
 * This is the main entry point — call this instead of the old scoreThread().
 * It:
 *   1. Embeds the business profile as a semantic query
 *   2. Searches VectorAI DB with metadata filters (subreddit, recency, engagement)
 *   3. Applies Hybrid Fusion (RRF) to merge semantic + engagement signals
 *   4. Returns ranked results
 */
export async function scoreThreadsWithVector(
  profile: {
    description: string;
    valueProposition: string;
    painPoints: string;
    buyerPersona: string;
  },
  config: VectorScoringConfig
): Promise<VectorScoredThread[]> {
  const semanticQuery = buildSemanticQuery(profile);

  // Run hybrid search against VectorAI DB
  const results = await hybridSearch({
    queryText: semanticQuery,
    topK: config.topK,
    subredditFilter: config.subreddits,
    maxAgeHours: config.maxAgeHours,
    minComments: config.minComments,
    minScore: config.minScore,
    engagementWeight: config.engagementWeight,
  });

  // Map results to scored threads
  return results.map(result => ({
    post: {
      id: result.id,
      title: result.title,
      selftext: result.selftext,
      subreddit: result.subreddit,
      author: result.author,
      score: result.score,
      num_comments: result.numComments,
      created_utc: result.createdAtReddit,
      url: result.url,
      permalink: '',
      is_self: true,
    } as RedditPost,
    semanticScore: result.similarity,
    engagementScore: calculateEngagementScore({
      score: result.score,
      num_comments: result.numComments,
      created_utc: result.createdAtReddit,
    } as RedditPost),
    fusedScore: result.fusedScore,
    subreddit: result.subreddit,
    similarity: result.similarity,
  }));
}

/**
 * Convert RedditPost to ThreadVectorRecord for ingestion into VectorAI DB.
 */
export function postToVectorRecord(
  post: RedditPost,
  embedding: number[]
): ThreadVectorRecord {
  return {
    id: post.id,
    vector: embedding,
    subreddit: post.subreddit,
    title: post.title,
    selftext: (post.selftext || '').substring(0, 5000),
    author: post.author,
    score: post.score,
    numComments: post.num_comments,
    createdAtReddit: post.created_utc,
    url: post.url,
  };
}

/**
 * Ingest Reddit posts into VectorAI DB.
 * Embeds each post's title + body and upserts into the collection.
 */
export async function ingestPosts(
  posts: RedditPost[]
): Promise<{ ingested: number; failed: number }> {
  const texts = posts.map(p => `${p.title} ${p.selftext || ''}`.substring(0, 5000));

  let embeddings: number[][];
  try {
    // Batch embed for efficiency
    embeddings = await embedTexts(texts);
  } catch {
    // Fallback to individual embeds if batch fails
    embeddings = [];
    for (const text of texts) {
      try {
        const emb = await embedText(text);
        embeddings.push(emb);
      } catch {
        embeddings.push([]); // placeholder for failed embeds
      }
    }
  }

  const records: ThreadVectorRecord[] = [];
  let failed = 0;

  for (let i = 0; i < posts.length; i++) {
    if (embeddings[i] && embeddings[i].length > 0) {
      records.push(postToVectorRecord(posts[i], embeddings[i]));
    } else {
      failed++;
    }
  }

  if (records.length > 0) {
    const { upsertThreads } = await import('./vector-store');
    await upsertThreads(records);
  }

  return { ingested: records.length, failed };
}
