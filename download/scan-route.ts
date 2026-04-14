/**
 * scan/route.ts — Updated scan route with VectorAI DB integration
 *
 * This replaces the original keyword-based scan with a hybrid approach:
 *   1. Fetch threads from Reddit (same as before)
 *   2. Embed and ingest into VectorAI DB
 *   3. Run hybrid fusion search (semantic + engagement)
 *   4. Store scored results in SQLite via Prisma
 */

import { NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { createRedditClient, searchReddit } from '@/lib/reddit';
import { scoreThreadsWithVector, buildSemanticQuery, ingestPosts } from '@/lib/vector-scorer';
import { initCollection, getCollectionStats } from '@/lib/vector-store';

export async function POST(request: Request) {
  try {
    const { businessId } = await request.json();
    const business = await db.businessProfile.findUnique({ where: { id: businessId } });
    if (!business) return NextResponse.json({ error: 'Business not found' }, { status: 404 });

    const scanRun = await db.scanRun.create({
      data: { businessId, status: 'running' },
    });

    try {
      const client = await createRedditClient();
      const subreddits: string[] = JSON.parse(business.subreddits || '[]');

      if (subreddits.length === 0) {
        await db.scanRun.update({
          where: { id: scanRun.id },
          data: { status: 'failed', errorMessage: 'No subreddits configured', completedAt: new Date() },
        });
        return NextResponse.json({ error: 'No subreddits configured' }, { status: 400 });
      }

      // Initialize VectorAI DB collection
      await initCollection();
      const stats = await getCollectionStats();
      console.log(`VectorAI DB collection: ${stats.count} existing threads`);

      // Step 1: Fetch threads from Reddit
      const posts = await searchReddit(client, {
        query: '',  // No keyword filter — we rely on semantic matching
        subreddits,
        sort: 'new',
        timeRange: 'week',
        limit: 50,
      });

      // Step 2: Ingest into VectorAI DB (embed + upsert)
      try {
        const ingestResult = await ingestPosts(posts);
        console.log(`Ingested ${ingestResult.ingested} threads into VectorAI DB`);
      } catch (ingestError: any) {
        console.warn(`Ingestion warning: ${ingestError.message}`);
        // Continue with search even if ingestion fails
      }

      // Step 3: Hybrid fusion search via VectorAI DB
      const scoredThreads = await scoreThreadsWithVector(
        {
          description: business.description,
          valueProposition: business.valueProposition,
          painPoints: business.painPoints,
          buyerPersona: business.buyerPersona,
        },
        {
          subreddits,
          engagementWeight: 0.3,
          topK: 20,
          maxAgeHours: 168,  // 7 days
          minComments: 0,
          minScore: 0,
        }
      );

      // Step 4: Store results in Prisma/SQLite
      let storedCount = 0;
      for (const thread of scoredThreads) {
        const existing = await db.redditThread.findFirst({
          where: { businessId, redditId: thread.post.id },
        });
        if (existing) continue;

        await db.redditThread.create({
          data: {
            businessId,
            redditId: thread.post.id,
            subreddit: thread.post.subreddit,
            title: thread.post.title,
            author: thread.post.author,
            selftext: (thread.post.selftext || '').substring(0, 5000),
            url: thread.post.url,
            score: thread.post.score,
            numComments: thread.post.num_comments,
            createdAtReddit: new Date(thread.post.created_utc * 1000),
            engagementScore: Math.round(thread.engagementScore),
            buyingIntentScore: Math.round(thread.semanticScore * 100),
            totalScore: Math.round(thread.fusedScore * 1000),
            matchedKeywords: JSON.stringify(['semantic-match']),
            matchedCompetitors: JSON.stringify([]),
            intentSignals: JSON.stringify([`similarity: ${(thread.similarity * 100).toFixed(1)}%`]),
            isRelevant: true,
            isProcessed: false,
            scanRunId: scanRun.id,
          },
        });
        storedCount++;
      }

      await db.scanRun.update({
        where: { id: scanRun.id },
        data: {
          status: 'completed',
          threadsFound: posts.length,
          threadsScored: scoredThreads.length,
          completedAt: new Date(),
        },
      });

      return NextResponse.json({
        success: true,
        scanRunId: scanRun.id,
        threadsFound: posts.length,
        threadsScored: scoredThreads.length,
        threadsStored: storedCount,
        vectoraiCollection: stats.count + storedCount,
      });
    } catch (error: any) {
      await db.scanRun.update({
        where: { id: scanRun.id },
        data: { status: 'failed', errorMessage: error.message, completedAt: new Date() },
      });
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
