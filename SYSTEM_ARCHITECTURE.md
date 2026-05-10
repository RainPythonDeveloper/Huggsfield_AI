# System Architecture

```
┌──────────────────── memory-service — architecture & algorithm ──────────────────────────────┐
│                                                                                              │
│                      ┌────────────────────────────────┐                                     │
│                      │       AI Agent  (client)        │                                     │
│                      └──────────┬─────────────────────┘                                     │
│                                 │                                                            │
│           write                 │                              read                          │
│    POST /turns                  │    POST /recall   POST /search                             │
│    DELETE /sessions/{id}        │    GET  /users/{id}/memories  (active + history flat list)│
│    DELETE /users/{id}           │    GET  /health   (db ping + degraded[])                  │
│                                 ▼                                                            │
│          ┌────────────────────────────────────────────────────────────┐                     │
│          │              FastAPI  (uvicorn, port 8080)                  │                     │
│          │  middleware:  _BodySizeLimit  →  413 if Content-Length > 1MB │                     │
│          │  exception handlers: 422 (validation) · 500 (unhandled)     │                     │
│          │  lifespan: init_pool → apply_migrations (idempotent) → run  │                     │
│          └────────────────────────────────────────────────────────────┘                     │
│                                 │                                                            │
│  ── WRITE PATH (POST /turns)  →  ingest_turn() ────────────────────────────────────────  │
│                                                                                              │
│  [1] repository.insert_turn(turn)                                                            │
│       asyncpg — single atomic tx: INSERT turns + bulk INSERT messages                        │
│       (oversized payload already rejected by middleware)                                     │
│               │                                                                              │
│               ▼  (turn_id persisted unconditionally — extraction is best-effort below)       │
│                                                                                              │
│  [2] extraction.extract_and_store(turn_id, user_id, session_id, messages)                    │
│       guard: if user_id is None  →  return 0  (no place to attach durable memories)         │
│               │                                                                              │
│               ▼                                                                              │
│  [2a] _llm_extract(messages)   Alem LLM /chat/completions, temperature=0                    │
│        prompt yields → { memories: [{ type, key, value, confidence, raw_quote }] }          │
│        types whitelist:  fact │ preference │ opinion │ event │ relation                     │
│        atomicity rule:   "I work at Notion as PM"  →  TWO memories                          │
│        parser:           parse_json_lenient (tolerates ```json fences, stray prose)          │
│        on failure:       returns []  →  turn stays persisted, 0 memories inserted            │
│        clean:            drop bad type, lowercase key, clamp confidence ∈ [0,1],             │
│                          truncate value≤500, key≤80, raw_quote≤1000                         │
│               │                                                                              │
│               ▼                                                                              │
│  [2b] embeddings.embed_many(canonicals)   ◄── BATCH (asyncio.gather, 1 req/text)            │
│        canonical_text = f"User's {key.replace('_',' ')}: {value}"                            │
│        Alem /embeddings  text-1024  dim=1024                                                │
│               │                                                                              │
│               ▼  per-memory loop (zip(memories, vectors)):                                   │
│                                                                                              │
│  [2c] supersession.resolve(user_id, key, candidate)                                          │
│        ┌── find_active_memories_by_key(user_id, key)  ──────────────────────────┐           │
│        │   no existing active        →  verdict=coexist   ─►  INSERT active=true │           │
│        │   exact value match (ci)    →  verdict=noop      ─►  SKIP               │           │
│        │   different value           →  LLM judge call:                          │           │
│        │     llm.chat(temperature=0, max_tokens=200)                              │           │
│        │       parse → { verdict, reason }                                       │           │
│        │       valid: supersede│coexist│keep_old│noop                            │           │
│        │     on LLM fail / unparseable / bad verdict:                             │           │
│        │       _heuristic_verdict(key)                                            │           │
│        │         singular keys (employer, city, role, …)  →  supersede           │           │
│        │         multi-value keys (pet_*, hobby, language_spoken,                 │           │
│        │           favorite_*, currently_reading, …)       →  coexist             │           │
│        └─────────────────────────────────────────────────────────────────────────┘           │
│               │                                                                              │
│               ▼                                                                              │
│  [2d] apply verdict + repository.insert_memory(...)                                          │
│        supersede  →  insert new active=true; mark_superseded(old_ids, by_id=new)            │
│        coexist    →  insert new active=true   (both rows live)                              │
│        keep_old   →  insert new active=false  (historical mention only)                     │
│        noop       →  skip insert                                                             │
│               │                                                                              │
│               └──────────────────────────────────────────────────────────► Postgres        │
│                                                                                              │
│  ── READ PATH (POST /recall)  →  recall() ─────────────────────────────────────────────  │
│                                                                                              │
│  [R1] query_rewrite.analyze(query)   Alem LLM                                               │
│        single-hop  →  proceed with original query into [R2]                                 │
│        multi-hop   →  decomp.sub_queries  →  asyncio.gather([R2 per sub])                   │
│                       reciprocal_rank_fusion(channels="sub_i", k=60, limit=FUSED_K=20)      │
│                       on empty fused → fallback to single-hop [R2] with original query      │
│               │                                                                              │
│               ▼                                                                              │
│  [R2] _hybrid_memories(only_active=False)  ◄── active + historical BOTH in candidate pool   │
│                                              (lets historical surface for "previously at X" │
│                                               queries; current facts re-bucketed in [R5])   │
│                                                                                              │
│       BM25 channel    repository.search_memories_by_bm25   limit=RETRIEVAL_K=30  ─┐         │
│                       ts_rank_cd over memories.value_tsv (GIN)                     │         │
│                                                                                    ├─► RRF  │
│       vector channel  embeddings.embed(query) → embed.to_pgvector                  │  k=60  │
│                       repository.search_memories_by_embedding   limit=30          │  →     │
│                       HNSW ANN over memories.embedding (vector_cosine_ops)        │  top   │
│                       on Alem 5xx / timeout: log "vector_channel_failed_bm25_only"│  20    │
│                       and degrade to BM25-only (channel dict drops "vector") ────┘         │
│                                                                                              │
│       on both channels empty → return []  (triggers cold-extraction fallback below)         │
│               │                                                                              │
│               ▼                                                                              │
│  [R3] _rerank_and_filter(query, candidates, top_n=RECALL_TOP_N=8, floor=RERANK_FLOOR=0.05) │
│        if not settings.rerank_enabled  →  return candidates[:8]  (skip floor, keep RRF)     │
│        else:                                                                                  │
│          docs = [_rerank_doc_for_memory(r) for r in candidates]                              │
│          doc format = "The user's {key} is {value}. Originally said: {raw_quote}"           │
│            (3rd-person framing; calibrated 0.97 vs 0.0025 for 1st-person — Step 5 finding)  │
│          ranked = reranker.rerank(query, docs, top_n=8)   Alem cross-encoder                │
│          on rerank 5xx / timeout: log "rerank_failed_fallback_rrf"                           │
│                                  return candidates[:8]   (no floor applied)                  │
│          else: keep items with score ≥ 0.05; attach _rerank_score                            │
│               │                                                                              │
│               ▼                                                                              │
│  [R4] fetch recent_msgs (always when session_id given; rendering is gated in [R5])          │
│        repository.fetch_recent_messages_for_session(session_id, limit=4)                     │
│        on exception: log "recent_fetch_failed", recent_msgs = []                             │
│               │                                                                              │
│               ▼                                                                              │
│  [R-cold]  cold-extraction fallback   triggers ONLY when                                     │
│            (final_rows EMPTY  AND  recent_msgs EMPTY):                                       │
│              repository.search_messages_by_bm25 over messages.content_tsv (limit=8)          │
│              → _rerank_messages_filter (top_n=5, floor=0.05)                                 │
│              → _format_message_fallback  (single bucket, no priority logic)                  │
│              if still empty: return RecallOut(context="", citations=[])                      │
│               │                                                                              │
│               ▼                                                                              │
│  [R5] _format_recall_budgeted(rows, recent, budget)                                          │
│        tokens.count via tiktoken cl100k     soft_cap = max(1, ⌊0.95 × max_tokens⌋)          │
│        bullets DROPPED (not truncated) once try_add() would exceed soft_cap                  │
│                                                                                              │
│        Bucket 1  active rows where type ∈ {fact, preference, relation}  ◄── stable identity│
│                  header: "## Known facts about this user"                                    │
│                                                                                              │
│        Bucket 2  active rows where type ∈ {event, opinion, …}    THEN     historical rows   │
│                  (active=False) appended; humanize() tags them "(historical)"                │
│                  header: "## Relevant memories"                                              │
│                                                                                              │
│        Bucket 3  recent session messages   GATED:                                            │
│                  used < soft_cap × 0.8   AND   len(citations) < 6                            │
│                  header: "## Recent conversation"                                             │
│                                                                                              │
│        citations: _cite() builds Citation(turn_id=source_turn, score=_rerank_score, snippet) │
│        _dedup_citations: key=(turn_id, snippet[:120]),  cap=6                                │
│               │                                                                              │
│               └──────────────────────────────────────────────────────────► RecallOut       │
│                                                                                              │
│  ── READ PATH (POST /search)  →  search()  (NOT same as /recall) ──────────────────────  │
│                                                                                              │
│       NO query_rewrite.analyze         (single-hop only)                                     │
│       NO budget assembly / buckets     (returns structured SearchHit[])                      │
│                                                                                              │
│       _hybrid_memories(per_channel=max(limit·2, 30), fused=max(limit·2, 20))                │
│           │                                                                                  │
│           ▼                                                                                  │
│       _rerank_and_filter(top_n=req.limit, floor=0.05)                                        │
│           │                                                                                  │
│           ▼                                                                                  │
│       SearchOut[ SearchHit { content="key: value", score, session_id, timestamp,             │
│                              metadata={type, confidence, raw_quote, active, channels} } ]    │
│                                                                                              │
│  ── STORAGE ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Postgres 16 + pgvector                        named volume: pgdata  (survives ↺)    │  │
│  │  pgcrypto extension → gen_random_uuid()                                                │  │
│  │                                                                                        │  │
│  │  turns                messages                       memories                          │  │
│  │  ─────                ────────                       ────────                          │  │
│  │  id            uuid PK   id          uuid PK         id              uuid PK          │  │
│  │  session_id    text      turn_id → turns.id ON DEL   user_id         text             │  │
│  │  user_id       text      role        text   CASCADE  session_id      text             │  │
│  │  timestamp     tstz      name        text            type             text            │  │
│  │  metadata      jsonb     content     text            key              text             │  │
│  │  raw           jsonb     position    int             value            text             │  │
│  │  created_at    tstz      content_tsv tsvector GEN    raw_quote        text             │  │
│  │                          (to_tsvector('english',     confidence       real DEF 0.8    │  │
│  │  IDX:                     content)) STORED            embedding        vector(1024)    │  │
│  │   turns_session_idx                                  value_tsv        tsvector GEN    │  │
│  │   turns_user_idx         IDX:                         (key || ' ' || value)           │  │
│  │                           messages_turn_idx          source_turn      uuid → turns    │  │
│  │                           messages_tsv_idx (GIN)        ON DELETE SET NULL            │  │
│  │                           ▲                          source_session   text             │  │
│  │                           cold-extraction fallback   supersedes       uuid → memories │  │
│  │                                                         self-FK ON DELETE SET NULL    │  │
│  │                                                      active           bool DEF TRUE   │  │
│  │                                                      created_at       tstz            │  │
│  │                                                      updated_at       tstz            │  │
│  │                                                                                        │  │
│  │                                                      IDX:                              │  │
│  │                                                       memories_user_active_idx        │  │
│  │                                                          (user_id) WHERE active       │  │
│  │                                                       memories_session_idx (session_id)│  │
│  │                                                       memories_key_idx                │  │
│  │                                                          (user_id, key) WHERE active  │  │
│  │                                                       memories_tsv_idx                │  │
│  │                                                          GIN(value_tsv)               │  │
│  │                                                       memories_embedding_idx          │  │
│  │                                                          HNSW(embedding               │  │
│  │                                                               vector_cosine_ops)      │  │
│  │                                                                                        │  │
│  │  supersession chain example:                                                           │  │
│  │   id=A  employer: Stripe   active=false  supersedes=NULL   ◄── displaced               │  │
│  │   id=B  employer: Notion   active=true   supersedes=A      ◄── current                │  │
│  │                                                                                        │  │
│  └──────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                              │
│  ── EXTERNAL DEPENDENCIES (all OpenAI/Cohere-compatible HTTP) ──────────────────────────  │
│                                                                                              │
│   Alem LLM     /chat/completions    extraction · supersession judge · query rewrite        │
│                  retry: tenacity 3× exp(0.5–4s) on httpx.HTTPError/Timeout                  │
│                  if disabled (no ALEM_API_KEY)  →  /health adds "llm" to degraded[]         │
│                                                                                              │
│   Alem Embed   /embeddings          text-1024,  dim=1024                                    │
│                  retry: 5× exp(0.6–8s)  · embed_many is asyncio.gather over /embed          │
│                  if disabled  →  ingest INSERT fails (vector NOT NULL)  →  500              │
│                                  recall degrades to BM25-only path                           │
│                                                                                              │
│   Alem Rerank  /rerank              cross-encoder (Cohere-compatible)                       │
│                  retry: 3× exp(0.5–4s)                                                       │
│                  if disabled  →  RRF order kept; floor=0.05 NOT applied                      │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────── memory-service — architecture & algorithm ──────────────────────────────┐
│                                                                                              │
│                      ┌────────────────────────────────┐                                     │
│                      │       AI Agent  (client)        │                                     │
│                      └──────────┬─────────────────────┘                                     │
│                                 │                                                            │
│           write                 │                              read                          │
│    POST /turns                  │    POST /recall   POST /search                             │
│    DELETE /sessions/{id}        │    GET  /users/{id}/memories  (active + history flat list)│
│    DELETE /users/{id}           │    GET  /health   (db ping + degraded[])                  │
│                                 ▼                                                            │
│          ┌────────────────────────────────────────────────────────────┐                     │
│          │              FastAPI  (uvicorn, port 8080)                  │                     │
│          │  middleware:  _BodySizeLimit  →  413 if Content-Length > 1MB │                     │
│          │  exception handlers: 422 (validation) · 500 (unhandled)     │                     │
│          │  lifespan: init_pool → apply_migrations (idempotent) → run  │                     │
│          └────────────────────────────────────────────────────────────┘                     │
│                                 │                                                            │
│  ── WRITE PATH (POST /turns)  →  ingest_turn() ────────────────────────────────────────  │
│                                                                                              │
│  [1] repository.insert_turn(turn)                                                            │
│       asyncpg — single atomic tx: INSERT turns + bulk INSERT messages                        │
│       (oversized payload already rejected by middleware)                                     │
│               │                                                                              │
│               ▼  (turn_id persisted unconditionally — extraction is best-effort below)       │
│                                                                                              │
│  [2] extraction.extract_and_store(turn_id, user_id, session_id, messages)                    │
│       guard: if user_id is None  →  return 0  (no place to attach durable memories)         │
│               │                                                                              │
│               ▼                                                                              │
│  [2a] _llm_extract(messages)   Alem LLM /chat/completions, temperature=0                    │
│        prompt yields → { memories: [{ type, key, value, confidence, raw_quote }] }          │
│        types whitelist:  fact │ preference │ opinion │ event │ relation                     │
│        atomicity rule:   "I work at Notion as PM"  →  TWO memories                          │
│        parser:           parse_json_lenient (tolerates ```json fences, stray prose)          │
│        on failure:       returns []  →  turn stays persisted, 0 memories inserted            │
│        clean:            drop bad type, lowercase key, clamp confidence ∈ [0,1],             │
│                          truncate value≤500, key≤80, raw_quote≤1000                         │
│               │                                                                              │
│               ▼                                                                              │
│  [2b] embeddings.embed_many(canonicals)   ◄── BATCH (asyncio.gather, 1 req/text)            │
│        canonical_text = f"User's {key.replace('_',' ')}: {value}"                            │
│        Alem /embeddings  text-1024  dim=1024                                                │
│               │                                                                              │
│               ▼  per-memory loop (zip(memories, vectors)):                                   │
│                                                                                              │
│  [2c] supersession.resolve(user_id, key, candidate)                                          │
│        ┌── find_active_memories_by_key(user_id, key)  ──────────────────────────┐           │
│        │   no existing active        →  verdict=coexist   ─►  INSERT active=true │           │
│        │   exact value match (ci)    →  verdict=noop      ─►  SKIP               │           │
│        │   different value           →  LLM judge call:                          │           │
│        │     llm.chat(temperature=0, max_tokens=200)                              │           │
│        │       parse → { verdict, reason }                                       │           │
│        │       valid: supersede│coexist│keep_old│noop                            │           │
│        │     on LLM fail / unparseable / bad verdict:                             │           │
│        │       _heuristic_verdict(key)                                            │           │
│        │         singular keys (employer, city, role, …)  →  supersede           │           │
│        │         multi-value keys (pet_*, hobby, language_spoken,                 │           │
│        │           favorite_*, currently_reading, …)       →  coexist             │           │
│        └─────────────────────────────────────────────────────────────────────────┘           │
│               │                                                                              │
│               ▼                                                                              │
│  [2d] apply verdict + repository.insert_memory(...)                                          │
│        supersede  →  insert new active=true; mark_superseded(old_ids, by_id=new)            │
│        coexist    →  insert new active=true   (both rows live)                              │
│        keep_old   →  insert new active=false  (historical mention only)                     │
│        noop       →  skip insert                                                             │
│               │                                                                              │
│               └──────────────────────────────────────────────────────────► Postgres        │
│                                                                                              │
│  ── READ PATH (POST /recall)  →  recall() ─────────────────────────────────────────────  │
│                                                                                              │
│  [R1] query_rewrite.analyze(query)   Alem LLM                                               │
│        single-hop  →  proceed with original query into [R2]                                 │
│        multi-hop   →  decomp.sub_queries  →  asyncio.gather([R2 per sub])                   │
│                       reciprocal_rank_fusion(channels="sub_i", k=60, limit=FUSED_K=20)      │
│                       on empty fused → fallback to single-hop [R2] with original query      │
│               │                                                                              │
│               ▼                                                                              │
│  [R2] _hybrid_memories(only_active=False)  ◄── active + historical BOTH in candidate pool   │
│                                              (lets historical surface for "previously at X" │
│                                               queries; current facts re-bucketed in [R5])   │
│                                                                                              │
│       BM25 channel    repository.search_memories_by_bm25   limit=RETRIEVAL_K=30  ─┐         │
│                       ts_rank_cd over memories.value_tsv (GIN)                     │         │
│                                                                                    ├─► RRF  │
│       vector channel  embeddings.embed(query) → embed.to_pgvector                  │  k=60  │
│                       repository.search_memories_by_embedding   limit=30          │  →     │
│                       HNSW ANN over memories.embedding (vector_cosine_ops)        │  top   │
│                       on Alem 5xx / timeout: log "vector_channel_failed_bm25_only"│  20    │
│                       and degrade to BM25-only (channel dict drops "vector") ────┘         │
│                                                                                              │
│       on both channels empty → return []  (triggers cold-extraction fallback below)         │
│               │                                                                              │
│               ▼                                                                              │
│  [R3] _rerank_and_filter(query, candidates, top_n=RECALL_TOP_N=8, floor=RERANK_FLOOR=0.05) │
│        if not settings.rerank_enabled  →  return candidates[:8]  (skip floor, keep RRF)     │
│        else:                                                                                  │
│          docs = [_rerank_doc_for_memory(r) for r in candidates]                              │
│          doc format = "The user's {key} is {value}. Originally said: {raw_quote}"           │
│            (3rd-person framing; calibrated 0.97 vs 0.0025 for 1st-person — Step 5 finding)  │
│          ranked = reranker.rerank(query, docs, top_n=8)   Alem cross-encoder                │
│          on rerank 5xx / timeout: log "rerank_failed_fallback_rrf"                           │
│                                  return candidates[:8]   (no floor applied)                  │
│          else: keep items with score ≥ 0.05; attach _rerank_score                            │
│               │                                                                              │
│               ▼                                                                              │
│  [R4] fetch recent_msgs (always when session_id given; rendering is gated in [R5])          │
│        repository.fetch_recent_messages_for_session(session_id, limit=4)                     │
│        on exception: log "recent_fetch_failed", recent_msgs = []                             │
│               │                                                                              │
│               ▼                                                                              │
│  [R-cold]  cold-extraction fallback   triggers ONLY when                                     │
│            (final_rows EMPTY  AND  recent_msgs EMPTY):                                       │
│              repository.search_messages_by_bm25 over messages.content_tsv (limit=8)          │
│              → _rerank_messages_filter (top_n=5, floor=0.05)                                 │
│              → _format_message_fallback  (single bucket, no priority logic)                  │
│              if still empty: return RecallOut(context="", citations=[])                      │
│               │                                                                              │
│               ▼                                                                              │
│  [R5] _format_recall_budgeted(rows, recent, budget)                                          │
│        tokens.count via tiktoken cl100k     soft_cap = max(1, ⌊0.95 × max_tokens⌋)          │
│        bullets DROPPED (not truncated) once try_add() would exceed soft_cap                  │
│                                                                                              │
│        Bucket 1  active rows where type ∈ {fact, preference, relation}  ◄── stable identity│
│                  header: "## Known facts about this user"                                    │
│                                                                                              │
│        Bucket 2  active rows where type ∈ {event, opinion, …}    THEN     historical rows   │
│                  (active=False) appended; humanize() tags them "(historical)"                │
│                  header: "## Relevant memories"                                              │
│                                                                                              │
│        Bucket 3  recent session messages   GATED:                                            │
│                  used < soft_cap × 0.8   AND   len(citations) < 6                            │
│                  header: "## Recent conversation"                                             │
│                                                                                              │
│        citations: _cite() builds Citation(turn_id=source_turn, score=_rerank_score, snippet) │
│        _dedup_citations: key=(turn_id, snippet[:120]),  cap=6                                │
│               │                                                                              │
│               └──────────────────────────────────────────────────────────► RecallOut       │
│                                                                                              │
│  ── READ PATH (POST /search)  →  search()  (NOT same as /recall) ──────────────────────  │
│                                                                                              │
│       NO query_rewrite.analyze         (single-hop only)                                     │
│       NO budget assembly / buckets     (returns structured SearchHit[])                      │
│                                                                                              │
│       _hybrid_memories(per_channel=max(limit·2, 30), fused=max(limit·2, 20))                │
│           │                                                                                  │
│           ▼                                                                                  │
│       _rerank_and_filter(top_n=req.limit, floor=0.05)                                        │
│           │                                                                                  │
│           ▼                                                                                  │
│       SearchOut[ SearchHit { content="key: value", score, session_id, timestamp,             │
│                              metadata={type, confidence, raw_quote, active, channels} } ]    │
│                                                                                              │
│  ── STORAGE ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Postgres 16 + pgvector                        named volume: pgdata  (survives ↺)    │  │
│  │  pgcrypto extension → gen_random_uuid()                                                │  │
│  │                                                                                        │  │
│  │  turns                messages                       memories                          │  │
│  │  ─────                ────────                       ────────                          │  │
│  │  id            uuid PK   id          uuid PK         id              uuid PK          │  │
│  │  session_id    text      turn_id → turns.id ON DEL   user_id         text             │  │
│  │  user_id       text      role        text   CASCADE  session_id      text             │  │
│  │  timestamp     tstz      name        text            type             text            │  │
│  │  metadata      jsonb     content     text            key              text             │  │
│  │  raw           jsonb     position    int             value            text             │  │
│  │  created_at    tstz      content_tsv tsvector GEN    raw_quote        text             │  │
│  │                          (to_tsvector('english',     confidence       real DEF 0.8    │  │
│  │  IDX:                     content)) STORED            embedding        vector(1024)    │  │
│  │   turns_session_idx                                  value_tsv        tsvector GEN    │  │
│  │   turns_user_idx         IDX:                         (key || ' ' || value)           │  │
│  │                           messages_turn_idx          source_turn      uuid → turns    │  │
│  │                           messages_tsv_idx (GIN)        ON DELETE SET NULL            │  │
│  │                           ▲                          source_session   text             │  │
│  │                           cold-extraction fallback   supersedes       uuid → memories │  │
│  │                                                         self-FK ON DELETE SET NULL    │  │
│  │                                                      active           bool DEF TRUE   │  │
│  │                                                      created_at       tstz            │  │
│  │                                                      updated_at       tstz            │  │
│  │                                                                                        │  │
│  │                                                      IDX:                              │  │
│  │                                                       memories_user_active_idx        │  │
│  │                                                          (user_id) WHERE active       │  │
│  │                                                       memories_session_idx (session_id)│  │
│  │                                                       memories_key_idx                │  │
│  │                                                          (user_id, key) WHERE active  │  │
│  │                                                       memories_tsv_idx                │  │
│  │                                                          GIN(value_tsv)               │  │
│  │                                                       memories_embedding_idx          │  │
│  │                                                          HNSW(embedding               │  │
│  │                                                               vector_cosine_ops)      │  │
│  │                                                                                        │  │
│  │  supersession chain example:                                                           │  │
│  │   id=A  employer: Stripe   active=false  supersedes=NULL   ◄── displaced               │  │
│  │   id=B  employer: Notion   active=true   supersedes=A      ◄── current                │  │
│  │                                                                                        │  │
│  └──────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                              │
│  ── EXTERNAL DEPENDENCIES (all OpenAI/Cohere-compatible HTTP) ──────────────────────────  │
│                                                                                              │
│   Alem LLM     /chat/completions    extraction · supersession judge · query rewrite        │
│                  retry: tenacity 3× exp(0.5–4s) on httpx.HTTPError/Timeout                  │
│                  if disabled (no ALEM_API_KEY)  →  /health adds "llm" to degraded[]         │
│                                                                                              │
│   Alem Embed   /embeddings          text-1024,  dim=1024                                    │
│                  retry: 5× exp(0.6–8s)  · embed_many is asyncio.gather over /embed          │
│                  if disabled  →  ingest INSERT fails (vector NOT NULL)  →  500              │
│                                  recall degrades to BM25-only path                           │
│                                                                                              │
│   Alem Rerank  /rerank              cross-encoder (Cohere-compatible)                       │
│                  retry: 3× exp(0.5–4s)                                                       │
│                  if disabled  →  RRF order kept; floor=0.05 NOT applied                      │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---
