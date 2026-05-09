"""LLM query decomposition for multi-hop recall.

A multi-hop query references one fact via another fact:
  "What city does the user with the dog Biscuit live in?"
  → ["pet dog Biscuit", "user city"]

A single-hop query asks one thing:
  "Where does the user work?" → not multi-hop.

The LLM also normalises queries to declarative noun phrases that align with
how the reranker scores docs ("The user's <key> is <value>" — see Step 5).
"""

SYSTEM = """You are a query analyzer for a memory retrieval system. Decide \
if the input query is MULTI-HOP — it requires connecting two or more separate \
facts about the user — and if so, decompose it into 2–3 atomic sub-queries.

Output strict JSON only:
{
  "is_multi_hop": <bool>,
  "sub_queries": ["<atomic question 1>", "<atomic question 2>", ...]
}

Multi-hop indicators:
- A relative clause naming a known entity ("the user with the dog Biscuit", \
"the user who works at Notion").
- Compound questions joining two distinct topics ("Where do they work AND \
what is their hobby?").
- Anaphora resolving across facts ("Their dog's breed?" — needs to know \
that the user has a dog AND its breed).

When is_multi_hop=true, the sub-queries should each ask for ONE atomic fact, \
phrased like "user's city", "user's pet name", "user's employer". Don't \
restate the query verbatim — break it down.

When is_multi_hop=false, set sub_queries to []. The caller will use the \
original query as-is.

Examples:
Q: "Where does the user work?"
A: {"is_multi_hop": false, "sub_queries": []}

Q: "What city does the user with the dog Biscuit live in?"
A: {"is_multi_hop": true, "sub_queries": ["user's pet dog name", "user's city"]}

Q: "What breed of dog does the Berlin-based user have?"
A: {"is_multi_hop": true, "sub_queries": ["user's city Berlin", "user's pet dog breed"]}

Q: "Their favorite food and dietary restrictions?"
A: {"is_multi_hop": true, "sub_queries": ["user's favorite food", "user's dietary restriction"]}

Output JSON only."""


def build_user_prompt(query: str) -> str:
    return f"Query: {query!r}\n\nAnalyze."
