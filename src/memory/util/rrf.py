"""Reciprocal Rank Fusion. Cormack et al. 2009.

  score(d) = Σ_channel  1 / (k + rank_channel(d))

We use k=60 (the value from the original paper). Higher k flattens score
contribution from low-rank items; 60 is the standard default.
"""

from collections.abc import Iterable
from typing import Any


def reciprocal_rank_fusion(
    channels: dict[str, list[dict[str, Any]]],
    *,
    id_key: str = "id",
    k: int = 60,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists into one. Returns items in fused order.

    Each channel is a list ordered best-first. The same item (matched by
    `id_key`) can appear across channels; their reciprocal ranks sum.

    Returned items have an extra `_rrf_score` key plus per-channel rank info
    in `_channels` so downstream stages can reason about which channel
    surfaced each hit.
    """
    seen: dict[Any, dict[str, Any]] = {}
    rrf_scores: dict[Any, float] = {}
    channel_ranks: dict[Any, dict[str, int]] = {}

    for channel_name, items in channels.items():
        for rank, item in enumerate(items):
            item_id = item.get(id_key)
            if item_id is None:
                continue
            if item_id not in seen:
                # First sighting — keep this row as the canonical record (it
                # has all the columns we need; channels may differ per query
                # but the SELECT shape is the same).
                seen[item_id] = dict(item)
                channel_ranks[item_id] = {}
                rrf_scores[item_id] = 0.0
            channel_ranks[item_id][channel_name] = rank
            rrf_scores[item_id] += 1.0 / (k + rank)

    fused = []
    for item_id, score in rrf_scores.items():
        item = seen[item_id]
        item["_rrf_score"] = score
        item["_channels"] = channel_ranks[item_id]
        fused.append(item)

    fused.sort(key=lambda x: x["_rrf_score"], reverse=True)
    if limit is not None:
        fused = fused[:limit]
    return fused


def normalize_ids(items: Iterable[dict[str, Any]], *, id_key: str = "id") -> list[dict[str, Any]]:
    """Drop items without a usable id_key. Useful when fusing rows from
    heterogeneous sources where some don't carry the id field."""
    return [i for i in items if i.get(id_key) is not None]
