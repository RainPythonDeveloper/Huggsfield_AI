"""LLM judge prompt for fact-evolution / contradiction resolution.

Given an existing ACTIVE memory and a freshly-extracted candidate that share
the same `(user_id, key)`, decide what to do:

  - supersede:   the new fact replaces the old. Mark old `active=false`,
                 insert new with `supersedes=old.id`.
                 Example: old "employer: Stripe", new "employer: Notion" with
                 raw_quote "I just started at Notion, switched from Stripe".

  - coexist:     both are concurrently true. Insert new alongside; both active.
                 Example: old "pet_dog_name: Biscuit", new "pet_dog_name: Rex".
                 (User owns two dogs.)

  - keep_old:    the new candidate is a HISTORICAL mention, the old is current.
                 Insert new with `active=false`, no supersession link.
                 Example: old "employer: Notion" (current), new "employer:
                 Stripe" with raw_quote "I switched from Stripe last year".

  - noop:        the new is a duplicate or a less-precise restatement. Skip.
                 Example: old "city: New York", new "city: NYC".
"""

SYSTEM = """You are a memory contradiction resolver. You will see one or more \
EXISTING memories about a user (currently active) and one NEW candidate \
memory about the same key. Decide what to do with the new candidate.

Output strict JSON only:
{"verdict": "supersede" | "coexist" | "keep_old" | "noop", "reason": "<short>"}

Verdict guide:
- "supersede": the NEW fact replaces the existing one. Use this for mutable \
properties (employer, role, city, dietary_restriction, current_project) \
when the new fact's quote indicates a CHANGE (signals: "started", "switched", \
"now", "moved", "joined", "actually I meant", "no longer", "used to").
- "coexist": both are simultaneously true. Use for keys that legitimately \
multi-value (pets, hobbies, languages, children).
- "keep_old": the existing fact is current; the new candidate is the user \
mentioning a historical or contextual fact about themselves. The signal is \
PAST tense ("used to work at X", "switched from X", "before that I lived in X").
- "noop": the new is a less-precise restatement, near-duplicate, or a typo \
of the existing.

Decide ONLY based on the user's words (the raw quotes). When in doubt between \
supersede and coexist, prefer supersede for keys that are typically singular \
(employer, city, role, age) and coexist for keys that are typically plural \
(pet_*, hobby, language, child_name, friend_name).

Output JSON only."""


def build_user_prompt(*, key: str, existing: list[dict], candidate: dict) -> str:
    lines = [f"Memory key: {key}", "", "EXISTING (active):"]
    for i, e in enumerate(existing):
        quote = e.get("raw_quote") or "(no quote available)"
        lines.append(f"  [{i}] value={e['value']!r}  quote={quote!r}")
    lines.extend(
        [
            "",
            "NEW candidate:",
            f"  value={candidate['value']!r}  quote={candidate.get('raw_quote') or '(no quote)'!r}",
            "",
            "Decide.",
        ]
    )
    return "\n".join(lines)
