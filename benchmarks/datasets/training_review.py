"""Apply separately frozen training-only visual reviews without changing held-out labels."""

from copy import deepcopy


def apply(bundle: dict, corpus: dict, review: dict) -> dict:
    if review.get("version") != 1 or review.get("scope") != (
        "train-only; labels shared across translations of the same family"
    ):
        raise ValueError("Unsupported training review schema or scope")
    provenance = review.get("review", {})
    if provenance.get("status") != "reviewed" or not all(
        provenance.get(key) for key in ("reviewer", "date", "method")
    ):
        raise ValueError("Training additions require explicit review provenance")
    images = {row["sha256"]: row for row in corpus["images"]}
    families: dict[str, list] = {}
    for query in bundle["queries"]:
        families.setdefault(query["family"], []).append(query)
    seen = set()
    additions: dict[str, list] = {}
    for row in review["pairs"]:
        family, key = row["family"], row["image_id"]
        pair = (family, key)
        if pair in seen:
            raise ValueError("Duplicate reviewed family/image pair")
        seen.add(pair)
        if (
            family not in families
            or key not in images
            or images[key]["split"] != "train"
            or images[key]["path"] != row["path"]
            or any(q["split"] != "train" for q in families[family])
        ):
            raise ValueError("Training review crosses a held-out boundary or image identity")
        if any(
            label["image_sha256"] == key
            for query in families[family]
            for label in query["judgments"]
        ):
            raise ValueError("Training review cannot replace an existing judgment")
        grade = row["grade"]
        if grade is not None and (type(grade) is not int or grade not in (0, 1, 2)):
            raise ValueError("Invalid reviewed grade")
        if not row.get("reason") or not row.get("observation"):
            raise ValueError("Reviewed pairs need observed evidence and a decision reason")
        if grade is not None:
            additions.setdefault(family, []).append(
                {"image_sha256": key, "grade": grade, "reason": row["reason"]}
            )
    updated = deepcopy(bundle)
    for query in updated["queries"]:
        if query["family"] in additions:
            query["judgments"].extend(additions[query["family"]])
    return updated
