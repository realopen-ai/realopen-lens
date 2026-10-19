"""Known-positive query construction from frozen dataset annotations."""


def query_items(rows: list[dict]) -> list[dict]:
    queries = []
    for row in rows:
        if row["source"]["kind"] == "derived" or row["expected"]["decode"] != "success":
            continue
        text = row.get("query")
        if not text and row.get("key_phrases"):
            text = " ".join(row["key_phrases"][:2])
        if text:
            # Templates share split groups; relevance includes only the same content.
            relevant = [
                r["id"]
                for r in rows
                if (r["id"] == row["id"] or r["source"].get("parent") == row["id"])
                and r["expected"]["decode"] == "success"
                and r["subcategory"] != "crop"
            ]
            variants = [{"language": row.get("language", "en"), "text": text, "kind": "natural"}]
            if row.get("identifier_query"):
                variants.append(
                    {
                        "language": row.get("language", "en"),
                        "text": row["identifier_query"],
                        "kind": "identifier",
                    }
                )
            variants.extend(
                {**item, "kind": "translated"} for item in row.get("query_variants", [])
            )
            seen = set()
            for number, item in enumerate(variants):
                if item["text"] in seen:
                    continue
                seen.add(item["text"])
                queries.append(
                    {
                        "id": row["id"] + f"::q{number}",
                        "text": item["text"],
                        "language": item["language"],
                        "kind": item["kind"],
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                        "split": row["split"],
                        "relevant": relevant,
                        **({"corpus": row["corpus"]} if "corpus" in row else {}),
                        "judgment": (
                            "Published caption or source/visually authored known positive; "
                            "translations not independently reviewed"
                        ),
                    }
                )
    return queries
