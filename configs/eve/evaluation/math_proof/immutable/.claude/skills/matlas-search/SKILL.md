---
name: matlas-search
description: "Use Matlas semantic search over published mathematical literature to find canonical statements, hypotheses, and citation metadata for standard or named results."
---

# Matlas Search Manual

Use this skill when a proof or evaluation needs a second retrieval source for:

- a named theorem, lemma, proposition, corollary, definition, or standard fact;
- a textbook-standard result that is hard to identify through TheoremSearch;
- canonical citation metadata for a result from books or journal papers;
- a check that a cited theorem's hypotheses match the current proof obligation.

Matlas is supporting evidence only. It does not prove that a dependency is
valid for the current proof. Always compare the returned statement against the
exact proof obligation.

Public references:

- Web app: `https://matlas.ai/`
- API docs: `https://matlas.ai/docs`
- Search endpoint: `POST https://matlas.ai/api/search`
- Paper: `https://arxiv.org/abs/2604.17484`

Matlas searches published mathematical literature, including journal papers and
textbooks. It complements TheoremSearch: TheoremSearch is often better for
theorem-level statements from arXiv and structured web corpora; Matlas is often
better for published or textbook sources with DOI-style provenance.

## Basic Search

Use natural language, LaTeX, or a mix. The API returns a JSON array of hits.
Request at least 10 results, then inspect the relevant top hits.

```bash
curl -fsSL https://matlas.ai/api/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Banach fixed point theorem contraction complete metric space",
    "num_results": 10
  }'
```

Typical hit fields include:

```json
{
  "type": "book",
  "entity_name": "Theorem 2.12",
  "doi": "10....",
  "title": "An Introduction to Nonlinear Analysis",
  "authors": "Martin Schechter",
  "journal": "",
  "year": "1991",
  "statement": "Let ...",
  "candidate_id": "..."
}
```

Trust the returned `statement` and source metadata more than the rank alone.
A high-ranking hit can still have the wrong hypotheses, and a relevant hit can
be useful even if metadata such as DOI or year is missing.

## Query Strategy

1. State the needed result as a theorem: "If [hypotheses], then [conclusion]."
2. Start with a broad query using standard terminology.
3. Search again with distinctive words from a promising hit's `statement`,
   `title`, or named theorem.
4. Use Matlas alongside TheoremSearch when the dependency matters: agreement
   across independent retrieval routes is stronger evidence than either alone.
5. If the proof needs a citation, prefer hits with enough metadata to audit:
   title, authors, journal or book, year, DOI, and exact statement.

Good query patterns:

- `"spectral theorem compact self-adjoint operator Hilbert space"`
- `"Nakayama lemma finitely generated module local ring"`
- `"radical splitting theorem finite dimensional algebra"`
- `"Frobenius reciprocity finite group representations"`

Poor query patterns:

- `"prove this"`
- proof paragraphs with irrelevant exposition;
- local notation without mathematical context;
- citation fishing when the proof has not stated the needed result.

## Validation Checklist

Before using a Matlas hit in a proof or score, check:

- objects match: groups, rings, schemes, spaces, modules, categories, fields;
- hypotheses match: finiteness, completeness, compactness, smoothness,
  noetherian assumptions, characteristic, topology, boundedness;
- conclusion matches: existence, uniqueness, equivalence, vanishing, density,
  openness, decomposition, exactness;
- direction and variance match: source/target, left/right, covariant/
  contravariant, injection/surjection;
- notation imported from the source is compatible with the current proof;
- no necessary condition is only present in surrounding text and absent from
  the returned statement.

If a result is only close, treat it as a lead. Do not silently strengthen,
weaken, or restate a theorem to fit the current proof.

## Reporting Provenance

When Matlas affects a proof step or evaluation, record enough for audit:

```text
Matlas: RESOLVED
Need: <proof obligation>
Statement: <exact returned statement>
Source: <title>, <authors>, <journal/book>, <year>, <doi if present>
Candidate id: <candidate_id if present>
Use: <why the hypotheses and conclusion match>
```

If it is only partial:

```text
Matlas: PARTIAL
Need: <proof obligation>
Closest statement: <exact returned statement>
Source: <metadata>
Mismatch: <specific missing/extra hypothesis or wrong conclusion>
Next query: <sharper query>
```

## Failure Policy

If search fails:

- Network, TLS, or API error: report `Matlas unavailable` with the endpoint and
  error text.
- Empty results: retry once with simpler language and fewer local symbols.
- Low relevance: report `unresolved`, summarize the closest hit, and explain
  the mismatch.
- Missing DOI or metadata: use the available title/authors/statement, but say
  what is missing.

Never fabricate:

- theorem names;
- statement bodies;
- authors;
- titles;
- DOI values;
- years;
- candidate IDs;
- search scores.
