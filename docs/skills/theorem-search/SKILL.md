---
name: theorem-search
description: "Use TheoremSearch to find, verify, and cite mathematical theorem statements, named results, proof routes, and literature dependencies."
---

# TheoremSearch Manual

Use this skill when a proof needs:

- a named theorem, lemma, proposition, corollary, definition, or remark;
- a literature dependency for a proof step;
- confirmation that a remembered theorem statement has the right hypotheses;
- leads for an external proof route;
- paper or theorem provenance for later human review.

TheoremSearch is supporting evidence only. It does not prove that a dependency
is valid for the current proof. Always compare the returned statement against
the exact proof obligation.

Official docs:

- Web docs: `https://www.theoremsearch.com/docs`
- API base: `https://api.theoremsearch.com`
- Main search endpoint: `POST https://api.theoremsearch.com/search`
- MCP endpoint: `https://api.theoremsearch.com/mcp`

Default to the REST API in this workspace: it is directly callable with `curl`,
requires no agent-runtime configuration, and returns the search payload without
protocol wrapping. Use MCP when the current agent runtime already has the
TheoremSearch MCP server configured and tested.

## Mental Model

TheoremSearch indexes theorem-level statements, not just whole papers. A query is
embedded, matched against theorem slogans, and reranked. The best hit may be:

- the exact result;
- a nearby theorem with missing or extra hypotheses;
- a related result in the right paper;
- a statement whose generated slogan is more relevant than its full body.

Therefore, trust `body` and paper metadata more than `slogan`, `name`, or rank.

## Basic Search

Use natural language, LaTeX, or a mix. Start broad enough to find the theorem,
then narrow with filters or a sharper query.

```bash
curl -sS https://api.theoremsearch.com/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "every projective module over a local ring is free",
    "n_results": 5
  }'
```

Expected top-level shape:

```json
{
  "theorems": [
    {
      "slogan_id": 7725846,
      "theorem_id": 22812802,
      "name": "Theorem 5.11",
      "body": "If $P$ is a projective module over a local ring $R$, then $P$ is free.",
      "slogan": "A projective module over a local ring is free.",
      "theorem_type": "theorem",
      "label": null,
      "link": null,
      "paper": {
        "paper_id": "1011.0038v1",
        "source": "arXiv",
        "title": "Faithfully flat descent for projectivity of modules",
        "authors": ["Alexander Perry"],
        "link": "http://arxiv.org/abs/1011.0038v1",
        "primary_category": "math.AC",
        "categories": ["math.AC"],
        "citations": 0,
        "year": 2010,
        "journal_published": false
      },
      "similarity": 0.9086642212279158,
      "score": 0.9086642212279158,
      "has_metadata": true
    }
  ]
}
```

## Search Request Fields

Required:

- `query`: natural-language or LaTeX description of the result.

Common optional fields:

- `n_results`: number of theorem results to return. Default is `10`.
- `sources`: source filter. Officially documented examples include `arXiv`,
  `Stacks Project`, `ProofWiki`, `CRing Project`, `HoTT Book`,
  `Open Logic Project`, and `An Infinitely Large Napkin`. The live API may
  expose additional sources over time.
- `authors`: partial author-name filters.
- `types`: theorem-like object filters, for example `theorem`, `lemma`,
  `proposition`, `corollary`, `definition`, or `remark`. Results may preserve
  source casing inconsistently, so if a type filter misses, retry without it.
- `tags`: subject tags, especially arXiv categories such as `math.AG`,
  `math.AC`, `math.NT`, `math.LO`.
- `paper_filter`: substring filter over paper titles.
- `year_range`: inclusive `[min_year, max_year]`.
- `citation_range`: inclusive `[min_citations, max_citations]`.
- `include_unknown_citations`: when using `citation_range`, keep papers with no
  citation data. Default is `true`.
- `citation_weight`: blend citation count into ranking. Use `0.0` for pure
  semantic search. Increase only when standard/high-citation sources matter.
- `prompt`: optional instruction prepended before embedding. Use sparingly.
- `db_top_k`: candidate-pool size before reranking. Defaults to `2 * n_results`.
  Increase when recall matters more than latency.

Filtered example:

```bash
curl -sS https://api.theoremsearch.com/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "smooth Deligne Mumford stack has a dense open subscheme",
    "n_results": 8,
    "sources": ["Stacks Project", "arXiv"],
    "tags": ["math.AG"],
    "year_range": [2000, 2026],
    "include_unknown_citations": true,
    "db_top_k": 40
  }'
```

## Query Strategy

1. Write the proof obligation as a theorem statement:
   "If [hypotheses], then [conclusion]."
2. Run an unfiltered search with `n_results` between 5 and 10.
3. Read `body` for every plausible hit. Ignore hits whose assumptions do not
   match, even if the slogan looks perfect.
4. If results are too broad, add one constraint at a time:
   `sources`, `tags`, `authors`, `paper_filter`, or `types`.
5. If results are too narrow or empty, remove filters, simplify notation, and
   search for equivalent phrases.
6. If a result is close but not exact, search again using distinctive words from
   the returned `body`, paper title, or named theorem.
7. Keep at least one exact or near-exact provenance record for every external
   dependency used in the final proof.

Good query patterns:

- Include the mathematical objects and the conclusion:
  `"finite etale morphism is open and closed"`
- Include key hypotheses:
  `"normal noetherian domain height one localization discrete valuation ring"`
- Use standard aliases:
  `"Nakayama lemma finitely generated module local ring"`
- Search by paper title plus theorem content when you know the source:
  `"EGA flat morphism openness theorem"`

Poor query patterns:

- `"prove this"` or `"main theorem"`
- overly local notation without context, such as `"X is dense in U"`
- long proof paragraphs with irrelevant narrative;
- filter-heavy first attempts that hide useful hits.

## Reading Results

For each plausible theorem, inspect:

- `body`: exact theorem statement. This is the primary evidence.
- `name`: local theorem label, for example `Theorem 5.11`.
- `theorem_type`: theorem, lemma, proposition, corollary, definition, remark.
- `slogan`: generated natural-language summary. Useful for search, not proof.
- `link`: statement-level link if available.
- `paper.source`: corpus/source, for example arXiv or Stacks Project.
- `paper.title`: paper or project title.
- `paper.authors`: authors, when available.
- `paper.link`: paper-level URL, often arXiv.
- `paper.primary_category` and `paper.categories`: subject categories.
- `paper.year`, `paper.journal_ref`, `paper.journal_published`.
- `paper.citations`: citation count if known.
- `similarity`: semantic similarity before citation weighting.
- `score`: final ranking score after optional weighting.
- `has_metadata`: whether richer metadata was available.

Do not treat score thresholds as correctness thresholds. A low-scoring exact
statement can be useful; a high-scoring result can still have wrong hypotheses.

## Validation Checklist

Before using a result in a proof, check:

- objects match: rings, schemes, stacks, categories, fields, modules, spaces;
- hypotheses match: finiteness, noetherian, separated, proper, smooth, local,
  algebraically closed, characteristic, completeness, boundedness;
- conclusion matches: existence, uniqueness, equivalence, isomorphism,
  vanishing, density, openness, freeness, exactness;
- variance and direction match: source/target, left/right, covariant/
  contravariant, injection/surjection;
- base category and notation match the current proof;
- no hidden condition is buried in surrounding paper notation;
- the result is appropriate to cite as a theorem rather than as a lead.

If a dependency is only close, write it as unresolved or as a search lead. Do
not silently strengthen or weaken theorem statements.

## Reporting Provenance

When a theorem is used, record enough information for audit:

```text
Dependency: projective modules over local rings are free
TheoremSearch hit: Theorem 5.11
Statement: If $P$ is a projective module over a local ring $R$, then $P$ is free.
Source: arXiv, "Faithfully flat descent for projectivity of modules"
Authors: Alexander Perry
Year: 2010
Link: http://arxiv.org/abs/1011.0038v1
Score: 0.9086642212279158
Assessment: exact match for the required algebra lemma.
```

If the proof only needs a standard result and the search hit is a pointer to a
secondary paper, prefer citing the classical theorem name in the proof while
keeping the search provenance in notes.

## Paper Search

Use paper search when you know a title fragment or arXiv ID and need a canonical
paper identifier.

```bash
curl -sS 'https://api.theoremsearch.com/paper-search?q=derived%20categories&limit=5'
```

Expected shape:

```json
{
  "papers": [
    {
      "paper_id": "f23156d8-433b-4f66-8b7b-8b58faabea6d",
      "title": "A Course on Derived Categories",
      "external_id": "1206.6632",
      "source": "arXiv"
    }
  ]
}
```

This endpoint is autocomplete-like. It is useful for locating a paper but does
not replace theorem-level search.

## Graph and Detail Endpoints

The public website documents dependency graph exploration. The live OpenAPI
schema exposes these graph/detail paths:

- `GET /graph/paper?external_id=<id>&sources=<source>`
- `GET /graph/paper/{paper_id}`
- `GET /graph/statement/{statement_id}`
- `GET /graph/embedding?query=<query>`
- `GET /graph/pagerank?query=<query>`
- `GET /statement/{statement_id}`
- `GET /paper/{paper_id}`
- `POST /statements` with `{"ids": [...]}`
- `POST /papers` with `{"ids": [...]}`

Use these only as auxiliary tools after search has identified a likely paper or
statement. Availability can vary by corpus and deployment; a `404` or empty
graph does not invalidate a theorem search hit.

Paper graph example:

```bash
curl -sS 'https://api.theoremsearch.com/graph/paper?external_id=1011.0038&sources=arXiv&mode=minimal'
```

Statement graph example:

```bash
curl -sS 'https://api.theoremsearch.com/graph/statement/<statement_id>?direction=src&formality=informal&mode=full'
```

Use graph output to discover:

- what a theorem depends on;
- what later statements depend on it;
- nearby restatements or equivalent results;
- whether a paper's internal lemmas form a usable proof route.

Do not cite graph edges as mathematical facts unless the corresponding
statements are inspected.

## MCP Reference

TheoremSearch also exposes an MCP server at:

```text
https://api.theoremsearch.com/mcp
```

The MCP server has a single `theorem_search` tool with the same parameters as
`POST /search`. MCP is useful when the agent runtime is already configured to
call MCP tools; REST is simpler for shell-based proof work.

Example MCP JSON-RPC request, for reference only:

```bash
curl -sS -X POST https://api.theoremsearch.com/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/call",
    "params": {
      "name": "theorem_search",
      "arguments": {
        "query": "every projective module over a local ring is free",
        "n_results": 5
      }
    }
  }'
```

## Failure Policy

If search fails:

- Network/server/API error: report `TheoremSearch unavailable` and include the
  endpoint and error text.
- Empty results: retry once with fewer filters and simpler language.
- Low relevance: report `unresolved`, summarize the closest hit, and explain
  the mismatch.
- Conflicting hits: list the competing statements and the hypothesis conflict.
- Missing links or metadata: use available paper title/source/body, but say what
  is missing.
- Graph endpoint failure: continue with `/search`; graph is auxiliary.

Never fabricate:

- theorem names;
- statement bodies;
- authors;
- links;
- scores;
- years;
- dependency edges;
- paper metadata.

## Final Answer Pattern

When reporting a search to a parent solver, use one of these concise outcomes:

```text
TheoremSearch: RESOLVED
Need: <proof obligation>
Result: <name/type>
Statement: <exact body>
Source: <source>, <paper title>, <authors>, <year>, <link>
Score: <score>
Use: <why the hypotheses and conclusion match>
```

```text
TheoremSearch: PARTIAL
Need: <proof obligation>
Closest result: <name/type and exact body>
Source: <metadata>
Mismatch: <specific missing/extra hypothesis or wrong conclusion>
Next query: <sharper query or filter>
```

```text
TheoremSearch: UNRESOLVED
Need: <proof obligation>
Searches tried: <queries/filters>
Reason: <empty, low relevance, inconsistent, unavailable>
```
