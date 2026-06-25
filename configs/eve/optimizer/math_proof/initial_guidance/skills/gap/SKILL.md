---
name: gap
description: "Use when a proof step involves finite groups, character tables, conjugacy data, finite fields, or other exact discrete-algebra computations that can be checked or explored with a local GAP3/GAP4 CLI."
---

# GAP

GAP is a system for computational discrete algebra. It is especially useful for
finite groups, permutation groups, character tables, conjugacy classes, finite
fields, and related exact algebra calculations.

Use GAP as a computational aid when a proof would benefit from exact finite
algebra experiments, counterexample searches, or regression checks. The examples
above are common uses, not an exhaustive boundary.

## Availability

GAP is optional in this workspace. Users may have installed neither GAP CLI,
only GAP4, only GAP3, or both. Treat the two command-line tools independently:

- `gap` is the expected GAP4 launcher. Do not assume a specific GAP4 version.
- `gap3` is the expected GAP3 launcher. Do not assume a specific GAP3 version.

From the workspace root, check GAP4 availability with:

```bash
command -v gap
gap -q -c 'Print(Size(SymmetricGroup(3)),"\n"); QUIT;'
```

Optionally inspect the GAP4 version:

```bash
gap --version
```

Check GAP3 availability separately. GAP3 does not support the GAP4 `-c` or
`--version` command-line options, so use stdin or a script file:

```bash
command -v gap3
printf 'Print(Size(SymmetricGroup(3)),"\\n"); quit;\n' | gap3 -q
```

If a launcher is missing or its smoke check fails, treat that GAP version as
unavailable and do not claim evidence from it.

## When To Use

Use GAP to check finite examples, search for counterexamples, compute small
character tables, verify conjugacy-class counts, test explicit formulas for
small parameters, or run other exact discrete-algebra computations that GAP is
well suited for.

Prefer GAP4 (`gap`) for ordinary modern GAP computations. Use GAP3 (`gap3`)
when the calculation depends on GAP3-only libraries, syntax, or historical
behavior. If both are installed and the distinction matters, state which CLI
was used.

Do not use GAP output as a substitute for a mathematical proof. Use it to find
mistakes, guide conjectures, and support symbolic arguments that are written in
the proof.

## How To Use

For any nontrivial or multi-line calculation, write a `.g` script file and run
it with the appropriate GAP CLI instead of packing the whole calculation into a
shell one-liner:

```bash
# GAP4
gap -q path/to/check.g

# GAP3
gap3 -q path/to/check.g
```

This makes the computation easier to inspect, rerun, and move into
`solver/proof/` when it supports a submitted proof claim.

## Artifact Policy

Scratch experiments may live under `logs/optimize/`.

If a GAP computation supports a submitted proof claim, put the GAP script,
input data, generated table, or summarized output under `solver/proof/`, then
cite the `solver/proof/...` path from the proof text. Do not rely on a claim
whose only evidence is under `logs/optimize/`.
