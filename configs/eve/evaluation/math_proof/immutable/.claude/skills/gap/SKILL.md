---
name: gap
description: "Use when evaluating candidate proof claims about finite groups, character tables, conjugacy data, finite fields, or other exact discrete-algebra computations that can be reproduced with a local GAP3/GAP4 CLI."
---

# GAP

GAP is a system for computational discrete algebra. It is especially useful for
finite groups, permutation groups, character tables, conjugacy classes, finite
fields, and related exact algebra calculations.

Use GAP as an evaluation aid when a candidate proof contains finite algebra
computations, GAP-backed evidence, or explicit tables that can be checked by
small exact computations. The examples above are common uses, not an exhaustive
boundary.

## Availability

GAP is optional in this evaluation workspace. Users may have installed neither
GAP CLI, only GAP4, only GAP3, or both. Treat the two command-line tools
independently:

- `gap` is the expected GAP4 launcher. Do not assume a specific GAP4 version.
- `gap3` is the expected GAP3 launcher. Do not assume a specific GAP3 version.

From the evaluation workspace root, check GAP4 availability with:

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
unavailable and do not claim GAP-backed evaluation evidence from it. If a local
module, container, or launcher is already available for a GAP version in this
run, use it and record the setup command with the check output.

## When To Use

Use GAP to rerun or spot-check exact finite algebra computations in candidate
proofs, especially when the claim can be reduced to finite groups, character
tables, conjugacy data, or small explicit cases.

Prefer GAP4 (`gap`) for ordinary modern GAP computations. Use GAP3 (`gap3`)
when the calculation depends on GAP3-only libraries, syntax, or historical
behavior. If both are installed and the distinction matters, state which CLI
was used.

Do not use GAP output as a substitute for judging the written proof. Use it to
check reproducibility, find likely errors, and assess whether computational
claims under `solver/proof/` support the candidate's argument.

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

This makes the computation easier to inspect, rerun, and preserve under
`solver/evaluation/<auditor-name>/` when it affects an evaluator finding.

## Artifact Policy

Do not edit `solver/proof/`. It is the read-only candidate proof.

Scratch evaluation experiments may live under `logs/evaluate/`.

If a GAP computation materially affects an auditor review or score, record the
command, relevant input, output, and interpretation under
`solver/evaluation/<auditor-name>/` and cite that evaluator artifact in the
review. A candidate proof claim whose only supporting artifact is under
`logs/optimize/` should not receive reproducibility credit as if it were part of
the submitted proof.
