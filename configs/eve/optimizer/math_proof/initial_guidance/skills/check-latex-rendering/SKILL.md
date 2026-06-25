---
name: check-latex-rendering
description: "Check the proof under solver/proof/ for LaTeX source patterns that commonly break Eve's renderer."
---

Use this skill when checking a mathematical proof before reporting PASS.

1. Read the proof files under `solver/proof/`; do not edit them.
2. Check the LaTeX source for obvious rendering hazards:
   - missing or mismatched math delimiters: `\[` / `\]`, `\(` / `\)`, `$...$`,
     and `$$...$$`;
   - obvious unpaired `\left` / `\right` delimiters;
   - mismatched `\begin{...}` / `\end{...}` environments;
   - malformed `\begin` or `\end` commands;
   - LaTeX math environments left outside math delimiters, for example raw
     `\begin{pmatrix}` in prose or inside plain `[...]` brackets.
3. Ignore LaTeX shown inside Markdown code spans or fenced code blocks.
4. Report `LaTeX source review: PASS` if no obvious hazard is visible.
5. Report `LaTeX source review: FAIL` if you find a hazard, with line numbers
   and snippets so the parent solver can fix the exact formula.
