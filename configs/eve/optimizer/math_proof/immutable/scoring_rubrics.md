# Math Proof Scoring Rubrics

Use these rubrics when interpreting evaluator score dimensions in score cards
and proof reviews. They describe how the evaluator scores proof candidates.

## Coverage

`coverage` audits whether the candidate proof covers the required components of
the target theorem, independently of whether the written arguments for those
components are mathematically correct.

Do not lower coverage merely because a covered argument is mathematically
wrong; that belongs under correctness. Lower coverage when a necessary part of
the theorem is missing, only asserted, replaced by a weaker statement, or
replaced by a different statement.

Scoring rubric:

- `0-20`: The proof basically does not cover the target theorem: it is empty,
  irrelevant, only restates the problem, or proves a clearly different theorem.
- `20-40`: The proof has a relevant direction, background, or outline, but does
  not actually cover the main theorem obligations.
- `40-60`: The proof covers one nontrivial part, such as one case, object
  construction, reduction, or intermediate target, but main obligations remain
  missing.
- `60-80`: The proof covers most of the theorem structure, but misses a
  required case, hypothesis, edge condition, existence or uniqueness clause, or
  final assembly.
- `80-100`: The proof essentially covers the complete target statement,
  including hypotheses, quantifiers, cases, reductions, edge conditions, and
  final conclusion.

## Correctness

`correctness` audits whether the mathematical reasoning actually written in the
candidate proof is valid.

Do not use correctness to punish parts of the target theorem that are simply
missing; that belongs under coverage. Do lower correctness when the proof
falsely claims a missing part was handled, proves a different statement as if it
were the target, uses a circular argument, or contains a fatal local error.
Empty, content-free, or purely scaffold proof has correctness 0.

Scoring rubric:

- `0-20`: The written argument's core reasoning is invalid: the main conclusion
  does not follow from previous material, or the proof contains obvious false
  claims or circular reasoning.
- `20-40`: Some local reasoning is correct, but the main line contains a fatal
  invalid step that breaks the proof.
- `40-60`: Most local reasoning is understandable, but there is a serious
  invalid implication, quantifier use, case argument, algebraic manipulation, or
  contradiction argument.
- `60-80`: The written main line is mostly valid, but there are non-fatal
  mathematical errors, condition misuses, mistaken identifications, or local
  claims that do not follow.
- `80-100`: The written key reasoning is basically valid, with only minor,
  local, repairable wording issues or low-risk details.

Do not lower correctness because a case is missing or because external theorem
hypotheses were not checked, unless the proof makes a false internal inference.

## Dependency

`dependency` audits whether the proof's mathematical dependencies are
identifiable, stated accurately, and applicable in the way the proof uses them.

Do not lower dependency merely because a local argument is wrong; that belongs
under correctness. Lower dependency when the proof relies on an unnamed, vague,
misstated, inapplicable, circular, or unverified theorem, lemma, reduction, or
definition.

Scoring rubric:

- `0-20`: A key dependency is nonexistent, unidentifiable, fabricated, or the
  dependency chain is basically unauditable.
- `20-40`: The proof cites relevant-looking results or standard facts, but their
  statements are vague, sources are unclear, or the used version cannot be
  identified.
- `40-60`: The main dependencies are roughly relevant, but important hypotheses,
  normalizations, domain assumptions, or reduction applicability checks are
  missing.
- `60-80`: Most key dependencies are stated accurately and mostly applicable,
  but a few important dependencies still lack a statement, hypothesis check, or
  reference path.
- `80-100`: Key theorems, lemmas, reductions, and definitions are clearly
  stated; the used versions are accurate; hypotheses are checked; and the
  dependency chain is auditable.

Do not rejudge internal algebra, implications, or case reasoning for
dependency; those belong under correctness.

## Clarity

`clarity` audits how easy the candidate proof is for a competent mathematician
to follow and audit.

Do not lower clarity merely because the proof is mathematically wrong; that
belongs under correctness. Lower clarity when the writing obscures what is being
claimed, why a step is relevant, what objects mean, or how the pieces are
supposed to compose.

Scoring rubric:

- `0-20`: A reader cannot identify the proof structure, main objects, local
  goals, or final claimed result.
- `20-40`: The proof contains relevant mathematical content, but organization,
  notation, or claim presentation makes it basically unauditable.
- `40-60`: The overall route is visible, but the reader must reconstruct many
  definitions, case splits, dependency links, or proof-order relations.
- `60-80`: The structure is basically clear and most definitions and steps can
  be followed, but there are notation drift, ordering, signposting, or local
  goal issues.
- `80-100`: The proof route is clear, notation is stable, claims are easy to
  locate, and a reader can smoothly check each main proof unit.

Do not lower clarity because the proof is wrong, incomplete, or has unverified
dependencies; lower clarity only when the writing itself blocks audit.

## Strategy

`strategy` audits whether the candidate proof uses an appropriate high-level
route for the target theorem and whether it has obvious route-level improvement
opportunities.

Do not lower strategy merely because a local step is wrong; that belongs under
correctness. Do not lower strategy merely because a cited dependency is vague
or inapplicable; that belongs under dependency. Lower strategy when the proof
chooses a route that is unnecessarily brute-force, brittle, redundant,
off-target, or unlikely to scale, especially when a more natural structural
route is visible.

Scoring rubric:

- `0-20`: The proof route is essentially off-target, ad hoc, or has no credible
  path to the theorem.
- `20-40`: The route has some relevant ideas, but it is dominated by brittle
  enumeration, avoidable case splitting, or missing high-level structure.
- `40-60`: The route is plausible but inefficient or fragile; a clearer
  theorem, classification, invariant, or reduction target is likely needed.
- `60-80`: The route is mostly appropriate, with some missed simplifications or
  unnecessary technical burden.
- `80-100`: The proof uses a natural, robust route for the theorem, with good
  reductions and little avoidable redundancy.
