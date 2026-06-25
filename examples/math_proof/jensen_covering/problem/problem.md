# Problem: Jensen Covering for \(L\) from Fine Structure

## Objective

Work in \(V\models\mathrm{ZFC}\). Assume

\[
0^\#\text{ does not exist.}
\]

Prove Jensen's covering theorem for \(L\) in the strong form

\[
\forall X\subseteq\mathrm{Ord}\;\exists Y\in L\;
\bigl(X\subseteq Y\land |Y|^V\le |X|^V+\aleph_1^V\bigr).
\tag{JC}
\]

Then derive the uncountable covering form:

\[
\forall X\subseteq\mathrm{Ord}\,
\Bigl(|X|^V\ge\aleph_1^V
\Rightarrow
\exists Y\in L\,\bigl(X\subseteq Y\land |Y|^V=|X|^V\bigr)
\Bigr).
\tag{UC}
\]

Equivalently, under \(\neg 0^\#\), every uncountable set of ordinals in
\(V\) is covered by a constructible set of the same \(V\)-cardinality.

The proof must be a mathematical manuscript, not a survey. It must identify the
precise fine-structural principles used, prove the reduction from failure of
covering to indiscernibility, and explicitly construct the final covering set
\(Y\in L\).

---

## Permitted external aids

Web search, theorem-search, Matlas-search, library search, and reference search
are permitted for locating exact statements, terminology, sources, theorem
numbers, and notation conventions.

They may not be used to trivialize the proof. In particular, the final solution
may not cite, invoke, or paraphrase as a black box any result whose conclusion is
Jensen covering, local covering for \(L\), strong covering for \(L\), weak
covering specialized to \(L\), or any theorem immediately implying \((JC)\) or
\((UC)\).

Search-supported material is acceptable only if it is entered into an explicit
**import ledger**. For every imported result, the solution must record:

1. the exact statement used;
2. the source and theorem/proposition/lemma identifier, if available;
3. the notation translation into the manuscript's notation;
4. all hypotheses needed for the application;
5. a verification that the result is not equivalent to Jensen covering and is not
   proved in the cited source by invoking Jensen covering.

A citation is not a proof step unless the cited result is explicitly allowed
below or is separately proved in the manuscript.

---

## Allowed black-box fine structure

The problem is not to rederive the whole Jensen fine-structure theory from
first principles. The solution may import the following standard fine-structural
facts, provided each is stated exactly and entered into the import ledger.

### F1. Jensen \(J\)-hierarchy basics

The solution may use the standard construction and elementary closure properties
of the Jensen hierarchy

\[
J_0=\varnothing,\qquad
J_{\alpha+1}=\operatorname{rud}(J_\alpha\cup\{J_\alpha\}),\qquad
J_\lambda=\bigcup_{\alpha<\lambda}J_\alpha
\]

including rudimentary closure, the canonical \(L\)-well-order, and the usual
identification of \(L\) with \(\bigcup_{\alpha\in\mathrm{Ord}}J_\alpha\).

The manuscript must still define the notation it uses and prove any nonstandard
consequence of these facts.

### F2. Finite-level fine structure of \(J\)-levels

The solution may use the standard definitions and basic formal properties of:

- amenable structures over \(J_\alpha\);
- \(\Sigma_n\)-elementarity;
- canonical \(\Sigma_n\)-Skolem functions;
- \(\Sigma_n\)-Skolem hulls;
- projecta \(\rho_n(J_\alpha)\);
- standard parameters \(p_n(J_\alpha)\);
- \(n\)-soundness and \(n\)-solidity, if invoked.

These notions must not be introduced as decorative background. Each such notion
appearing in the manuscript must be used later in an essential proof step.

### F3. Fine-structural condensation

The solution may import Jensen's condensation theorem in the exact form needed:
roughly, that under the appropriate \(\Sigma_n\)-elementarity, soundness,
projectum, and standard-parameter hypotheses, the transitive collapse of the
relevant hull is an initial segment of the \(J\)-hierarchy, and the collapse map
is the corresponding fine-structural elementary embedding.

The imported condensation theorem must specify:

1. the language and structures involved;
2. the level of elementarity required;
3. the required projectum bound;
4. the role of the standard parameter;
5. the soundness/solidity hypotheses, if any;
6. the exact conclusion identifying the collapse as some \(J_{\bar\alpha}\) or
   standard code thereof.

Every later use of condensation must check these hypotheses explicitly.

### F4. \(0^\#\), Silver indiscernibles, and self-embeddings of \(L\)

The solution may import the standard equivalences among:

\[
0^\#\text{ exists},
\]

\[
L\text{ has a proper class of Silver indiscernibles},
\]

and the corresponding elementary self-similarity of \(L\), such as the existence
of nontrivial elementary embeddings of suitable elementary extensions or, in the
presence of a full class of indiscernibles, the shift map on Silver
indiscernibles.

The solution must still prove that the objects constructed from a failure of
covering satisfy the hypotheses of this imported equivalence.

---

## Results that may **not** be imported

The following may not be used as black boxes:

1. Jensen's covering theorem for \(L\), in any form.
2. The local cofinality form
   \[
   L\cap[\lambda]^{<\mu}\text{ is cofinal in }[\lambda]^{<\mu}
   \]
   for regular \(\mu>\aleph_1\).
3. Any theorem stating directly that failure of covering implies \(0^\#\).
4. Any theorem stating directly that failure of local covering yields Silver
   indiscernibles.
5. Core model covering specialized to \(K=L\).
6. Weak covering as a substitute for the required strong covering statement.
7. Any theorem whose proof in the cited source uses Jensen covering in the form
   being proved here.

The manuscript must prove the obstruction analysis itself: starting from a
minimal failure of local covering, it must construct the coherent elementary
self-similarity that yields \(0^\#\).

---

## Required proof architecture

The proof should proceed through the following local covering statement.

### Local covering target

For every regular cardinal \(\mu>\aleph_1^V\), every ordinal \(\lambda\), and
every \(a\in[\lambda]^{<\mu}\), prove that there exists

\[
b\in L\cap[\lambda]^{<\mu}
\]

such that

\[
a\subseteq b.
\tag{LC}_{\mu,\lambda}
\]

Then derive \((JC)\) by setting

\[
\mu=(|X|^V+\aleph_1^V)^+.
\]

The main labor of the proof is to establish \((LC)_{\mu,\lambda}\) under
\(\neg 0^\#\), without citing any covering theorem.

---

## Section requirements

### 1. Dependency roadmap

Begin with a roadmap listing all imported fine-structural results and all
lemmas proved in the manuscript.

The roadmap must distinguish between:

- imported fine-structure;
- lemmas proved from imported fine-structure;
- the obstruction analysis;
- the final cardinal-arithmetic reduction.

---

### 2. Fine-structural setup

Work uniformly with the Jensen \(J\)-hierarchy.

Define the structures in which hulls and collapses are formed. Specify the
language, predicates, parameters, and elementarity relation. In particular,
fix notation for:

\[
\operatorname{Hull}^{J_\theta}_{\Sigma_n}(A),
\]

\[
\rho_n(J_\theta),
\]

\[
p_n(J_\theta),
\]

and the inverse collapse map

\[
\pi_X:J_{\bar\theta_X}\longrightarrow J_\theta.
\]

This section must prove the basic hull consequences that will be used later,
including:

1. if \(X\) is the relevant \(\Sigma_n\)-hull, then \(|X|^V\) is bounded by the
   cardinality of the generating set plus \(\aleph_0\);
2. the transitive collapse is a fine-structural initial segment of \(L\), by the
   imported condensation theorem;
3. the map \(\pi_X\) preserves the relevant projecta and standard parameters;
4. intersections of the form \(X\cap\lambda\) are constructible whenever the
   collapse analysis implies that this intersection is coded in the collapsed
   \(J\)-level.

---

### 3. Minimal counterexample setup

Assume toward contradiction that local covering fails for some regular
\(\mu>\aleph_1^V\). Choose a counterexample

\[
(\mu,\lambda,a)
\]

with

\[
a\in[\lambda]^{<\mu}
\]

according to a fixed minimality scheme, for example:

1. first minimize \(\mu\), if needed;
2. then minimize \(\lambda\);
3. then minimize \(a\) in the canonical \(L\)-well-order among coded candidates;
4. finally choose the least sufficiently closed \(J_\theta\) witnessing the
   failure of the hull construction.

The exact minimality scheme is flexible, but it must be strong enough to prove
all later coherence claims. The manuscript must state the scheme precisely and
use it explicitly.

Prove that every proper initial segment of the counterexample is covered. In
particular, prove the appropriate form of:

\[
\forall \lambda'<\lambda\;\exists b_{\lambda'}\in L\cap[\lambda']^{<\mu}
\quad
(a\cap\lambda'\subseteq b_{\lambda'}).
\]

Explain exactly where regularity of \(\mu\) and the assumption
\(\mu>\aleph_1^V\) are used.

---

### 4. Good hulls and bad hulls

Define the class of hulls suitable for the covering argument. A suitable hull
should include the relevant parameters and satisfy enough fine-structural
closure to let condensation identify its collapse.

For each candidate hull \(X\prec_{\Sigma_n}J_\theta\), analyze:

1. its generating set;
2. its cardinality in \(V\);
3. its transitive collapse;
4. the inverse collapse embedding;
5. the ordinal \(X\cap\lambda\);
6. whether \(X\cap\lambda\in L\);
7. whether \(a\subseteq X\cap\lambda\).

Show that if a suitable hull \(X\) exists with

\[
a\subseteq X\cap\lambda
\qquad\text{and}\qquad
|X|^V<\mu,
\]

then local covering holds by taking

\[
b=X\cap\lambda.
\]

Thus, under the assumed failure of local covering, every attempted suitable hull
must fail in a precise way. Isolate this failure as a fine-structural obstruction
rather than as a vague lack of closure.

---

### 5. Basic construction from the obstruction

From the minimal counterexample and the obstruction isolated above, construct a
coherent directed system of fine-structural elementary embeddings

\[
\left\langle
M_\xi,\pi_{\xi\eta}:M_\xi\to M_\eta
\mid \xi\le\eta<\Gamma
\right\rangle
\]

for arbitrarily large ordinals \(\Gamma\), where each \(M_\xi\) is a
fine-structural initial segment of \(L\), or a standard code of one.

The construction must prove:

1. \(\pi_{\xi\xi}=\operatorname{id}\);
2. \(\pi_{\eta\zeta}\circ\pi_{\xi\eta}=\pi_{\xi\zeta}\);
3. the embeddings preserve the relevant projecta and standard parameters;
4. successor steps are nontrivial whenever required;
5. direct limits at limit stages are well-founded;
6. the system is continuous at limits of sufficiently large cofinality;
7. the construction can be continued to arbitrary length unless covering holds.

The well-foundedness argument must be included. It is not acceptable to say
"take the direct limit" without proving that the direct limit is well-founded in
the cases used.

This section is the core of the problem. It may use the imported condensation
and finite-level fine structure, but it may not cite a theorem saying that the
basic construction yields covering or yields \(0^\#\).

---

### 6. Critical points and indiscernibility

For each nontrivial successor map, let

\[
\kappa_\xi=\operatorname{crit}(\pi_{\xi,\xi+1}).
\]

Prove that the sequence of critical points is coherent:

\[
\pi_{\xi\eta}(\kappa_\xi)=\kappa_\eta
\quad\text{whenever this expression is defined.}
\]

Then prove that a proper class extracted from the \(\kappa_\xi\)'s is a class of
Silver indiscernibles for \(L\), or satisfies the exact hypothesis of the
imported \(0^\#\)-criterion.

The proof must verify indiscernibility formula-by-formula:
for every first-order formula \(\varphi(v_1,\dots,v_m)\) and increasing tuples

\[
\xi_1<\cdots<\xi_m,
\qquad
\eta_1<\cdots<\eta_m,
\]

from the constructed class, show that

\[
L\models\varphi(\kappa_{\xi_1},\dots,\kappa_{\xi_m})
\]

if and only if

\[
L\models\varphi(\kappa_{\eta_1},\dots,\kappa_{\eta_m}).
\]

It is acceptable to prove this first inside sufficiently large \(J_\alpha\)'s and
then pass to \(L\), but the passage to \(L\) must be justified.

Conclude, using the imported \(0^\#\)-criterion, that the assumed failure of
local covering implies

\[
0^\#\text{ exists.}
\]

This contradicts the hypothesis. Therefore local covering holds.

---

### 7. Derivation of global covering

Assume local covering. Let \(X\subseteq\mathrm{Ord}\), and put

\[
\kappa=|X|^V.
\]

Choose an ordinal \(\lambda\) with \(X\subseteq\lambda\), and let

\[
\mu=(\kappa+\aleph_1^V)^+.
\]

Apply local covering to obtain

\[
Y\in L\cap[\lambda]^{<\mu}
\]

such that \(X\subseteq Y\). Prove explicitly that

\[
|Y|^V\le\kappa+\aleph_1^V.
\]

Then prove the uncountable form by observing that if \(\kappa\ge\aleph_1^V\),
then

\[
\kappa+\aleph_1^V=\kappa,
\]

and since \(X\subseteq Y\), one has \(|Y|^V=\kappa\).

The proof must handle singular \(\kappa\) without replacing \(\kappa\) by a
regular cardinal of the same size. The regular cardinal used in the local
covering argument is \((\kappa+\aleph_1^V)^+\), not \(\kappa\) itself.

---

### 8. Sharpness when \(0^\#\) exists

After proving the theorem, prove that the hypothesis is sharp.

Using the imported equivalence between \(0^\#\) and Silver indiscernibles,
explain how a proper class of Silver indiscernibles yields elementary
self-similarity of \(L\), for example through the shift

\[
i_\alpha\mapsto i_{\alpha+1}.
\]

Then exhibit an uncountable set of ordinals \(X\) such that no constructible
\(Y\) of the same \(V\)-cardinality covers \(X\), or state and prove an
appropriate local failure of covering derived from the indiscernibles.

This section must distinguish sharply between:

- indiscernibles as external objects in \(V\);
- sets belonging to \(L\);
- cardinality computed in \(V\);
- cardinality computed in \(L\).

---

### 9. Consequences

Derive at least two consequences of Jensen covering.

At minimum include the following kinds of consequences.

#### 9.1 Successor-cardinal consequence

State and prove a precise comparison theorem of the form:

\[
(\kappa^+)^L=\kappa^+
\]

under correct hypotheses on \(\kappa\). The proof must specify whether \(\kappa\)
is a cardinal of \(V\), a cardinal of \(L\), regular, singular, uncountable, or
otherwise constrained.

The proof must distinguish between

\[
\kappa^+\text{ in }V
\qquad\text{and}\qquad
(\kappa^+)^L.
\]

#### 9.2 Singular-cardinal/cardinal-arithmetic consequence

State and prove a singular-cardinal consequence, such as the standard relation
between Jensen covering and SCH. If a separate cardinal-arithmetic theorem is
used to derive SCH from covering, state it explicitly and include it in the
import ledger unless it is proved in the manuscript.

The conclusion must specify whether it is a statement of \(V\), \(L\), or a
comparison between \(V\) and \(L\).

---

## Prohibited shortcuts

A solution is unacceptable if it:

- cites Jensen covering to prove Jensen covering;
- cites local covering for \(L\) to prove local covering for \(L\);
- replaces strong covering by weak covering;
- replaces \(L\)-covering by core model covering;
- says "by fine structure" without naming the exact lemma and verifying its
  hypotheses;
- invokes condensation without stating the exact condensation theorem being used;
- forms a Skolem hull without proving that its collapse is the relevant
  fine-structural initial segment;
- blurs \(|\cdot|^L\) and \(|\cdot|^V\);
- ignores singular cardinals;
- treats Silver indiscernibles as arbitrary indiscernibles;
- treats search results, encyclopedia entries, lecture notes, or theorem names as
  proof steps;
- imports any theorem whose conclusion is equivalent to the theorem being
  proved.

---

## Required output format

Write the solution as a mathematical manuscript.

Use numbered definitions, lemmas, propositions, and theorems.

The manuscript must contain:

1. a dependency roadmap at the beginning;
2. an import ledger for all externally used fine-structural facts;
3. a precise local covering theorem;
4. a minimal-counterexample analysis;
5. the basic construction from the obstruction;
6. the derivation of indiscernibles or the exact \(0^\#\)-criterion;
7. the cardinal-arithmetic reduction from local to global covering;
8. the uncountable-covering corollary;
9. sharpness when \(0^\#\) exists;
10. at least two consequences;
11. a final dependency graph showing exactly which lemmas imply Jensen covering.

The final theorem must be exactly \((JC)\), followed by \((UC)\) as a corollary.
The covering set \(Y\in L\) must be explicitly obtained from the local covering
construction and its cardinality must be computed in \(V\).
