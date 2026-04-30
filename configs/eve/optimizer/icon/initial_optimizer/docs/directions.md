# Directions

This file is a family enumeration, not a ranked recommendation. Use it to map
the current search space and to notice when one family is dominating local
evidence too strongly.

## 1. Sinusoidal absolute

Fixed sinusoidal bases give a simple, interpretable position signal with no
learned position table. They are cheap to try and often make extrapolation
behavior easier to reason about than fully learned embeddings.

## 2. Learned absolute

This is the vanilla baseline family: each position gets a learned embedding.
Outside the trained range, behavior depends on how the learned table and any
interpolation or extension logic generalize beyond seen rows.

## 3. Rotary phase encoding (RoPE family)

RoPE-style methods encode relative phase through query/key rotations instead of
adding a pure absolute embedding. Variants such as NTK-style scaling or YaRN
modify the frequency schedule so longer contexts stay better conditioned.

## 4. Attention bias (ALiBi family)

These methods add a distance-aware bias directly to the attention logits rather
than rotating token representations. They can preserve simple extrapolation
behavior because the bias keeps extending with distance instead of relying on a
fixed embedding table.

## 5. Relative position bias

Relative bias families model distance effects directly in attention without
requiring an explicit absolute-position embedding. They are a natural probe when
the main question is how strongly pairwise distance should shape retrieval.

## 6. Position interpolation / context-length scaling

These methods map long-context positions back toward the coordinate range seen
in training. The key question is whether the model already has usable behavior
that is being lost only because the raw index scale grows too far out of range.

## 7. Structured / hierarchical position decomposition

When the input has known semantic structure, position can be decomposed into
multiple axes rather than a single flat index. Classic examples include BERT's
segment embeddings for sentence pairs, TAPAS's row/column/rank for tables, and
LayoutLM's 2D coordinates for documents. For ICON, the natural decomposition
follows the per-example structure: example identity vs within-example token role. This
family asks whether explicit structured PE gives a cleaner inductive bias than
a single flat position index.

## 8. Hybrid / gated mechanisms

Hybrid families combine two or more mechanisms or learn when to switch between
them. These probes are higher variance but can be useful once the examples
suggest two families each capture part of the behavior we want.

## Meta-rule

This family list is a starting map, not a closed set.

Stalled running best usually reveals a narrowed search, not an exhausted
surface. When recent iterations stop advancing, broaden the family before
deepening local refinement.

Do-Not-Promote notes are local memory of past attempts, not permanent family
bans.

Positional encoding for OOD generalization is a well-studied area and web
search is available. Reaching outside the visible set is a first-class
research move. Research findings accumulate across iterations in the sibling
`literature_notes.md` — entries there carry forward, so anything you learn
from literature is worth recording.
