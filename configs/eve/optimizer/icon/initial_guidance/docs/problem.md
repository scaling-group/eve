ICON is a transformer-based in-context operator network trained on `weno_cubic`.
In this example, training uses `num_examples=5`, then evaluation measures error across
`d1..d10` example counts. The practical goal is simple: find a positional
encoding change that helps the model generalize across example count without
damaging the short-context regime.

The solver score is the negative of `mean_d1_d10`, so lower average error on
`d1..d10` means a better score. Keep an eye on `mean_d1_d4` and the full
`d1..d10` curve as diagnostics: a candidate that improves the average by
collapsing the short-context regime is not a satisfying result.

The training budget is intentionally modest: 2k steps, 2-GPU DDP, bf16 mixed
precision, roughly one iteration-scale experiment.
