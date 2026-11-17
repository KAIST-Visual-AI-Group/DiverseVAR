## VBench-2.0 score summaries

After evaluating all configured dimensions, run:

```sh
python scripts/parse_vbench2_eval_results.py EVAL_DIR
```

Each dimension needs its underscore-named directory, an `*_eval_results.json`
file, and a matching dimension key. Both `Camera_Motion` and `Camera Motion`
(and the corresponding spellings for other dimensions) are supported. A score
can be a numeric scalar or the first element of a nonempty score/details list.
If both key spellings appear, the underscore spelling retains precedence.

Missing directories/files cause a nonzero exit with dimension-specific
diagnostics instead of zero-filled aggregate scores. Empty, malformed,
unrelated-key, or non-finite results are errors. A measured zero is valid.

For intentional subsets, such as the README's Diversity-only generation example:

```sh
python scripts/parse_vbench2_eval_results.py EVAL_DIR --allow-partial
```

This prints available per-dimension scores and coverage (for example, `1/18`),
names missing evaluations on stderr, and withholds category, total and flat-mean
aggregates when incomplete. It never fills missing scores with zero or skips
malformed result files. Complete inputs produce the original scores in either
mode.

The parser currently configures 18 dimensions; its existing complete-output
label `flat mean (19d)` is retained for compatibility. Its divisor and benchmark
formulas are unchanged.
