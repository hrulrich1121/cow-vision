# Choosing the minimum bout length

A "bout" is a continuous stretch of one activity. Counting them needs one
decision that the video cannot make for us: **how brief an interruption still
counts as interrupting the bout.** A calf that shifts its weight for twenty
seconds has not ended its lying bout, but somewhere between twenty seconds and
twenty minutes it has.

This threshold is `bouts.min_bout_s` in `config.json`. It matters a lot, so it
is worth being explicit about what it does and what it was set to.

## What the threshold does

`aggregate.bouts()` does not simply discard runs shorter than the threshold —
that would leave the stretches either side of a discarded run as two separate
bouts, so a single flickering minute would still add one to the count. Instead
the shortest sub-threshold run is repeatedly **absorbed into its longer
neighbour** until everything left clears the floor (`_merge_short_runs`). This
is the usual convention in lying-behaviour work.

Two other settings sit alongside it:

- `bouts.max_unknown_s` (default 120 s) — how long the detector may lose the
  calf before the bout is treated as over. Shorter undetected patches are
  bridged; this is a statement about the detector, not about the calf.
- `bouts.max_gap_s` (default: 3x the sample spacing) — a break in the *footage*.
  Bouts never bridge one, so a missing segment always splits a bout rather than
  silently fusing the hours either side of it.

## Sensitivity

Measured on `results/minute_activity.csv`, 37 full calf-days (>=23 h observed),
19 days for the left calf and 18 for the right:

| floor | left bouts/d | right bouts/d | mean bout (min) | lying h/d |
|---|---|---|---|---|
| 1 min | 48.4 | 42.8 | 22 | 16.4 |
| 2 min | 27.9 | 28.8 | 35 | 16.4 |
| 3 min | 22.5 | 23.1 | 44 | 16.4 |
| 4 min | 18.9 | 20.4 | 50 | 16.4 |
| **5 min** | **16.4** | **18.6** | **57** | **16.4** |
| 7 min | 12.7 | 15.9 | 69 | 16.5 |
| 10 min | 9.5 | 11.6 | 95 | 16.6 |

Two things to read off this table:

1. **Total lying time is flat at 16.4 h/day regardless of the threshold.** The
   merge rule moves time between bouts, it does not create or destroy it. So the
   daily lying/standing *hours* already reported to the researcher are unaffected
   by this choice — only the bout counts depend on it.
2. **The bout count is almost entirely a product of the threshold.** Quoting a
   bout count without quoting the threshold beside it is meaningless.

## Why 5 minutes

Below about 3 minutes the counts are dominated by single-sample flips rather
than by real behaviour: at a 1-minute floor, 21% of all "bouts" are exactly one
minute long and the mean lying bout falls to 22 minutes, which is not what a
calf does. Above about 7 minutes genuine short bouts start being swallowed.

5 minutes puts both calves at 16-19 lying bouts/day with a mean bout near an
hour, which sits inside the range usually reported for individually housed
pre-weaned dairy calves (roughly 15-30 bouts/day, 16-19 h/day). 3 and 4 minutes
are also defensible and the researcher may prefer one of them for comparability
with a particular paper — it is one number in `config.json`, and re-running
`cowvision bouts` takes a few seconds.

## A caveat about resolution

The numbers above come from `minute_activity.csv`, which is already a per-minute
majority vote, so the finest interruption it can represent is one minute. The
per-frame `detections.csv` on the server is sampled at 1 fps and would resolve
shorter interruptions. Re-running `cowvision bouts` against the per-date
`detections.csv` files is the better basis for a publication number; expect
somewhat *more* bouts, because interruptions shorter than a minute become
visible. The code path is identical — only the source file changes.
