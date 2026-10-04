# Labelling round 3: adding eating and drinking

The researcher wants six variables. Two of them (lying bouts, standing bouts)
came out of data we already had. The other four — drinking minutes, eating
minutes, drinking bouts, eating bouts — need the detector to tell those two
behaviours apart from plain standing, and that needs a new round of hand
labelling.

## Why the model cannot do this today

`models/calf_posture_v2.pt` knows two classes, `lying` and `standing`. A calf
with its head in the feed bucket is currently labelled `standing`, and so is a
calf asleep on its feet. The v2 label set contains **581 lying frames and 178
standing frames, and no feeding at all** — it was sampled from four segments of
28 July (04:37, 08:54, 15:20, 21:46), none of which covers a feeding.

## Why a zone rule is not enough

The obvious shortcut is to skip labelling and call it "eating" whenever the
calf's box overlaps the feed buckets. The buckets are in a fixed place, so this
is easy to write — `cowvision/labelset.py` already computes exactly this
overlap.

It does not work as a label. Pulling the 28 highest-overlap standing frames out
of the existing sample and looking at them, roughly half are a calf standing
*beside* or *under* the buckets with its head somewhere else entirely. The pen
is small and the buckets are on the wall the calf walks past; proximity is not
eating. The overlap is a good way to *find* frames worth labelling, which is
what it is used for, and a bad way to label them.

## The four classes

Keep one box per calf and put the behaviour in the class name, as now. The
detector then needs four classes:

| class | what it looks like from above |
|---|---|
| `lying` | body on the floor, as in v2 — unchanged |
| `standing` | upright, head not in a bucket and no bottle |
| `eating` | upright, **head down inside or directly over one of the two wall buckets**; the head is usually hidden by the bucket rim |
| `drinking` | upright at the pen front, **nose at the bottle**; a person's arm or body is normally in frame at the gate |

`config.json` already maps these:

- `posture.class_map` folds `eating` and `drinking` back into `standing`, so the
  lying/standing hours already delivered stay directly comparable.
- `posture.activity_map` keeps all four, and the eating/drinking minutes and
  bouts are built from that.

So nothing downstream changes shape — a 4-class model drops into the same
pipeline, and if the extra two classes were ever dropped the old numbers would
come back unchanged.

## Labelling rules

Decide these once and apply them to every frame, because they define what the
researcher's minutes actually mean.

1. **Eating is head-in-bucket, not near-bucket.** If the head is not over the
   bucket opening, it is `standing`. When the head is ambiguous, look at the
   neighbouring frames in the same segment: a calf that is eating stays put for
   several consecutive samples.
2. **Both wall buckets count as `eating`.** One is feed and one is water, but
   they cannot be told apart reliably from above and the researcher asked for
   eating at "the eating apparatus". Note this is therefore *eating + bucket
   drinking combined*, and say so when reporting the number.
3. **`drinking` means the milk bottle only**, matching the researcher's
   definition. It is the one behaviour with a person in frame, which is the
   easiest cue — if there is no person at the gate, it is almost certainly not
   bottle feeding.
4. **One box per calf, drawn around the calf**, not around the bucket and not
   around the person. Never label the person.
5. **Empty pen stays a null** (no box), exactly as in v1/v2. 31 such frames are
   already in the set and they matter — they are what keeps the model from
   inventing a calf in an empty pen.
6. **Left and right pens are different animals**; label each crop on its own.

## What to relabel and what to keep

- **Keep the 581 `lying` frames as they are.** Lying is lying; none of it is
  feeding.
- **Re-review all 178 `standing` frames.** Some of them are really eating. They
  are the cheapest source of `eating` examples because they are already in the
  set.
- **Add new frames from the feeding windows**, which is where `drinking` lives.
  None of the existing frames can contain a bottle feed.

## Which frames to add

The diurnal pattern across all 19 full days shows two unambiguous activity
peaks, which are the two feedings:

- **morning 06:30-08:00**, peaking 07:00-07:20 (standing rises from ~20% to 60-72%)
- **evening 18:30-20:00**, peaking 18:50-19:40 (standing rises from ~20% to 50-72%)

A uniform sample over a day would put well over 90% of its frames in an idle
pen and would likely contain no bottle feed at all. So sample those windows
only, and spread the days out so the model is not learning one afternoon's
lighting:

```
cowvision extract --videos <dir> --between 06:30-08:00,18:30-20:00
```

Six days spread across the study (29 Jul, 2, 6, 10, 14, 18 Aug) at one frame per
20 s gives 3 h of window per day per calf. `cowvision labelset` then removes
near-duplicates and fills a quota from each (side x posture x at-bucket) group,
so the rare "standing at the buckets" frames are taken whole instead of being
swamped by identical crops of a calf that has not moved.

Target roughly 600 new frames to label — the same order as the 286 hand-labelled
in round 1, allowing for the two new classes being rarer.

Exact commands are in `docs/server-runbook.md`.

## After labelling

1. Export from Roboflow as YOLOv8 into `../dataset_v3/`, `nc: 4`,
   `names: ['lying','standing','eating','drinking']`.
2. Retrain from `calf_posture_v2.pt` rather than from scratch — it already knows
   these two calves and this camera.
3. Check the confusion matrix specifically on `standing` vs `eating`. That is the
   pair the whole deliverable rests on, and it is the one the geometry cannot
   separate. Recall on `drinking` will be built on few examples; report it
   honestly, and if it is weak, say that drinking minutes are provisional rather
   than quietly shipping them.
4. Re-run `detect` and `minutely` on the server, then `cowvision bouts`. The
   eating/drinking minutes and bouts appear in `daily_summary.csv` and
   `bout_summary.csv` with no further code changes.

## One thing to confirm with the researcher

Whether the two wall buckets are feed and water, or two feeds. If one is water,
"eating time" as defined above silently includes water drinking, and the
researcher may want that split out — which would need a third bucket-level class
and is only worth doing if the two buckets are reliably distinguishable from
above. They are different colours in both pens, so it is probably possible, but
it is not worth the labelling effort unless the researcher asks.
