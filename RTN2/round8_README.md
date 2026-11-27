# Round 8: the RTN skeleton as one module, tested on three encoders (COCO-MLT)

## Skeleton version 2 (current)

**What changed (one component).** In component 3, a negative's relation
counts only the labels **the anchor does not have**. Version 1's diagnostics
showed that RTN pushed down true positives that also contain a look-alike
class. The cause: an anchor with a fire hydrant *and* a car pushed car-only
images away twice as hard in its fire-hydrant term, while its own car term
pulls them closer. The shared car explains why such a negative looks like
the anchor, so it is not confusion, and version 2 leaves it at weight 1.
Everything else is exactly as in version 1: relation source, relation
values, alpha = 1, host, split, seeds, measurements and decision rule.

**What to replace.** In your `RTN/` folder, replace these three files with
the new ones. Keep a single copy of each.

- `rtn_skeleton.py` (version 2)
- `rtn_diagnostics.py`
- `coco_mlt_rtn_diagnostics_round8.ipynb`

Do **not** change or replace the four encoder and summary notebooks: they
just call the skeleton. Do not delete anything either. Version 1's runs and
results stay where they are.

**What to run** (Run All in each, one at a time):

1. `coco_mlt_rtn_clip_round8.ipynb`, `coco_mlt_rtn_vit_b16_round8.ipynb`,
   `coco_mlt_rtn_resnet50_round8.ipynb`. Each one trains 3 new RTN runs
   (about 1.5 h with a GPU). The MulSupCon runs are reused for all three
   encoders, because MulSupCon did not change.
2. `coco_mlt_rtn_summary_round8.ipynb`. It now also shows a table of every
   skeleton version so far.
3. `coco_mlt_rtn_diagnostics_round8.ipynb`. Its Step 5 shows whether the
   push-down of positives is gone, next to version 1.

Then save the five notebooks (Ctrl+S) and send them back. Running the
notebooks again replaces the outputs shown inside them. If you want to keep
version 1's outputs in notebook form, copy the executed notebooks first.
Version 1's saved results (CSVs, figures, `result.json`) are kept anyway.

**Where version 2's results go.** Each version gets its own subfolder, so
nothing is overwritten:

```
coco_mlt/rtn_skeleton/clip/            version 1's results (as before), runs/ (all versions)
coco_mlt/rtn_skeleton/clip/v2/         version 2's results
coco_mlt/rtn_skeleton/summary/v2/      version 2's summary (+ all_versions.csv)
coco_mlt/rtn_skeleton/diagnostics/v2/  version 2's diagnostics
```

The same applies to `vit_b16/` and `resnet50/`.

**New self-tests.** Step 1 now checks four things:

- RTN's fast loss equals its row-by-row definition, both the values and the gradients.
- With alpha 0, RTN is exactly MulSupCon.
- Without the anchor term, RTN is exactly version 1.
- The host really trains the RTN runs with this loss.

Step 3 checks the loss on your real training labels. The host files are
unchanged: RTN plugs into them at run time (`plug_into_host`), in the
notebook and in any parallel worker processes. The fix for the ViT feature
extraction error on Windows (UTF-8) is included.

---

## How version 1 was set up

The first version of this README replaced every round 8 file sent before it
(`rtn.py`, `coco_mlt_rtn_skeleton_round8.ipynb`,
`coco_mlt_rtn_strict_skeleton_round8.ipynb`; delete those if you still have
them).

## Files

| File | What it is |
|---|---|
| `rtn_skeleton.py` | **The skeleton**: the RTN components, MulSupCon as host and baseline, and the fixed protocol (split, seeds, training, measurements, decision rule). |
| `coco_mlt_rtn_clip_round8.ipynb` | COCO-MLT with CLIP ViT-B/16 image features. |
| `coco_mlt_rtn_vit_b16_round8.ipynb` | COCO-MLT with ViT-B/16 trained on ImageNet: same architecture and input as CLIP, different pretraining. |
| `coco_mlt_rtn_resnet50_round8.ipynb` | COCO-MLT with ResNet-50 trained on ImageNet (input 448): a different architecture. |
| `coco_mlt_rtn_summary_round8.ipynb` | One table and one figure across the three encoders, plus every skeleton version so far. It trains nothing. |
| `rtn_diagnostics.py`, `coco_mlt_rtn_diagnostics_round8.ipynb` | Where AP is lost and what RTN changes (current version vs the previous one). They train nothing. |

The three encoder notebooks are identical apart from the encoder. They only
say which dataset and which encoder to use, then call `rtn_skeleton.py`.

## Changing the skeleton

Change components only in `rtn_skeleton.py`, and raise `SKELETON_VERSION` by
one. Then rerun the encoder notebooks and the summary.

- Run names contain the skeleton version, so no run from an older skeleton is
  ever reused.
- Every result carries a fingerprint of `rtn_skeleton.py`. The summary refuses
  to compare results from different skeletons.

## The skeleton (version 2)

- **Relation source:** GloVe word vectors of each class name plus its WordNet definition. No CLIP is used, either in the method or in the measurement.
- **Relation types:** the top 5% most similar names count as related. R is 1 for related and seen together, 0.5 for related but never together, 0.25 for an informative companion, and 0 otherwise.
- **Where RTN acts:** the negatives of MulSupCon's loss, with weight w = 1 + R, where R counts only the negative's labels that the anchor does not have (version 1 counted all of them). Nothing is tuned.
- **Host and baseline:** MulSupCon's official pipeline (round 6 code).
- **Protocol:** an 80 / 20 training / dev split, seeds 42, 43 and 44, and the decision rule from rounds 1-2.

## Where to put the files

The notebooks can sit in a new folder inside your project folder:

```
MSc-Project---RTN/                  your project folder
  coco_mlt/                         data, as now (results go to coco_mlt/rtn_skeleton/)
  context_reliance_check.py         round 6, stays here
  relation_contrastive_method.py    round 6, stays here
  relation_contrastive_runs.py      round 6, stays here
  coco_mlt_encoder_features.py      round 7, stays here
  rtn_skeleton.py                   the skeleton: one copy, here or in the new folder
  <new folder>/                     the four round 8 notebooks (and this README)
```

- **Where `rtn_skeleton.py` goes:** put it in the project folder itself, so
  notebooks for later datasets in other folders use the same copy. Each
  notebook finds the project folder, and stops if it sees two different
  copies of `rtn_skeleton.py`.
- **Naming the new folder:** any name works except `rtn_skeleton`.
- **Earlier runs:** keep round 6's saved runs (`coco_mlt/contrastive/runs/`)
  and round 7's (`coco_mlt/round7/relation_contrastive_resnet50/runs/`) where
  they are. Their MulSupCon runs are reused after a check that they come from
  the same split.

## Run order (version 1, for the record)

For version 2, see the top of this file. Version 1 ran the encoder
notebooks one at a time, in any order (Run All in each):

| Notebook | Trained | Time with a GPU |
|---|---|---|
| CLIP | 3 RTN runs | about 1.5 h |
| ResNet-50 | 3 RTN runs | about 1.5 h |
| ViT-B/16 (ImageNet) | extracted its features first, then 3 MulSupCon + 3 RTN runs | about 3 h |

Every notebook is resumable: if one stops, run it again and finished runs
are kept.

## Where the results go

```
coco_mlt/
  relations/                     GloVe file, WordNet mapping, class embeddings
  rtn_skeleton/clip/             runs/ (every version), version 1's tables, figure, result.json
  rtn_skeleton/clip/v2/          version 2's tables, figure.png, result.json, summary.txt
  rtn_skeleton/vit_b16/          the same for ViT-B/16 (ImageNet)
  rtn_skeleton/resnet50/         the same for ResNet-50
  rtn_skeleton/summary/          version 1's summary; summary/v2/ for version 2
  rtn_skeleton/diagnostics/      version 1's diagnostics; diagnostics/v2/ for version 2
```

Later datasets get their own notebooks calling the same `rtn_skeleton.py`,
for example `voc_mlt/rtn_skeleton/<encoder>/`.
