# Round 8: the RTN skeleton as one module, tested on three encoders (COCO-MLT)

This replaces every round 8 file sent earlier today. Delete these and don't
run them:

- `rtn.py`
- `coco_mlt_rtn_skeleton_round8.ipynb`
- `coco_mlt_rtn_strict_skeleton_round8.ipynb`

## Files

| File | What it is |
|---|---|
| `rtn_skeleton.py` | **The skeleton**: the RTN components, MulSupCon as host and baseline, and the fixed protocol (split, seeds, training, measurements, decision rule). |
| `coco_mlt_rtn_clip_round8.ipynb` | COCO-MLT with CLIP ViT-B/16 image features. |
| `coco_mlt_rtn_vit_b16_round8.ipynb` | COCO-MLT with ViT-B/16 trained on ImageNet: same architecture and input as CLIP, different pretraining. |
| `coco_mlt_rtn_resnet50_round8.ipynb` | COCO-MLT with ResNet-50 trained on ImageNet (input 448): a different architecture. |
| `coco_mlt_rtn_summary_round8.ipynb` | One table and one figure across the three encoders. It trains nothing. |

The three encoder notebooks are identical apart from the encoder. They only
say which dataset and which encoder to use, then call `rtn_skeleton.py`.

## Changing the skeleton

Change components only in `rtn_skeleton.py`, and raise `SKELETON_VERSION` by
one. Then rerun the encoder notebooks and the summary.

- Run names contain the skeleton version, so no run from an older skeleton is
  ever reused.
- Every result carries a fingerprint of `rtn_skeleton.py`. The summary refuses
  to compare results from different skeletons.

## The skeleton (version 1)

- **Relation source:** GloVe word vectors of each class name plus its WordNet definition. No CLIP is used, either in the method or in the measurement.
- **Relation types:** the top 5% most similar names count as related. R is 1 for related and seen together, 0.5 for related but never together, 0.25 for an informative companion, and 0 otherwise.
- **Where RTN acts:** the negatives of MulSupCon's loss, with weight w = 1 + R. Nothing is tuned.
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

## Run order

Run the encoder notebooks one at a time, in any order (Run All in each):

| Notebook | Trains | Time with a GPU |
|---|---|---|
| CLIP | 3 RTN runs | about 1.5 h |
| ResNet-50 | 3 RTN runs | about 1.5 h |
| ViT-B/16 (ImageNet) | extracts its features first (a few minutes), then 3 MulSupCon + 3 RTN runs | about 3 h |

The first notebook you run downloads GloVe (about 390 MB, once) and a 10 KB
WordNet mapping. Every notebook is resumable: if one stops, run it again and
finished runs are kept.

When all three are done, run `coco_mlt_rtn_summary_round8.ipynb`. Then save
the four notebooks (Ctrl+S) and send them back.

## Where the results go

```
coco_mlt/
  relations/                     GloVe file, WordNet mapping, class embeddings
  rtn_skeleton/clip/             runs, tables, figure.png, result.json, summary.txt
  rtn_skeleton/vit_b16/          the same for ViT-B/16 (ImageNet)
  rtn_skeleton/resnet50/         the same for ResNet-50
  rtn_skeleton/summary/          the table and figure across encoders
```

Later datasets get their own notebooks calling the same `rtn_skeleton.py`,
for example `voc_mlt/rtn_skeleton/<encoder>/`.
