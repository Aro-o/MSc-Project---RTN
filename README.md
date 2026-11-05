# Companions Matter: Not All Co-occurring Labels Are Equal

**Relation-Typed Negatives (RTN) for long-tailed multi-label contrastive learning**

MSc Artificial Intelligence and Robotics with Advanced Research, University of Hertfordshire\
Author: Aroosha Rasheed · Supervisor: Dr Guangming Huang

## Overview

In long-tailed multi-label classification, many errors on rare (tail) classes are false positives on
images of *related* classes: a photo of a snowboarder is ranked above real photos of skis. RTN gives
every pair of classes one of four relation types, computed from CLIP text similarity and training
co-occurrence (related and seen together, related but never seen together, informative companion,
none), and pushes negatives of the confusable types away harder: as weights in a multi-label
contrastive loss (MulSupCon), as margins or a class-anchored contrast in a classifier, or by
re-scoring predictions at test time.

All experiments use **COCO-MLT** (Wu et al., 2020): 1,909 training images and 80 classes (22 head,
33 medium and 25 tail), evaluated on the 5,000 COCO val2017 images, with frozen CLIP ViT-B/16
features.

| Notebook | What it does |
|---|---|
| `coco_mlt_context_check.ipynb` | Downloads the COCO-MLT split (checked against pinned SHA-256 hashes), the COCO 2017 annotations and the 6,909 images; builds the labels; extracts frozen CLIP ViT-B/16 features; trains linear classifiers with BCE, ASL and DB-Focal (3 seeds); runs the context-reliance check (diagnostic study) |
| `coco_mlt_followup.ipynb` | Breaks the related-class false positives down by relation type, from the first notebook's saved outputs (no training) |
| `coco_mlt_relation_contrastive_fast.ipynb` | Round 1: faithful MulSupCon reproduction; RTN against SCHaNe, LCL, all-companion weighting and BCE (3 seeds) |
| `coco_mlt_relation_contrastive_round2.ipynb` | Round 2: stronger relation weights, RTN combined with SCHaNe, breakdown by relation type |
| `coco_mlt_rtn_clip_round3.ipynb` | Round 3: CLIP-initialised classifier (TA) with zero-shot blend; RTN on image embeddings; zero-shot CLIP and linear-probe baselines (5-fold cross-validation, 10 seeds) |
| `coco_mlt_rtn_clip_round4.ipynb` | Round 4: relation-typed margins and class contrast in the classifier |
| `coco_mlt_rtn_clip_round5.ipynb` | Round 5: headroom analysis, class-anchored contrast and relation-typed re-scoring at test time |
| `coco_mlt_relation_contrastive_round6.ipynb` | Round 6: ALL and ANY positives, the Similarity-Dissimilarity loss (Huang et al., 2026) and its combination with RTN, dataset analysis, GPU and CPU timing |

All notebooks except round 6 are saved with their outputs; round 6 is being run. Round 6 uses our
own implementation of the Similarity-Dissimilarity loss, written from the paper (the similarity and
dissimilarity factors weight each positive's log-probability term), not the authors' code.

## Code

| File | Contents |
|---|---|
| `context_reliance_check.py` | Shared helpers: average precision, beat rates, CLIP text similarity and related classes. Every notebook imports it, so it must be in the same folder |
| `relation_contrastive_method.py` | MulSupCon pipeline: data split, models, the MulSupCon loss and its variants (RTN, SCHaNe, LCL, ALL, ANY, Similarity-Dissimilarity), pretraining and fine-tuning |
| `relation_contrastive_runs.py` | Running the MulSupCon pipeline: data preparation, single runs and parallel worker processes |
| `rtn_clip_method.py` | CLIP pipeline: TA classifier, RTN image and class contrast, margins, class-anchored contrast, re-scoring, headroom, cross-validation folds |

The notebooks write the last three files themselves when they run (`%%writefile`), so each notebook
contains the exact version it used. The copies here are the latest versions (round 6 for the
MulSupCon pipeline, round 5 for the CLIP pipeline).

## Setup

Tested on Windows with Python 3.14, PyTorch 2.11 (CUDA 13.0), NumPy 2.5 and pandas 3.1.

1. Install PyTorch with the CUDA build for your GPU (<https://pytorch.org/get-started/locally/>),
   then the other packages, including OpenAI's CLIP package from GitHub:

   ```
   pip install -r requirements.txt
   ```

2. Open the notebooks from this folder. They create `coco_mlt/` (data, features, saved runs and
   results) and `model_cache/` (the CLIP ViT-B/16 weights).
3. A CUDA GPU is recommended; the experiments were run on an NVIDIA GeForce RTX 4070 Laptop GPU
   (8 GB). The training notebooks stop if no GPU is visible unless `ALLOW_CPU = True` is set in their
   settings cell, and `QUICK_TEST = True` runs a tiny version to check that everything works.

## Data

No data is stored in this repository. `coco_mlt_context_check.ipynb` downloads about 1.5 GB:

- the COCO-MLT split files from the authors' repository (Wu et al., 2020);
- the COCO 2017 annotations (Creative Commons Attribution 4.0, COCO Consortium) and images (Flickr
  terms of use) (Lin et al., 2014).

## Reproducibility

- Fixed seeds: 42–44 in the MulSupCon pipeline, 42–51 in the CLIP pipeline.
- Settings are chosen on training images only (a held-out development split, or 5-fold
  cross-validation); the test images are used only for the final evaluation.
- Comparisons use paired, class-level 95% bootstrap intervals and decision rules fixed before each
  round.
- Every notebook writes its results tables and a text summary to `coco_mlt/`.

## Main results so far (COCO-MLT test set)

- **Diagnostic:** images of classes that are related to a tail class and co-occur with it in training
  outrank its true positives 7.6 AUC points more often than images of unrelated classes, after
  controlling for busy images. ASL and DB-Focal leave this gap essentially unchanged.
- **Rounds 1–2:** RTN was the only negative weighting that significantly reduced this confusion
  (−0.38 AUC points, 95% CI [−0.64, −0.13]); hardness-based (SCHaNe) and learned-confusion (LCL)
  weighting increased it. Tail mAP did not change significantly.
- **Round 5:** removing the confusion would add up to 9.3 tail-mAP points. Relation-typed re-scoring
  raised tail mAP by 0.49 [0.20, 0.87] and overall mAP by 0.29 [0.07, 0.50] over the same model
  without relation types, with head AP unchanged. Best system: 72.87 overall and 78.22 tail mAP.

## References

- Huang, G., Long, Y. and Luo, C. (2026) 'Similarity-dissimilarity loss for multi-label supervised
  contrastive learning', *Transactions on Machine Learning Research*.
  <https://openreview.net/forum?id=W445zcqThv>
- Lin, T.-Y. et al. (2014) 'Microsoft COCO: common objects in context', in *Computer Vision – ECCV
  2014*. Cham: Springer, pp. 740–755. doi:10.1007/978-3-319-10602-1_48
- Long, Z., Killick, G., McCreadie, R., Aragon Camarasa, G. and Meng, Z. (2023) 'When hard negative
  sampling meets supervised contrastive learning', arXiv:2308.14893 (SCHaNe).
- Radford, A. et al. (2021) 'Learning transferable visual models from natural language supervision',
  in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139,
  pp. 8748–8763 (CLIP).
- Ridnik, T. et al. (2021) 'Asymmetric loss for multi-label classification', in *Proceedings of the
  IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 82–91 (ASL).
- Suresh, V. and Ong, D.C. (2021) 'Not all negatives are equal: label-aware contrastive loss for
  fine-grained text classification', in *Proceedings of EMNLP 2021*, pp. 4381–4394.
  doi:10.18653/v1/2021.emnlp-main.359 (LCL).
- Wu, T., Huang, Q., Liu, Z., Wang, Y. and Lin, D. (2020) 'Distribution-balanced loss for multi-label
  classification in long-tailed datasets', in *Computer Vision – ECCV 2020*. Cham: Springer,
  pp. 162–178. doi:10.1007/978-3-030-58548-8_10 (COCO-MLT, DB-Focal).
- Zhang, P. and Wu, M. (2024) 'Multi-label supervised contrastive learning', *Proceedings of the AAAI
  Conference on Artificial Intelligence*, 38(15), pp. 16786–16793. doi:10.1609/aaai.v38i15.29619
  (MulSupCon).
