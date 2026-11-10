# Algae Detection Toolkit

A reproducible toolkit for training, evaluating and applying deep-learning detectors on
high-throughput microalgae microscopy images. It covers the full workflow: dataset
preparation, detector training, adaptive tiled inference, detection and counting metrics,
robustness tests and multi-run aggregation.

The toolkit is built around the six-class VisAlgae 2023 dataset and supports:

- D-FINE-S and RT-DETRv2-R50 through Hugging Face Transformers;
- YOLO11n / YOLO11s / YOLOv8s baselines through Ultralytics;
- whole-image and adaptive tiled inference;
- COCO detection metrics, per-image counting, Bland-Altman limits of agreement, ICC, CCC,
  Bray-Curtis dissimilarity and Shannon diversity;
- eight synthetic image degradations at three or four severity levels;
- tile-size sweeps and multi-GPU job scheduling.

## Repository layout

```
Algae_detection/
├── src/
│   ├── data/
│   │   ├── dataset.py          # dataset, letterboxing, augmentation, RFS sampler
│   │   └── prepare.py          # YOLO labels -> COCO annotations and splits
│   ├── models/build.py         # D-FINE / RT-DETR construction, FiLM, EMA
│   ├── losses/nwd.py           # normalised Wasserstein distance regression
│   ├── train.py                # D-FINE / RT-DETR training
│   ├── evaluate.py             # COCO AP, counting and community metrics
│   ├── infer.py                # inference on labelled or unlabelled images
│   ├── train_yolo.py           # Ultralytics YOLO baseline
│   ├── eval_yolo_tiled.py      # tiled inference for YOLO checkpoints
│   ├── robustness.py           # degradation benchmark
│   ├── tile_sweep.py           # tile-size sweep
│   ├── counts_csv.py           # per-image count table
│   ├── aggregate.py            # collect run metrics into CSV files
│   └── stats.py                # paired bootstrap and Wilcoxon tests
├── tools/
│   ├── eval_suite.sh           # full + tiled + counting + robustness for one run
│   ├── run_queue.py            # JSONL job queue across GPUs
│   └── diag_predictions.py     # draw ground truth and predictions
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
cd Algae_detection
conda create -n algae python=3.10 -y
conda activate algae

# Install the CUDA build of PyTorch that matches your driver first, for example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

`transformers>=4.53` is required for the D-FINE and RT-DETRv2 implementations. A single
CUDA GPU is sufficient for training and evaluation; the default D-FINE-S recipe uses about
10.2 M parameters.

## Data

The VisAlgae 2023 images are not distributed with this repository. Download them from the
official dataset repository:

<https://github.com/juntaoJianggavin/Visalgae2023>

Place the files under `algae/` in the project root. The loader expects the following
layout (note the directory name `lables`):

```
algae/
├── images/
│   ├── train/          # labelled images
│   ├── val/            # labelled images; merged with train and re-split
│   └── test/           # optional unlabelled images for inference demos
└── lables/
    ├── train/*.txt     # YOLO labels: class cx cy w h, normalised to [0, 1]
    └── val/*.txt
```

If the downloaded release already provides a single labelled folder, put all labelled
images in `images/train/` and all labels in `lables/train/`. The preparation script merges
the `train` and `val` folders before splitting, so the original folder assignment does not
affect the result.

Build the COCO annotations, the YOLO-format dataset and the data splits:

```bash
python -m src.data.prepare --root . --src algae --seed 42 --val-frac 0.2
```

The script produces:

```
data/coco/annotations/instances_train.json     # 560 images
data/coco/annotations/instances_val.json       # 140 images
data/coco/annotations/instances_test_unlabeled.json
data/yolo/                                     # YOLO-format images, labels and data.yaml
data/splits/main.json                          # image ids for each split
```

The split is stratified by acquisition group and class composition. Additional 5-fold and
cross-domain annotations are written for users who need them; the commands below only use
the main split.

## Training

Train D-FINE-S at 1280 px with the default recipe:

```bash
python -m src.train \
    --run-id dfine_s_1280 \
    --arch dfine \
    --size 1280 \
    --epochs 100 \
    --batch-size 8 \
    --seed 0
```

The pretrained `ustc-community/dfine-small-obj365` weights are downloaded from Hugging
Face on first use. If the weights are already cached locally, point the downloader at the
cache instead of the network:

```bash
export HF_HUB_CACHE=/path/to/huggingface/hub
python -m src.train --run-id dfine_s_1280 --size 1280
```

You can also bypass the model id entirely with a local snapshot directory:

```bash
python -m src.train --run-id dfine_s_1280 --ckpt /path/to/dfine-small-obj365 --size 1280
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--arch rtdetr` | Train RT-DETRv2-R50 instead of D-FINE-S |
| `--size 640/960/1280/1536` | Square network input |
| `--epochs`, `--batch-size`, `--lr` | Training schedule |
| `--patience 20` | Early stopping on validation mAP |
| `--no-tile-train` | Disable the random 2:1 training crop |
| `--tile-prob`, `--scale-crop-prob` | Training-time tiling and scale-crop probabilities |
| `--film`, `--nwd`, `--rfs`, `--copy-paste 0.5` | Optional modules and recipes |
| `--limit-train 280` | Train on a subset for learning-curve experiments |
| `--ann-test path.json` | Separate labelled set for the final evaluation |
| `--seed` | Repeat the run with a different seed |

Each run is written to `runs/<run-id>/`:

```
best.pt          best validation checkpoint
last.pt          last epoch, optimizer and scheduler state
metrics.json     final full-image and tiled metrics, parameter count and FPS
train_log.csv    per-epoch loss and validation metrics
config.json      full command-line configuration
```

By default the validation split is used both for checkpoint selection and for reporting.
For an unbiased final number, keep a separate labelled test set and pass it with
`--ann-test`; the validation split is still used for model selection.

## Evaluation

Whole-image evaluation:

```bash
python -m src.evaluate \
    --ckpt runs/dfine_s_1280/best.pt \
    --ann data/coco/annotations/instances_val.json \
    --out results/dfine_s_1280.json
```

Adaptive tiled inference:

```bash
python -m src.evaluate \
    --ckpt runs/dfine_s_1280/best.pt \
    --ann data/coco/annotations/instances_val.json \
    --tiling \
    --out results/dfine_s_1280_tile.json
```

The tiling policy is image-dependent:

- aspect ratio > 2.2: two 2:1 tiles, with the tile height set by `--tile-size` or the image
  height (maximum 800 px);
- area > 1.5 MP: a grid of overlapping 800 x 800 px tiles;
- otherwise: whole-image inference.

Tiles overlap by 25% by default and are merged with class-wise NMS at IoU 0.6. Use
`--tile-size` to override the tile short side, `--overlap` for the overlap and
`--nms-iou` for merging.

The output JSON contains COCO mAP, mAP50, mAP75, AP_S/AP_M/AP_L, per-class and per-group
AP, counting metrics (MAE, RMSE, MAPE, bias, ICC, CCC), community metrics (Bray-Curtis
dissimilarity and Shannon diversity difference) and inference timing. A
`<name>_preds.json` file with all detections is written next to it.

To evaluate a different checkpoint size, pass `--size`; otherwise the value stored in the
checkpoint is used.

## Inference and counting

Run inference on unlabelled images and draw the predictions:

```bash
python -m src.infer \
    --ckpt runs/dfine_s_1280/best.pt \
    --ann data/coco/annotations/instances_test_unlabeled.json \
    --tiling \
    --score-thr 0.3 \
    --out results/test_detections.json \
    --viz-dir results/visualisations
```

`--viz-dir` writes JPEG files with ground-truth boxes in green and predictions in species
colours. Omit it to save only the detection JSON.

Build a per-image count table from saved detections:

```bash
python -m src.counts_csv \
    --preds results/dfine_s_1280_preds.json \
    --ann data/coco/annotations/instances_val.json \
    --thr 0.3 \
    --out results/counts.csv
```

The CSV has one row per image with ground-truth and predicted counts, both in total and
per class. It can be used directly for Bland-Altman plots or calibration curves.

## Robustness and tiling sweeps

Evaluate all eight degradation types at up to four severity levels:

```bash
python -m src.robustness \
    --ckpt runs/dfine_s_1280/best.pt \
    --name dfine_s_1280 \
    --tiling \
    --out results/robustness/dfine_s_1280.json
```

The degradations are Gaussian noise, Gaussian blur, motion blur, brightness, contrast,
JPEG compression, colour temperature and random occlusion. Use `--levels` to run a subset,
for example `--levels gaussian_noise,occlusion`.

Measure mAP for several tile sizes on wide-field images:

```bash
python -m src.tile_sweep \
    --ckpt runs/dfine_s_1280/best.pt \
    --tile-sizes 608,768,960,1216,1536 \
    --out results/tile_sweep.json
```

## YOLO baselines

Train a YOLO baseline with augmentations matched to the D-FINE recipe:

```bash
python -m src.train_yolo \
    --model yolo11s.pt \
    --run-id yolo11s_1280 \
    --size 1280 \
    --batch 8 \
    --seed 0
```

The run is written to `runs_yolo/yolo11s_1280/`, including `weights/best.pt` and
`metrics.json`. Ultralytics downloads the pretrained checkpoint automatically.

Evaluate the YOLO checkpoint with the shared evaluator, using whole-image or tiled
inference:

```bash
python -m src.eval_yolo_tiled \
    --weights runs_yolo/yolo11s_1280/weights/best.pt \
    --imgsz 1280 \
    --no-tiling \
    --out results/yolo11s_1280_full.json

python -m src.eval_yolo_tiled \
    --weights runs_yolo/yolo11s_1280/weights/best.pt \
    --imgsz 1280 \
    --out results/yolo11s_1280_tile.json
```

## Batch evaluation and aggregation

Run the standard evaluation chain for one trained model:

```bash
bash tools/eval_suite.sh dfine_s_1280
```

This writes full-image and tiled metrics, a counting CSV and a robustness JSON under
`results/`. If the environment is not activated, set the interpreter explicitly:

```bash
PY=/path/to/python bash tools/eval_suite.sh dfine_s_1280
```

Collect all `metrics.json` files into tidy CSVs:

```bash
python -m src.aggregate --runs runs --runs-yolo runs_yolo --out results
```

This writes `results/all_runs.csv` plus the summary tables that apply to the runs found
(for example `benchmark.csv`, `modules.csv` and the cross-domain summary). Run paired
bootstrap and Wilcoxon tests on the aggregated runs:

```bash
python -m src.stats --runs results/all_runs.csv --out results/stats.json
```

## Utility scripts

Draw ground truth and decoded predictions for a few images:

```bash
python tools/diag_predictions.py \
    --ckpt runs/dfine_s_1280/best.pt \
    --ann data/coco/annotations/instances_val.json \
    --size 1280 \
    --out-dir results/diagnostics
```

Run a JSONL job manifest across several GPUs:

```bash
python tools/run_queue.py --manifest jobs.jsonl --gpus 0,1,2,3
```

Each manifest line has the form
`{"id": "dfine_s_1280_s0", "cmd": ["python", "-m", "src.train", "..."], "env": {}}`.
The queue writes logs to `logs/<id>.log` and status records to
`logs/queue_status.jsonl`.

## Class order

Class ids are fixed across the code and the annotation files:

| id | species |
| --- | --- |
| 0 | Platymonas |
| 1 | Chlorella |
| 2 | Dunaliella salina |
| 3 | Effrenium |
| 4 | Porphyridium |
| 5 | Haematococcus |

If you adapt the toolkit to another dataset, keep this order or update `CLASS_NAMES` in
`src/data/dataset.py` and the class-dependent summary code in `src/evaluate.py`.

## Notes

- All commands are run from the repository root.
- Checkpoints and dataset files are excluded by `.gitignore`; only code and configuration
  are versioned.
- The default configuration is tuned for six-species VisAlgae 2023 images. Other
  magnifications, staining protocols or camera settings may need a different input size,
  tiling policy and colour augmentation.
- For reproducible comparisons, keep the split seed, input size and evaluation threshold
  fixed and report the checkpoint-selection rule.

## Dataset attribution

This toolkit uses the VisAlgae 2023 dataset. If you use the data, cite:

> Sun M, Jiang J, Yang Z, et al. VisAlgae 2023: a dataset and challenge for algae detection
> in microscopy images. arXiv:2505.20687, 2025.

The dataset is distributed by its original authors under their own terms; it is not
included in this repository.
