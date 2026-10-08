# Skin Lesion Classification with Lightweight CNN

This research project classifies HAM10000 dermoscopy images with two PyTorch models: an ImageNet pretrained MobileNetV3-Small baseline and a MobileNetV3-Small feature extractor followed by frequency-domain attention. It is designed to run on CPU, with optional CUDA if available. The code is provided without trained weights or claimed results.

**Research and educational use only. It is not a medical diagnosis tool.**

## Dataset and classes

HAM10000 metadata links each `image_id` to a diagnosis label in `dx`. The seven labels are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. The scripts use the metadata IDs to find JPG images in either image folder. Place your existing dataset here:

```text
data/HAM10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/*.jpg
  HAM10000_images_part_2/*.jpg
```

The screenshot shows these three dataset items inside `8th_sem_project`. Copy or move only those items into the paths above. The `hmnist_*.csv` files are not used. Dataset files are intentionally excluded from Git.

## How the models work

MobileNetV3-Small is a compact convolutional network. Transfer learning initializes its feature extractor from ImageNet weights, while the final classifier is adapted to seven classes. The proposed model computes a 2D FFT on the extracted feature map, summarizes frequency magnitudes into channel weights, and uses those weights to enhance the original features before global average pooling and classification. The FFT and attention operations remain differentiable.

## Project layout

```text
data/HAM10000/             Dataset location
src/config.py              Paths and training defaults
src/dataset.py             Image loading and transforms
src/data_analysis.py       Metadata summary and sample plots
src/models/                Baseline, attention, proposed model
src/metrics.py             Overall and per-class metrics
src/visualization.py       Reusable plotting functions
src/utils.py               Seeds, splits, checkpoints, helpers
src/train.py               Training and checkpointing
src/evaluate.py            Held-out test evaluation
src/compare_models.py      Comparison from saved evaluations
src/predict.py             Single-image inference
results/                   Generated checkpoints, tables, plots
requirements.txt           Python dependencies
```

## Installation

From the project root, use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The first training run requests ImageNet pretrained weights through torchvision, which may download them if they are not already cached. Evaluation and prediction use saved local checkpoints and do not request pretrained weights.

## Commands

Run all commands from the project root. Module form keeps imports consistent.

```bash
python -m src.data_analysis
python -m src.train --model baseline
python -m src.train --model frequency_attention
python -m src.train --model baseline --fast-debug
python -m src.evaluate --model baseline
python -m src.evaluate --model frequency_attention
python -m src.compare_models
python -m src.predict --image path/to/image.jpg --model frequency_attention
```

The equivalent direct entry points `python src/train.py`, `python src/evaluate.py`, and `python src/predict.py` also work when launched from this project root. Training uses a deterministic stratified 70/15/15 split. Class weights come only from the training split. Both models use the same test split, so comparison reads real metrics produced by evaluation. `FAST_DEBUG` in `src/config.py` or `--fast-debug` limits training to a small subset and two epochs; those checkpoints are only for pipeline checks.
