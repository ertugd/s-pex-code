# S-PEX: Space-to-Depth and P2-Head Ablations for Aircraft Surface Damage Detection with YOLO26

## Description

This repository contains the code used to study whether two architectural
modifications improve small- and medium-object detection of aircraft surface
damage (cracks and dents) over a plain YOLO26-nano baseline:

1. **SPD-Conv** — every stride-2 downsampling convolution in the backbone and
   in the neck's bottom-up path is replaced by a *space-to-depth* convolution
   (Sunkara & Purushotham, 2022). A 2x2 sub-sampling moves spatial information
   into the channel dimension before a stride-1 convolution, so no pixel is
   discarded while downsampling.
2. **P2 head** — an additional stride-4 detection branch is added, extending
   `Detect` from three levels `[P3/8, P4/16, P5/32]` to four
   `[P2/4, P3/8, P4/16, P5/32]`.
3. **SPD-Conv + P2** — both combined.

Each modification is implemented as a single-variable ablation against the same
baseline: the backbone depth/width multipliers, training schedule and
hyper-parameters are held fixed, and only the architecture changes.

For context, the repository also contains baseline training notebooks for
several other YOLO generations (v5, v8, v9, v11, v12) at the nano scale,
trained on the same data with the same schedule.

---

## Dataset Information

| Property | Value |
|---|---|
| Source | Roboflow — `ke-project/yolo-tb1l6`, version 1 |
| URL | https://universe.roboflow.com/ke-project/yolo-tb1l6/dataset/1 |
| Task | Object detection (YOLO format) |
| Classes | 2 — `crack`, `dent` |
| License | CC BY 4.0 |

**Splits** (as released by Roboflow; used unmodified):

| Split | Images | Annotated boxes |
|---|---|---|
| train | 2,747 | 8,688 |
| valid | 568 | 1,907 |
| test | 279 | 1,023 |
| **total** | **3,594** | **11,618** |

**Object-size profile of the training split** (box side computed as
`sqrt(w*h)` at 640x640 input):

| Bucket | Share |
|---|---|
| small (< 32 px) | 30.5 % |
| medium (32-96 px) | 15.6 % |
| large (> 96 px) | 53.9 % |

Median box side is 122 px and there are 3.16 boxes per image. The 30.5 %
small-object share is the reason the P2 branch is worth testing on this
dataset — see *Methodology*.

> **Note.** A small number of images carry mixed segmentation/detection label
> rows. Ultralytics warns about this and keeps the boxes only; the behaviour is
> identical across all runs, so it does not affect the comparison.

---

## Code Information

```
s-pex-code/
├── README.md
├── requirements.txt
│
├── baseline/            YOLO26n baseline (reference for every ablation)
│   └── yolo26.ipynb
│
├── p2/                  Ablation 1 — P2 head only
│   ├── p2.yaml              4-level model definition (plain Conv)
│   └── p2-main.ipynb
│
├── SPD-Conv/            Ablation 2 — SPD-Conv only
│   ├── spd-covn.yaml        3-level model definition, SPDConv downsampling
│   ├── spdconv.py           SPDConv module + parser registration
│   └── spd-conv-main.ipynb
│
├── SPD-Conv-P2/         Ablation 3 — SPD-Conv + P2 combined
│   ├── spd-covn-p2.yaml
│   ├── spdconv.py
│   └── spd-conv-main-p2.ipynb
│
└── yolo5/ yolo8/ yolo9/ yolo11/ yolo12/    Cross-generation baselines
    └── yolo<N>-baseline.ipynb
```

### `spdconv.py`

Defines `SPDConv`, a drop-in replacement for Ultralytics' `Conv` with an
identical constructor signature `(c1, c2, k, s, p, g, d, act)`, so the YAML
parser computes channel counts correctly and `[out_channels, kernel, stride]`
arguments stay unchanged.

* When `s == 2`: space-to-depth (H, W halved, channels x4) followed by a
  stride-1 convolution. When `s != 2` it behaves as an ordinary Conv-BN-SiLU,
  so every downsampling layer can be swapped safely.
* Subclasses Ultralytics' `Conv` so that `model.fuse()` folds BatchNorm into
  the convolution. A plain `nn.Module` is silently skipped by `fuse()`.
* `register_spdconv()` binds the name `SPDConv` used in the YAML files to this
  implementation. **It must be called before `YOLO(...)`.**

### Model sizes

| Configuration | Parameters | Detection levels |
|---|---|---|
| YOLO26n baseline | 2.50 M | 3 |
| + P2 | 2.52 M | 4 |
| + SPD-Conv | 4.51 M | 3 |
| + SPD-Conv + P2 | 4.55 M | 4 |

The P2 branch is nearly free in parameters but multiplies the activation area
at stride 4, so its cost is memory and time rather than weights.

---

## Requirements

* Python 3.12 (developed on 3.12.9)
* NVIDIA GPU with CUDA — the reference runs used an RTX 4060 (8 GB)
* Packages: see `requirements.txt`

```bash
pip install -r requirements.txt
```

Reference environment:

| Package | Version |
|---|---|
| ultralytics | 8.4.14 |
| torch | 2.7.1+cu118 |

> Ultralytics changes training defaults between releases. Pin the version above
> if you intend to compare your numbers against the ones reported here.

---

## Usage Instructions

### 1. Get the dataset

Download the dataset in **YOLO format** from the Roboflow link above and place
it so that `data.yaml` sits at the repository root under `dataset/`:

```
s-pex-code/
└── dataset/
    ├── data.yaml
    ├── train/{images,labels}
    ├── valid/{images,labels}
    └── test/{images,labels}
```

Every notebook resolves the dataset itself:

```python
_CANDIDATES = ["../dataset/data.yaml", "../../../dataset2/yolo26/YOLO.v1i.yolo26/data.yaml"]
DATA = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
```

The first entry is the layout above; the second is the author's local
development path. The setup cell prints the resolved absolute path and whether
it exists — check that line before training.

### 2. Run the notebooks in order

```
1. baseline/yolo26.ipynb          # train the reference model first
2. p2/p2-main.ipynb               # ablation 1
3. SPD-Conv/spd-conv-main.ipynb   # ablation 2
4. SPD-Conv-P2/spd-conv-main-p2.ipynb   # ablation 3
```

The baseline should be trained first: the ablations initialise from its
weights. If the baseline checkpoint is missing they still run — the notebooks
detect this, print a notice and start from COCO weights instead — but the
reported initialisation coverage will differ.

Each notebook has the same five cells:

| Cell | Purpose |
|---|---|
| 1 | imports and module registration |
| 2 | device selection + dataset path resolution |
| 3 | build the model, load pretrained weights, report transfer coverage |
| 4 | training (`epochs=500, imgsz=640, batch=16`) |
| 5 | read `results.csv`, report the best epoch, compare against the baseline |
| 6 | evaluate on the **test** split and print per-class metrics |

### 3. Memory

The reference runs use `batch=16` at `imgsz=640` on 8 GB. The two SPD-Conv
configurations are close to that limit because space-to-depth temporarily
quadruples channel counts; if you hit `CUDA out of memory`, set `batch=8`.

### 4. Cross-generation baselines

`yolo5/`, `yolo8/`, `yolo9/`, `yolo11/`, `yolo12/` are independent and can be
run in any order. They use the nano-scale checkpoint of each generation:

| Folder | Checkpoint | Parameters |
|---|---|---|
| yolo5 | `yolov5nu.pt` | 2.65 M |
| yolo8 | `yolov8n.pt` | 3.16 M |
| yolo9 | `yolov9t.pt` | 2.13 M |
| yolo11 | `yolo11n.pt` | 2.62 M |
| yolo12 | `yolo12n.pt` | 2.60 M |

Note the naming: v5 uses the anchor-free *u* variant, v9's nano-equivalent is
`t` (tiny), and v11/v12 have no `v` in the name. Weights download automatically
on first run.

---

## Methodology

### Experimental protocol

All runs share the same settings, so the architecture is the only variable:

| Setting | Value |
|---|---|
| epochs | 500 |
| image size | 640 |
| batch | 16 |
| patience | 100 (early stopping) |
| optimizer | see note below |

Model selection uses the best `mAP@0.5:0.95` epoch on the **validation** split;
the reported numbers are then computed once on the held-out **test** split.

> **Optimizer note.** The YOLO26 baseline is trained with `optimizer="auto"`,
> which Ultralytics resolves to MuSGD (lr0 = 0.01, momentum = 0.9). The three
> ablation notebooks pass `optimizer="SGD"` (momentum = 0.937) explicitly.
> This difference is inherited from the original experiment logs and is
> disclosed here rather than silently corrected; set the ablations to `"auto"`
> if you want the optimizer held fixed as well.

### Weight-transfer chain

`SPD-Conv-P2` is initialised from several single-component checkpoints in
increasing order of relevance, because `YOLO.load()` overwrites every matching
key and the last call wins. Measured coverage of the 902-tensor target:

| Source | Tensors matched | Parameters | Unique contribution |
|---|---|---|---|
| baseline | 355 / 902 | 22.9 % | shared skeleton |
| P2 | 894 / 902 | 40.7 % | 539 tensors: P2 branch, bottom-up path, 4-level `Detect` |
| SPD-Conv | 360 / 902 | 65.3 % | 5 tensors but 42.4 % of parameters: the heavy SPD backbone convolutions |
| **chain total** | **899 / 902** | **83.1 %** | |

The two sources are complementary. SPD-Conv fills few but very heavy tensors —
an SPD convolution weight is `c2 x (4*c1) x k x k` against `c2 x c1 x k x k` for
a plain one — while the P2 checkpoint fills the many light tensors of the
four-level head that neither other source contains. SPD-Conv is applied last
because the model's backbone *is* SPD-based, so its BatchNorm running
statistics match the activation distribution the network actually produces.

Single-component ablation checkpoints are used deliberately; the combined
model is never initialised from a previously trained combined model, which
would invalidate any claim that the combination improves on its parts.

### Results on the test split (279 images / 1,023 boxes)

| Configuration | Parameters | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| YOLO26n baseline | 2.50 M | 88.35 % | 63.51 % |
| + P2 | 2.52 M | **89.44 %** | 63.43 % |
| + SPD-Conv | 4.51 M | 88.28 % | 64.83 % |
| + SPD-Conv + P2 | 4.55 M | 89.30 % | **65.39 %** |

Validation-split figures at the selected epoch:

| Configuration | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| YOLO26n baseline | 0.8727 | 0.6301 |
| + P2 | 0.8809 | 0.6302 |
| + SPD-Conv | 0.8801 | 0.6434 |
| + SPD-Conv + P2 | **0.8898** | **0.6457** |

The two modifications act on different parts of the metric. P2 raises
`mAP@0.5` — it finds objects the three-level head misses — but leaves
`mAP@0.5:0.95` unchanged, i.e. it does not localise them more precisely.
SPD-Conv does the opposite: it improves `mAP@0.5:0.95` by preserving detail
through downsampling, without finding more objects. Combining them carries
both effects, giving the best `mAP@0.5:0.95` overall.

Results for the cross-generation baselines (`yolo5`–`yolo12`) are not included
here; those notebooks are provided so the comparison can be reproduced.

---

## Citations

If you use this code, please cite the dataset and the method it builds on.

**Dataset**

```
@misc{yolotb1l6,
  title        = {yolo-tb1l6 Dataset},
  author       = {ke-project},
  howpublished = {\url{https://universe.roboflow.com/ke-project/yolo-tb1l6/dataset/1}},
  year         = {2024},
  note         = {Roboflow Universe, CC BY 4.0}
}
```

**SPD-Conv**

```
@inproceedings{sunkara2022spdconv,
  title     = {No More Strided Convolutions or Pooling: A New CNN Building Block
               for Low-Resolution Images and Small Objects},
  author    = {Sunkara, Raja and Luo, Tie},
  booktitle = {ECML PKDD},
  year      = {2022}
}
```

**Ultralytics**

```
@software{ultralytics,
  author  = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  title   = {Ultralytics YOLO},
  version = {8.4.14},
  year    = {2024},
  url     = {https://github.com/ultralytics/ultralytics}
}
```

---

## License & Contribution Guidelines

The **dataset** is redistributed by Roboflow under **CC BY 4.0**; attribution
is required and the licence terms apply to any derivative of the images or
annotations.

The **code** in this repository depends on Ultralytics, which is licensed under
**AGPL-3.0**. Anything derived from it inherits that licence, so this
repository is released under **AGPL-3.0** as well. A commercial Ultralytics
licence is required for closed-source use.

Contributions are welcome. When proposing a change:

* Keep ablations single-variable — if a pull request changes the architecture,
  it should not also change the training schedule.
* State the environment (Ultralytics and PyTorch versions), since defaults
  shift between releases and results are not comparable across them.
* Report both validation and test metrics, and say which epoch was selected
  and on what criterion.
