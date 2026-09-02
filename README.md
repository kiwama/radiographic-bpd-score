# Radiographic BPD Likelihood Score

Inference code and trained model weights for the **radiographic bronchopulmonary dysplasia (BPD)
likelihood score**, a deep-learning score that quantifies BPD-related radiographic abnormalities
on chest radiographs of extremely preterm infants.

This repository accompanies the following study and is provided as its Data / Code Availability
material:

> Iwama K, Yasui M, Sakai S, Nomura R, Kemmotsu T, Yamashita R, Ito S.
> *Deep Learning–Based Radiographic Bronchopulmonary Dysplasia Likelihood Scoring Using Lung Field
> Segmentation on Chest Radiographs of Extremely Preterm Infants.*

> **This software is for research use only. It is not a medical device and must not be used for
> diagnosis or treatment decisions.**

---

## What the pipeline does

```
chest radiograph (grayscale)
        │
        ├─[1] U-Net lung field segmentation            (256 × 256 × 1 → 256 × 256 × 1)
        │
        ├─[2] left / right lung field extraction       (2 largest connected components)
        │
        ├─[3] three 48 × 48 patches per lung field     (upper / middle / lower thirds)
        │        position optimised to maximise the lung area inside each patch
        │
        └─[4] EfficientNet-B0 classifier (shared weights, 3 patches concatenated)
                 → sigmoid output 0–1  =  lung field-level BPD likelihood score
```

The **chest radiograph-level (radiographic) BPD likelihood score** reported in the paper is the
mean of the left and right lung field-level scores. The prespecified classification threshold,
derived by the Youden index on the model development test set, is **0.554**.

---

## Repository contents

| Path | Description |
|---|---|
| `BPD_PatchCropScoring.ipynb` | End-to-end inference notebook (segmentation → patch extraction → scoring → CSV) |
| `finetuning_keras_20260624/` | Lung field segmentation model, TensorFlow **SavedModel** format (U-Net). **Not in the repository — download it from the release, see below** |
| `patch_cropping_20260714.keras` | BPD likelihood scoring model, Keras v3 archive (EfficientNet-B0 × 3 patches) |
| `requirements.txt` | Python dependencies of the environment used in the study |
| `LICENSE` | MIT License — applies to the source code only |
| `LICENSE_WEIGHTS.md` | CC BY-NC 4.0 — applies to the trained model weights, with third-party data attribution |
| `CITATION.cff` | Citation metadata (GitHub renders a "Cite this repository" button) |
| `.gitattributes` | Text/binary handling rules (Git LFS is not used) |
| `.gitignore` | Excludes the large segmentation model, notebook outputs, checkpoints and virtual environments |
| `README.md` | This file |

### Model specifications

**Segmentation model** — `finetuning_keras_20260624/` (SavedModel, Keras 2.13.1)

- Architecture: U-Net (57 Conv2D / 12 Conv2DTranspose, skip connections), final `Conv2D(1, sigmoid)`
- Input: `(None, 256, 256, 1)`, grayscale, pixel values scaled to `[0, 1]`
- Loss / metrics used in training: `bce_dice_loss`, `dice_coef`, `dice_loss`
  (these must be passed as `custom_objects` when loading — see the notebook)
- Pretrained on three public chest radiograph datasets with lung masks (V7 Labs COVID-19,
  Montgomery County, Shenzhen), then fine-tuned on 100 manually annotated preterm radiographs

**Scoring model** — `patch_cropping_20260714.keras` (Keras 2.13.1, saved 2026-07-14)

- Three named inputs `upper`, `middle`, `lower`, each `(None, 192, 192, 3)`
- Each input passes a `Lambda` layer performing `efficientnet.preprocess_input(x * 255.0)`,
  therefore **the notebook must feed values in `[0, 1]`** (it does)
- One **weight-shared** ImageNet-pretrained EfficientNet-B0 encoder is applied to all three
  patches (siamese configuration); the three feature vectors are concatenated
- Head: `Dense(256, gelu)` → `Dropout(0.5)` → `Dense(1, sigmoid)`
- Class weights of 1.0 (non-BPD) and 3.0 (radiographic BPD) were used during training

---

## Requirements

The models were trained and validated in the following environment. Because the scoring model
stores its `Lambda` layers as **marshalled Python 3.8 bytecode**, loading it on a different Python
minor version may fail; please use Python 3.8.

| Package | Version used |
|---|---|
| Python | 3.8.10 |
| TensorFlow | 2.13.0 (Keras 2.13.1) |
| numpy | (compatible with TF 2.13) |
| opencv-python | any recent 4.x |
| scipy | any recent |
| pillow | any recent |
| pandas | any recent |
| matplotlib | any recent |

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon (as used in the study: MacBook Pro, Apple M3 Pro/Max) can instead install
`tensorflow-macos==2.13.0` with `tensorflow-metal` for GPU acceleration via the Metal (MPS) backend.

### Downloading the segmentation model weights

`finetuning_keras_20260624/` is approximately **356 MB**, which exceeds GitHub's 100 MiB per-file
limit, so it is **not stored in the repository**. It is attached to the corresponding GitHub
release instead. The scoring model `patch_cropping_20260714.keras` (~20 MB) is included in the
repository itself.

Download and unpack the segmentation model at the repository root:

```bash
git clone https://github.com/kiwama/radiographic-bpd-score.git
cd radiographic-bpd-score

# Download finetuning_keras_20260624.tar.gz from the latest release, e.g. with the GitHub CLI:
gh release download --pattern "finetuning_keras_20260624.tar.gz"
#  or download it manually from
#  https://github.com/kiwama/radiographic-bpd-score/releases

tar -xzf finetuning_keras_20260624.tar.gz
```

The repository should then contain:

```
radiographic-bpd-score/
├── BPD_PatchCropScoring.ipynb
├── patch_cropping_20260714.keras
└── finetuning_keras_20260624/
    ├── saved_model.pb
    ├── keras_metadata.pb
    └── variables/
```

## Usage

Open `BPD_PatchCropScoring.ipynb` and edit the paths at the top of the first cell:

```python
input_dir         = "PATH_to_input"           # directory of chest radiographs to score
segmentation_dir  = "PATH_to_segmentation"    # output: segmentation overview PNGs
regions_dir       = "PATH_to_regions"         # output: extracted left/right lung field PNGs
output_csv        = "PATH_to_output/output.csv"  # output: per-lung-field scores

segmentation_model = "PATH_to_MODEL/finetuning_keras_20260624"
scoring_model      = "PATH_to_MODEL/patch_cropping_20260714.keras"
```

Then run the cells in order:

| Section | Step |
|---|---|
| 1 | Load radiographs from `input_dir`, resize to 256 × 256 grayscale, scale to `[0, 1]`; create the output directories |
| 2 | Load the U-Net and predict masks (`preds > 0.5`) |
| 3 | Save an original / mask / overlay figure per radiograph into `segmentation_dir` |
| 4 | Extract the two largest lung field components, assign `right` / `left` by x-centroid, save PNGs into `regions_dir` |
| 5 | Define patch cropping (`crop_three_patches_predict`) and model input helpers |
| 6 | Load the scoring model and write per-lung-field scores to `output_csv` |
| 7 | Average the left and right scores and write the radiograph-level scores to `<output_csv>_radiograph_level.csv` |

### Input

- Any of `.jpg .jpeg .png .tif .tiff .bmp`, read as grayscale
- One radiograph per file; files are processed in sorted filename order
- Radiographs were acquired in routine clinical care; no windowing or histogram normalisation is
  applied beyond the resize and `[0, 1]` scaling shown above

### Output

`output_csv` contains one row per lung field:

| column | description |
|---|---|
| `file` | file name of the extracted lung field PNG (`<radiograph>_<side>.png`) |
| `side` | `right` or `left` (`unknown` if only one lung field component was found) |
| `score` | lung field-level BPD likelihood score, 0–1 (`NaN` if three patches could not be cropped) |

The last section of the notebook averages the two sides and writes the radiograph-level score
reported in the paper to `<output_csv stem>_radiograph_level.csv`:

| column | description |
|---|---|
| `radiograph` | source radiograph file name (without the `_left` / `_right` suffix) |
| `bpd_likelihood_score` | chest radiograph-level BPD likelihood score (mean of the available lung fields) |
| `n_lung_fields` | number of lung fields the mean is based on; `< 2` means one lung field could not be extracted or scored, and the value should be interpreted with care |
| `above_threshold` | whether the score exceeds the prespecified threshold of 0.554 |

---

## Important notes and caveats

- **Sign convention of the segmentation mask.** In the lung field extraction cell the predicted mask
  is inverted (`mask = ~mask`) before connected-component labelling, because the fine-tuned network
  outputs high values for the *non-lung* region. Consequently the "Binary Predicted Mask" panel in
  the segmentation figure displays the non-lung region in white. Do not remove the inversion.
- **Visual inspection is required.** Marked rotation, inappropriate exposure, or extensive overlying
  devices may impair lung field segmentation. Inspect the figures written to `segmentation_dir`
  before interpreting any score.
- **`safe_mode=False`** is required to load the scoring model because it contains `Lambda` layers,
  which deserialise arbitrary Python bytecode. Load this file only if you trust its source.
- **Generalisability.** The models were developed at two Japanese tertiary NICUs using the same
  radiographic imaging system, in infants born at <28 weeks' gestation with a birth weight <1,500 g.
  Performance on other imaging systems, populations, or gestational age ranges has not been
  evaluated. Calibration, resampling-based internal validation and fairness across sociodemographic
  subgroups were not assessed.
- **Data.** Individual chest radiographs cannot be shared publicly for privacy reasons. Per-infant
  characteristics are provided as Supplementary Table 2 of the paper; other datasets are available
  from the corresponding author on reasonable request.

---

## Citation

```bibtex
@article{iwama_rbpd_score,
  author  = {Iwama, Kazuhiro and Yasui, Masaki and Sakai, Shunsuke and Nomura, Ryosuke
             and Kemmotsu, Takahiro and Yamashita, Riu and Ito, Shuichi},
  title   = {Deep Learning--Based Radiographic Bronchopulmonary Dysplasia Likelihood Scoring
             Using Lung Field Segmentation on Chest Radiographs of Extremely Preterm Infants},
  year    = {2026},
  note    = {Manuscript}
}
```

## Ethics

The study was approved by the Ethics Committee of Yokohama City University School of Medicine
(approval number F240900003). Because of the retrospective design, written informed consent was
waived and an opt-out procedure was used.

## License

This repository uses two licenses:

| Component | License |
|---|---|
| Source code (`BPD_PatchCropScoring.ipynb` and any scripts) | **MIT** — see [`LICENSE`](LICENSE) |
| Trained model weights (`finetuning_keras_20260624/`, `patch_cropping_20260714.keras`) | **CC BY-NC 4.0** — see [`LICENSE_WEIGHTS.md`](LICENSE_WEIGHTS.md) |

The weights are non-commercial because the scoring model is fine-tuned from an
ImageNet-pretrained EfficientNet-B0, and the
[ImageNet Terms of Access](https://image-net.org/accessagreement) limit use of that database to
non-commercial research and educational purposes.

### Attribution for third-party data

The lung field segmentation model was pretrained on publicly available chest radiograph datasets
with lung segmentation masks — the V7 Labs COVID-19 X-ray dataset and the Montgomery County and
Shenzhen chest X-ray sets of the U.S. National Library of Medicine — obtained through a compiled
distribution whose files are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Changes were made:** the images were
resized, augmented and used to train a U-Net segmentation model. The images themselves are **not**
redistributed here, and nothing in this repository implies endorsement by the rights holders.
Full attribution and citations are given in [`LICENSE_WEIGHTS.md`](LICENSE_WEIGHTS.md).

## Contact

Kazuhiro Iwama — Department of Pediatrics, Graduate School of Medicine, Yokohama City University
<kiwama@yokohama-cu.ac.jp>
