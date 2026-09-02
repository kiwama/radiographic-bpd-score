# License for the trained model weights

The trained model weights distributed with this repository —

- `finetuning_keras_20260624/` (lung field segmentation model, U-Net)
- `patch_cropping_20260714.keras` (BPD likelihood scoring model, EfficientNet-B0)

— are licensed under the

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- Human-readable summary: https://creativecommons.org/licenses/by-nc/4.0/
- Full legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode

You are free to share and adapt the weights for **non-commercial** purposes,
provided that you give appropriate credit, provide a link to the license, and
indicate if changes were made. This license does **not** apply to the source
code in this repository, which is released under the MIT License (see `LICENSE`).

## How to give credit

> Radiographic BPD likelihood score model weights by Kazuhiro Iwama et al.,
> licensed under CC BY-NC 4.0. https://github.com/kiwama/radiographic-bpd-score

Please also cite the accompanying paper (see `CITATION.cff`).

## Why non-commercial

The scoring model is fine-tuned from an ImageNet-pretrained EfficientNet-B0.
The ImageNet Terms of Access limit use of that database to non-commercial
research and educational purposes (https://image-net.org/accessagreement), and
this restriction is treated here as extending to weights derived from it. The
NonCommercial term is therefore applied to both model files for consistency
and to avoid any ambiguity for downstream users.

## Third-party data used to train the segmentation model

The lung field segmentation model was pretrained on publicly available chest
radiograph datasets with lung segmentation masks, obtained through a compiled
distribution of the following sources. The weights are a derivative work
(adapted material) produced by training on those images; **the images
themselves are not redistributed in this repository.**

- **V7 Labs, COVID-19 X-ray dataset** — https://github.com/v7labs/covid-19-xray-dataset
- **Montgomery County chest X-ray set**, U.S. National Library of Medicine
- **Shenzhen chest X-ray set**, U.S. National Library of Medicine
  - Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. Two public chest
    X-ray datasets for computer-aided screening of pulmonary diseases.
    *Quant Imaging Med Surg.* 2014;4(6):475-7.
  - Jaeger S, Karargyris A, Candemir S, et al. Automatic tuberculosis screening
    using chest radiographs. *IEEE Trans Med Imaging.* 2014;33(2):233-45.
  - Candemir S, Jaeger S, Palaniappan K, et al. Lung segmentation in chest
    radiographs using anatomical atlases with nonrigid registration.
    *IEEE Trans Med Imaging.* 2014;33(2):577-90.

The compiled distribution from which these images were obtained states that its
files are licensed under a **Creative Commons Attribution 4.0 International
(CC BY 4.0)** license (https://creativecommons.org/licenses/by/4.0/), with the
reservation that further permission may be required for any content within the
dataset identified as belonging to a third party. **Changes were made:** the
images were resized, augmented and used to train a U-Net lung field
segmentation model. Nothing in this repository implies endorsement by the
rights holders of the source datasets.

CC BY 4.0 contains no ShareAlike term, so applying a NonCommercial license to
the derived weights is permitted. This does not alter the license of the
original datasets, which remain available from their respective distributors
under their own terms.

## No warranty

The weights are provided "as is", without warranty of any kind. **This is
research software. It is not a medical device and must not be used for
diagnosis or treatment decisions.**
