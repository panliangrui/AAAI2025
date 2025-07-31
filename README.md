

<div align="center">
  <a href="(https://github.com/panliangrui/AAAI2026/blob/main/STAS.jpg)">
    <img src="https://github.com/panliangrui/AAAI2026/blob/main/STAS.jpg" width="800" height="400" />
  </a>

  <h1>STAS(micropapillary clusters, single cells, solid nests)</h1>

  <p>
  Anonymous Author et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md)

  </samp>
  </strong>
  </div>
</div>

# STAMP: Multi‑pattern Attention‑aware Multiple Instance Learning for STAS Diagnosis in Multi‑center Histopathology Images

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
- [Feature Extraction](#Feature-Extraction)
- [Models](#Train-models)
- [Train Models](#Train-models)
- [Datastes](#Datastes)
- [Installation](#Installation)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.

### Feature Extraction

Features extracted based on GigaPath.
Please refer to Prov-GigaPath: [https://github.com/Xiyue-Wang/TransPath](https://github.com/prov-gigapath/prov-gigapath)

Feature extraction code reference project: [https://github.com/mahmoodlab/CLAM](https://huggingface.co/prov-gigapath/prov-gigapath)


## Models
**STAMP**

  <a href="(https://github.com/panliangrui/AAAI2026/blob/main/liucheng.jpg)">
    <img src="https://github.com/panliangrui/AAAI2026/blob/main/liucheng.jpg" width="800" height="400" />
  </a>

The complete workflow for diagnosing STAS from histopathological images. a): Annotation (cross-validation) of histopathological images from lung cancer patients and digitization of WSIs; b): Preprocessing of WSIs, including segmentation, tessellation, and patching; c): Preprocessing of WSI image features, dual-token embedding, feature extraction (including transformer-based instance encoding and MPAA modules), and STAS diagnosis (classification with regularized similarity loss



**Baseline MIL Methods**

This repository provides implementations and comparisons of various MIL-based methods for Whole Slide Image (WSI) classification.

- **Maxpooling**: Represents a slide by selecting the instance with the maximum activation, thereby mimicking the focus on the most prominent lesion.
- **Meanpooling**: Aggregates all instance features by computing their mean, thus treating each patch equally in the overall representation.
- **ABMIL**: Employs an attention mechanism to assign weights to instances, effectively prioritizing diagnostically relevant regions.
- **TransMIL**: A transformer-based MIL framework that leverages both morphological and spatial correlations among instances to enhance visualization, interpretability, and performance in WSI pathology classification.
- **CLAM-SB**: A clustering constraint-based attention MIL method that employs a single attention branch to aggregate instance features and generate a bag-level representation.
- **CLAM-MB**: The multi-branch version of the CLAM model, computing attention scores for each class separately to produce multiple unique bag-level representations.
- **DTFD-MIL**: Addresses the challenge of limited WSI samples in MIL by introducing pseudo-bags to virtually enlarge the bag count and implementing a double-tier framework that leverages an attention-based derivation of instance probabilities to effectively utilize intrinsic features.
- **ILRA**: Incorporates a pathology-specific Low-Rank Constraint for feature embedding and an iterative low-rank attention model for feature aggregation, achieving enhanced performance in gigapixel-sized WSI classification.
- **DSMIL**: Utilizes a dual-stream architecture, where one stream classifies instances and the other aggregates contextual information for final prediction.

## Train Models
```markdown
python ./baseline/train/manage.py
```


## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
link: https://pan.baidu.com/s/1k5XTvMyP_xO2bd-WRY187A?pwd=fgws password: fgws 
```
Please contact the corresponding author or first author by email.

- STAS‑SXH:We provide clinical data on STAS patients, including patient age, gender, stage and protein level expression data.
- We included WSIs from the Xiangya Second Hospital of Central South University cohort as our internal training and validation dataset. Among the 12,169 patients who underwent lung nodule resection and were histologically confirmed with LUAD between April 2020 and December 2023, 206 individuals with a definitive diagnosis of STAS and 150 individuals without STAS were selected. From these selected cases, we collected a total of 1,290 WSIs (approximately four slides per patient), along with associated clinical data and immunohistochemistry images (e.g., markers such as TTF‑1, CK, CD68).

- STAS‑TXH: We included a cohort from the Third Xiangya Hospital of Central South University, consisting of 304 histopathology slides from 68 lung cancer patients diagnosed with STAS between 2022 and 2023, with each WSI annotated for the presence or absence of STAS.

- STAS‑TCGA: We collected 417 WSIs from 366 patients in the TCGA\_LUAD cohort dataset based on the inclusion/exclusion method. All WSIs are accompanied by corresponding STAS labels.

## Installation
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 4090)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.1.2.30), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.5.10).


## License
If you need the original histopathology image slides, please send a request to our email address. The email address will be announced after the paper is accepted. Thank you!

[License MIT](../LICENSE)
