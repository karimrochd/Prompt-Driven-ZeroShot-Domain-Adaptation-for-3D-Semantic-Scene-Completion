# PØDA-MonoScene: Zero-Shot Domain Adaptation for 3D Semantic Scene Completion



> **Integrating Prompt-driven Zero-shot Domain Adaptation (PØDA) with MonoScene for robust 3D scene understanding under adverse weather conditions**

This project combines two state-of-the-art methods to enable **zero-shot domain adaptation** of 3D Semantic Scene Completion (SSC) models to adverse weather conditions (fog, rain, snow) using only **text prompts**—no target domain images required during training.


## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Method](#-method)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [References](#-references)

## 🎯 Overview

### Problem Statement
3D Semantic Scene Completion (SSC) models trained on clear weather data suffer significant performance degradation when deployed in adverse conditions (fog, rain, snow). Traditional domain adaptation requires collecting and labeling target domain data, which is expensive and sometimes dangerous.

### My Solution
I integrate **PØDA** (Prompt-driven Zero-shot Domain Adaptation) with **MonoScene** (Monocular 3D SSC) to adapt a source-trained model using only natural language descriptions of target conditions:
- `"driving in fog"`
- `"driving under rain"`  
- `"driving in snow"`

### Papers
- **PØDA**: [Prompt-driven Zero-shot Domain Adaptation](https://arxiv.org/abs/2212.03241) (Fahes et al., ICCV 2023)
- **MonoScene**: [Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726) (Cao & de Charette, CVPR 2022)

## 🌟 Key Contributions

1. **First integration** of prompt-driven domain adaptation with 3D semantic scene completion
2. **CLIP-based backbone replacement**: Replace MonoScene's EfficientNetB7 with CLIP ResNet-50 to enable vision-language alignment
3. **PIN layer insertion**: Apply Prompt-driven Instance Normalization after Layer1 for optimal style transfer
4. **Unified style mining**: Mine and combine styles from multiple adverse weather domains
5. **Zero-shot adaptation**: Adapt to fog/rain/snow without any target domain images

## 🔬 Method

### Overview

The method consists of three phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PØDA + MonoScene Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: Source-Only Training                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │  RGB    │───▶│  CLIP   │───▶│  FLoSP  │───▶│  3D     │───▶ SSC     │
│  │  Image  │    │  RN50   │    │         │    │  UNet   │    Output    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘              │
│                                                                         │
│  Phase 2: Style Mining (Offline)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  For each target prompt ("driving in fog", "driving in rain"):  │   │
│  │  • Extract Layer1 features from source images                    │   │
│  │  • Optimize (μ, σ) to minimize cosine distance to text embedding│   │
│  │  • Store mined styles in style bank                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 3: Zero-Shot Adaptation                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │  RGB    │───▶│  CLIP   │───▶│   PIN   │───▶│  Rest   │───▶ SSC     │
│  │  Image  │    │ Layer1  │    │  Layer  │    │  of Net │    Output    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘              │
│                      │              ▲                                   │
│                      │              │ Sample random style               │
│                      │         ┌────┴────┐                              │
│                      └────────▶│  Style  │                              │
│                                │  Bank   │                              │
│                                └─────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Prompt-driven Instance Normalization (PIN)

PIN transforms source features toward target domain style:

```
f_{s→t} = σ_t * ((f_s - μ(f_s)) / σ(f_s)) + μ_t
```

Where:
- `f_s`: Source feature map from Layer1 (shape: B × 256 × H × W)
- `μ(f_s), σ(f_s)`: Channel-wise mean and std of source features
- `μ_t, σ_t`: Target style statistics (optimized via CLIP)

### Style Mining

For each source image, we optimize style statistics to minimize:

```
L(μ, σ) = 1 - cos(f̄_{s→t}, TrgEmb)
```

Where:
- `f̄_{s→t}`: CLIP embedding of stylized features
- `TrgEmb`: CLIP text embedding of target prompt (e.g., "driving in fog")

## 🏗 Architecture

### Model Components

```
MonoScenePODA
├── CLIPBackbone (frozen)
│   ├── stem (conv1-3, avgpool)
│   ├── layer1 → 256 channels  ← PIN insertion point
│   ├── layer2 → 512 channels
│   ├── layer3 → 1024 channels
│   └── layer4 → 2048 channels
│
├── PIN Layer
│   └── Prompt-driven Instance Normalization
│
├── FLoSP (Features Line of Sight Projection)
│   ├── 1x1 conv projections for each scale
│   └── 2D→3D feature lifting via ray casting
│
├── 3D UNet
│   ├── Encoder (DDR blocks, 2 layers)
│   ├── 3D CRP (Context Relation Prior)
│   └── Decoder (deconv layers)
│
└── Completion Head
    ├── 3D ASPP (dilations: 1, 2, 3)
    └── Softmax → 20 classes
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone | CLIP RN50 | Vision-language alignment for PØDA |
| PIN Location | After Layer1 | Low-level features encode style; high-level encode content |
| Freeze Strategy | Freeze all except Completion Head | Preserve CLIP latent space compatibility |

## 📁 Dataset Setup

### SemanticKITTI (Source Domain)

```
data/SemanticKITTI/
├── dataset/
│   └── sequences/
│       ├── 00/
│       │   ├── image_2/
│       │   ├── calib.txt
│       │   └── voxels/
│       │       ├── 000000.bin
│       │       ├── 000000.label
│       │       └── 000000.invalid
│       ├── 01/ ... 10/
│       └── 08/  (validation)
└── semantic-kitti.yaml
```


### SemanticSTF (Target Domain - Evaluation Only)

```
data/SemanticSTF/
├── train/
│   ├── rgb/
│   ├── voxel/
│   └── calib/
└── weather_split.json
```


## 📈 Results

### SemanticSTF Evaluation (Zero-Shot Transfer)

| Weather | Source SC-IoU | Adapted SC-IoU | Δ IoU | Source mIoU | Adapted mIoU | Δ mIoU |
|---------|--------------|----------------|-------|-------------|--------------|--------|
| Snow | 20.95% | **41.84%** | +20.89% | 2.49% | **4.44%** | +1.95% |
| Rain | 15.66% | **39.87%** | +24.21% | 2.27% | **4.60%** | +2.33% |
| Dense Fog | 16.39% | **47.50%** | +31.11% | 2.41% | **6.07%** | +3.66% |
| Light Fog | 20.04% | **50.76%** | +30.72% | 2.67% | **5.03%** | +2.36% |

**Key Observations:**
- **Significant SC-IoU improvements**: +20-31% across all weather conditions
- **Consistent mIoU gains**: +2-4% semantic accuracy improvement
- **Best performance on fog**: Dense fog shows largest improvement (+31% IoU)
- **Zero-shot transfer**: No target domain images used during training

### SemanticKITTI Validation (Source Domain Retention)

| Metric | Value |
|--------|-------|
| SC-IoU | 16.97% |
| mIoU | 9.33% |

## 📚 References

```bibtex
@inproceedings{fahes2023poda,
  title={PØDA: Prompt-driven Zero-shot Domain Adaptation},
  author={Fahes, Mohammad and Vu, Tuan-Hung and Bursuc, Andrei and P{\'e}rez, Patrick and de Charette, Raoul},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{cao2022monoscene,
  title={MonoScene: Monocular 3D Semantic Scene Completion},
  author={Cao, Anh-Quan and de Charette, Raoul},
  booktitle={CVPR},
  year={2022}
}


```

## 🙏 Acknowledgments

- [PØDA](https://github.com/astra-vision/PODA) by Astra-Vision
- [MonoScene](https://github.com/astra-vision/MonoScene) by Astra-Vision
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [SemanticKITTI](http://www.semantic-kitti.org/)
- [SemanticSTF](https://github.com/xiaoaoran/SemanticSTF)

