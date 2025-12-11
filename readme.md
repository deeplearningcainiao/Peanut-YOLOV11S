🎯 Overview

P-YOLOv11s is an enhanced framework specifically designed for peanut seedling detection in UAV remote sensing imagery. Building upon YOLOv11, the framework addresses field heterogeneity challenges arising from morphological variations (genotype, growth stage, planting density) and imaging variability (flight altitude, solar angle) through three key innovations:

P2 Fine-grained Feature Layer: Enhances detection capability for small-sized seedlings, improving accuracy in early growth stages.

Asymptotic Feature Pyramid Network (AFPN): Achieves more effective multi-scale feature fusion, adapting to seedling size variations across different growth stages and flight altitudes.

Improved EMA Attention Mechanism (iEMA): Increases model robustness against inter-seedling occlusions, improving detection stability in dense planting conditions.

The method was validated on a dataset encompassing significant agronomic diversity, including 1025 genotypes, different nitrogen treatments, multiple planting years, varying planting densities, and ecological zones. It covers multiple developmental stages from three-leaf to six-leaf and considers different flight altitudes (15, 25, 40 m) and four diurnal imaging intervals.

🚀 Quick Start

Environment Setup

Clone this repository

git clone https://github.com/yourusername/P-YOLOv11s.git

cd P-YOLOv11s

Install dependencies

pip install -r requirements.txt

Data Preparation

Dataset paths are already configured in data.yaml, which correctly points to the above directory structure.

Model Training

Use Pretrained Weights

We provide the official YOLOv11s pretrained weight yolov11s.pt to accelerate convergence.

Start Training

python train.py --weights yolov11s.pt --cfg configs/pyolov11s.yaml --data data.yaml --epochs 200 --batch-size 16

Parameter description:

--weights: Path to initial weights

--cfg: Model configuration file (includes configurations for P2 layer, AFPN, and iEMA modules)

--data: Dataset configuration file

--epochs: Number of training epochs

--batch-size: Batch size (adjust according to GPU memory)

Innovative Modules

In this study, the architecture incorporates three core enhancements. Two of these are custom modules—the Asymptotic Feature Pyramid Network (AFPN) and the Inverted Efficient Multi-scale Attention (iEMA) mechanism—implemented in the AFPN.py and iEMA.py files located within the ultralytics/nn/Addmodules/ directory. Additionally, the model integrates the P2 fine-grained feature layer, which is natively implemented and directly adopted from the official Ultralytics 

framework.

Inference and Evaluation

After training, evaluate the model with:

python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --task test

📄 License

This implementation is based on Ultralytics YOLO (AGPL-3.0). Use of this code is subject to the AGPL-3.0 license terms.
