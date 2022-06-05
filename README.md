# HEAT: Holistic Edge Attention Transformer for Structured Reconstruction 

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" width="8%" /> [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 

Official implementation of the paper [HEAT: Holistic Edge Attention Transformer for Structured Reconstruction](https://arxiv.org/abs/2111.15143) (**CVPR 2022**).

Please use the following bib entry to cite the paper if you are using resources from this repo.

```
@inproceedings{chen2022heat,
     title={HEAT: Holistic Edge Attention Transformer for Structured Reconstruction},
     author={Chen, Jiacheng and Qian, Yiming and Furukawa, Yasutaka},
     booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
     year={2022}
} 
```

## Introduction

<img src="assets/img/problem_description.png" width="90%">

This paper focuses on a typical family of structured reconstruction tasks: planar graph reconstruction. Two different tasks are included: 1) outdoor architecture reconstruction from a satellite image; or 2) floorplan reconstruction from a point density image. The above below shows examples. The key contributions of the paper are:

- a transformer-based architecture w/ state-of-the-art performance and efficiency on two different tasks, w/o domain specific heuristics
- a geometry-only decoder branch w/ a masked training strategy to enhance the geometry learning


<img src="assets/img/pipeline.png" width="90%">

As shown by the above figure, the overall pipeline of our method consists of three key steps: 1) edge node initialization; 2) edge image feature fusion and edge filtering; and 3) holistic structural reasoning with two weight-sharing transformer decoders. Please refer to the paper for more details.


This repo provides the code, data, and pre-trained checkpoints of HEAT for the two tasks covered in the paper.


## Preparation

### Environment

This repo was developed and tested with both ```Python3.7```

Install the required packages, and compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))

```
pip install -r requirements.txt
cd  models/ops/
sh make.sh
cd ...
```

### Data

Please download the data for the two tasks from the [link](https://drive.google.com/file/d/1BL58xl2U8H96YBkkB7WjDmtznuEj6PbG/view?usp=sharing) here.  Extract the data into the ```./data``` directory.

The file structure should be like the following:
```
data
├── outdoor
│   ├── cities_dataset  # the outdoor architecture dataset from previous works
│   │      ├── annot    # the G.T. planar graphs
│   │      ├── rgb      # the input images
│   │      ├── ......   # dataset splits, miscs
│   │
│   └── det_finals   # corner detection results from previous works (not used by our full method, but used for ablation studies) 
│
└── s3d_floorplan       # the Structured3D floorplan dataset, produced with the scripts from MonteFloor
    ├── annot           # the G.T. planar graphs 
    │
    ├── density         # the point density images
    │
    │── ......          # dataset splits, miscs
```
Note that the Structured3D floorplan data is generated with the scripts provided by MonteFloor[1]. We thank the authors for kindly sharing the processing scripts, please cite their paper if you use the corresponding resources. 

### Checkpoints

We provide the checkpoints for our full method under [this link](https://drive.google.com/file/d/1Oua4RCaxOIm7-mWXoUJZHTNE3oSPrDTw/view?usp=sharing), please download and extract.


## Inference, evaluation, and visualization

We provide the instructions to run the inference, quantitative evaluation, and qualitative visualization in this section.

### Inference

In ```infer.py```, set up the checkpoint path and the corresponding image resolution, then run:

```
python infer.py
```


## Training

In ```train.py```, set up the paths for saving intermediate training results and checkpoints, as well as the input image resolution, run:

```
CUDA_VISIBLE_DEVICES={gpu_ids} python train.py  --output_dir {ckpts_output_dir}
```

With the default setting (e.g., model setup, batch size, etc.), training the full HEAT (i.e., the end-to-end corner and edge modules) needs at least 2 GPUs with >15GB memory each. 



## References

