# HEAT: Holistic Edge Attention Transformer for Structured Reconstruction 

![image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 

Official PyTorch implementation of the paper [HEAT: Holistic Edge Attention Transformer for Structured Reconstruction](https://arxiv.org/abs/2111.15143) (**CVPR 2022**).

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

Enter the ```./data``` directory, extract the outdoor architecture dataset:

```
unzip cities_dataset.zip

unzip det_final.zip
```

### Checkpoints


## Inference, evaluation, and visualization

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

