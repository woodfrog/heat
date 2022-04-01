# HEAT- Holistic Edge Attention Transformer for Structured Reconstruction 

A quick guide to run HEAT with the outdoor architecture reconstruction benchmark.

## Environment prep

This repo was developed and tested with both ```Python3.7```

Install the required packages, and compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))


```
pip install -r requirements.txt
cd  models/ops/
sh make.sh
cd ...
```


Enter the ```./data``` directory, extract the outdoor architecture dataset:

```
unzip cities_dataset.zip

unzip det_final.zip
```


## Run the inference and evaluation

In ```infer.py```, set up the checkpoint path and the corresponding image resolution, then run:

```
python infer.py
```


## Run the training

In ```train.py```, set up the paths for saving intermediate training results and checkpoints, as well as the input image resolution, then run:

```
CUDA_VISIBLE_DEVICES={gpu_ids} python train.py
```

With the default batch size, we need at least 2 GPUs with >10GB memory. 

