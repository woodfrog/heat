# Structured3D preprocessing for floorplan data

We thank the authors of [MonteFloor](https://openaccess.thecvf.com/content/ICCV2021/papers/Stekovic_MonteFloor_Extending_MCTS_for_Reconstructing_Accurate_Large-Scale_Floor_Plans_ICCV_2021_paper.pdf) for providing the preprocessing scripts to generate the floorplan data from Structured3D dataset. 


We prepare the training data for HEAT based on the generated density/normal images and the raw floorplan annotations. Note that all the data used in our paper can be downloaded from [our links](https://github.com/woodfrog/heat#data), and this readme doc is an inexhaustive explanation for those who interested in the data preprocessing process.


## Generate floorplan data (the original readme provided by MonteFloor)

This code is based on Structured3D repository. 

To generate floorplans, run generate_floors.py script:

```
python generate_floors.py
```

Prior to that, you should modify path variables in DataProcessing.FloorRW. (daataset_path, and mode)


Some scenes have missing/wrong annotations. These are the indices that you should additionally exclude from test set:

```
wrong_s3d_annotations_list = [3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]
```


## Generate the training annotations for HEAT

In HEAT's formulation, each floorplan is represented by a planar graph. However, the raw annotations from Structured3D represent the floorplan by a list of closed loops. To prepare the **ground-truth training data** for HEAT, we need to further process the raw annotations to get proper planar graphs. We refer to the room merging step of [Floor-SP](https://arxiv.org/abs/1908.06702) and implement a merging algorithm (in ```generate_planar_graph.py```) to generate planar graphs from the raw annotations. 

**Note**: the generated planar graphs are **only used for training HEAT**. For evaluation, we extract the rooms from the estimtaed planar graph as closed loops and follow the original evaluation pipeline established by MonteFloor. Check the [quantitative evaluation section](https://github.com/woodfrog/heat#floorplan-reconstruction) for the details.

Please run the script ```generate_planar_graph.py``` to merge the rooms and get the training annotations for HEAT. 




