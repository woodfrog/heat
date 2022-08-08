from DataProcessing.path_variables import *
from DataProcessing.PointCloudReaderPanorama import PointCloudReaderPanorama
from DataProcessing.PointCloudReaderPerspective import PointCloudReaderPerspective

if __name__ == "__main__":
    scenes = [scene_path]
    print(scenes)
    for scene in scenes:

        reader = PointCloudReaderPanorama(scene, random_level=0, generate_color=True, generate_normal=False)
        path = "/home/sinisa/Sinisa_Projects/papers/ICCV21/supp_figures/blender_project/vis/" + scene_name + ".ply"
        # reader.export_ply(path)
        density_map = reader.generate_density()
        reader.visualize(export_path=path)

        # print("Creating point cloud from perspective views...")
        # reader = PointCloudReaderPerspective(scene, random_level=0, generate_color=True, generate_normal=False,
        #                                      generate_segmentation=True)
        # print("Subsampling point cloud...")
        # o3d_pcd = reader.subsample_pcd(seg=False)
        # reader.visualize(o3d_pcd)
        # reader.generate_density()


        # print("Writing point cloud...")
        # reader.export_ply_from_o3d_pcd(scene_ply_path, o3d_pcd, seg=False)
        #
        # print("Subsampling segmented point cloud...")
        # o3d_seg_pcd = reader.subsample_pcd(seg=True)
        # print("Writing segmented point cloud...")
        # reader.export_ply_from_o3d_pcd(scene_segmented_ply_path, o3d_seg_pcd, seg=True)