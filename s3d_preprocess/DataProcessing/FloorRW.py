import numpy as np
import open3d as o3d
import os
import json
import time
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import io
import PIL
import time

from misc.figures import plot_coords
from misc.colors import colormap_255, semantics_cmap
from visualize_3d import visualize_floorplan

from DataProcessing.PointCloudReaderPanorama import PointCloudReaderPanorama


class FloorRW:
    def __init__(self):
        self.dataset_path = "./s3d_raw/"  # set the path to the raw data here
        self.mode = "train"
        self.scenes_path = os.path.join(self.dataset_path, self.mode)

        self.out_folder = "floor_data_with_normals"
        self.density_map_file_name = "density.png"
        self.normals_map_file_name = "normals.png"
        self.anno_file_name = "annotation_3d.json"
        self.vis_file_name = "vis.jpg"

        self.coco_floor_json_path = os.path.join(self.dataset_path, self.mode + "_floor.json")

        # TODO Don't change these values, Adapt PointCloudReaderPanorama first
        self.w = 256
        self.h = 256

        self.invalid_scenes = ["scene_00183", "scene_01155", "scene_01816"]

    def generate_floors(self):
        scenes = os.listdir(self.scenes_path)
        scenes.sort()



        for scene_ind, scene in enumerate(scenes):
            # if scene == "scene_01155":
            #     continue
            if scene in self.invalid_scenes:
                continue
            # if scene_ind < 178:
            # #     # if scene_ind != 0:
            #     continue
            print("%d / %d Current scene %s" % (scene_ind + 1, len(scenes), scene))
            start_time = time.time()

            scene_path = os.path.join(self.scenes_path, scene)
            # annotation_json = self.normalize_annotations(scene_path, {})

            reader = PointCloudReaderPanorama(scene_path, random_level=0, generate_color=True, generate_normal=False)
            density_map, normals_map, normalization_dict = reader.generate_density()

            normalized_annotations = self.normalize_annotations(scene_path, normalization_dict)

            # visualize_floorplan(normalized_annotations)
            # self.vis_scene_data(density_map, normalized_annotations)
            # reader.visualize()
            self.export_scene(scene_path, density_map, normals_map, normalized_annotations)

            print("Scene processing time %.3f" % (time.time() - start_time))

    def generate_coco_json(self):
        scenes = os.listdir(self.scenes_path)
        scenes.sort()

        img_id = -1
        instance_id = -1
        coco_dict = {}
        coco_dict["images"] = []
        coco_dict["annotations"] = []
        coco_dict["categories"] = [{"supercategory": "room", "id": 1, "name": "room"}]
        for scene_ind, scene in enumerate(scenes):
            # if scene_ind != 66:
            #     continue
            if scene in self.invalid_scenes:
                continue
            # if scene_ind > 1000:
            #     break

            img_id += 1
            print("%d / %d Current scene %s" % (scene_ind + 1, len(scenes), scene))

            scene_path = os.path.join(self.scenes_path, scene)

            img_relative_path = os.path.join("./", scene, self.out_folder, self.density_map_file_name)
            annos_path = os.path.join(scene_path, self.out_folder, self.anno_file_name)

            with open(annos_path, "r") as f:
                annos = json.load(f)

            img_dict = {}
            img_dict["file_name"] = img_relative_path
            img_dict["id"] = img_id
            img_dict["width"] = self.w
            img_dict["height"] = self.h

            coco_annotation_dict_list = self.parse_coco_annotation(annos, instance_id, img_id)

            coco_dict["images"].append(img_dict)
            coco_dict["annotations"] += coco_annotation_dict_list
            instance_id += len(coco_annotation_dict_list)

        with open(self.coco_floor_json_path, 'w') as f:
            json.dump(coco_dict, f)

    def parse_coco_annotation(self, annos, curr_instance_id, curr_img_id):
        polygons = visualize_floorplan(annos, vis=False, ret=True)

        ignore_types = ['outwall', 'door', 'window']

        coco_annotation_dict_list = []
        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
        for (poly, poly_type) in polygons:
            if poly_type in ignore_types:
                continue

            poly = junctions[np.array(poly)]
            poly_shapely = Polygon(poly)
            area = poly_shapely.area

            # assert area > 10
            if area < 100:
                continue

            rectangle_shapely = poly_shapely.envelope

            coco_seg_poly = []
            for p in poly:
                coco_seg_poly += list(p)

            # Slightly wider bounding box
            bound_pad = 5
            bb_x, bb_y = rectangle_shapely.exterior.xy
            bb_x = np.unique(bb_x)
            bb_y = np.unique(bb_y)
            bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
            bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

            bb_x_max = np.minimum(np.max(bb_x) + bound_pad, self.w - 1)
            bb_y_max = np.minimum(np.max(bb_y) + bound_pad, self.h - 1)

            bb_width = (bb_x_max - bb_x_min)
            bb_height = (bb_y_max - bb_y_min)

            coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

            curr_instance_id += 1
            coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": curr_img_id,
                "bbox": coco_bb,
                "category_id": 1,
                "id": curr_instance_id}
            coco_annotation_dict_list.append(coco_annotation_dict)

            # plt.figure()
            # plt.imshow(np.zeros((256, 256)))
            # x, y = poly_shapely.exterior.xy
            # plt.plot(x, y, "r")
            #
            # plt.plot([bb_x_min, bb_x_min + bb_width, bb_x_min + bb_width, bb_x_min, bb_x_min],
            #          [bb_y_min, bb_y_min, bb_y_min + bb_height, bb_y_min + bb_height, bb_y_min], "b")
            #
            # plt.show()

        return coco_annotation_dict_list



    def export_scene(self, scene_path, density_map, normals_map, annos):
        def export_density():
            density_path = os.path.join(floorplan_folder_path, self.density_map_file_name)
            density_uint8 = (density_map * 255).astype(np.uint8)
            cv2.imwrite(density_path, density_uint8)

        def export_normals():
            normals_path = os.path.join(floorplan_folder_path, self.normals_map_file_name)
            normals_uint8 = (np.clip(normals_map, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(normals_path, normals_uint8)

        def export_annos():
            anno_path = os.path.join(floorplan_folder_path, self.anno_file_name)
            with open(anno_path, 'w') as f:
                json.dump(annos, f)

        def export_vis():
            vis_path = os.path.join(floorplan_folder_path, self.vis_file_name)
            vis = self.vis_scene_data(density_map, annos, show=False)
            if vis is not None:
                cv2.imwrite(vis_path, vis)
            else:
                print("Visualization is None. Skip exporting the visualization...")

        floorplan_folder_path = os.path.join(scene_path, self.out_folder)
        if not os.path.isdir(floorplan_folder_path):
            os.mkdir(floorplan_folder_path)

        export_density()
        export_normals()
        export_annos()
        export_vis()

    def normalize_annotations(self, scene_path, normalization_dict):
        annotation_path = os.path.join(scene_path, "annotation_3d.json")
        with open(annotation_path, "r") as f:
            annotation_json = json.load(f)

        for line in annotation_json["lines"]:
            point = line["point"]
            point = self.normalize_point(point, normalization_dict)
            line["point"] = point

        for junction in annotation_json["junctions"]:
            point = junction["coordinate"]
            point = self.normalize_point(point, normalization_dict)
            junction["coordinate"] = point

        normalization_dict["min_coords"] = normalization_dict["min_coords"].tolist()
        normalization_dict["max_coords"] = normalization_dict["max_coords"].tolist()
        normalization_dict["image_res"] = normalization_dict["image_res"].tolist()

        return annotation_json

    def normalize_point(self, point, normalization_dict):

        min_coords = normalization_dict["min_coords"]
        max_coords = normalization_dict["max_coords"]
        image_res = normalization_dict["image_res"]

        point_2d = \
            np.round(
                (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)
        point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)),
                              image_res - 1)

        point[:2] = point_2d.tolist()

        return point

    def vis_scene_data(self, density_map, annos, show=True):
        polygons = visualize_floorplan(annos, vis=False, ret=True)

        if polygons is None:
            return None

        fig = plt.figure()
        gs = fig.add_gridspec(1, 2)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(density_map)
        plt.axis('equal')
        plt.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])

        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
        for (polygon, poly_type) in polygons:
            polygon = Polygon(junctions[np.array(polygon)])
            plot_coords(ax1, polygon.exterior, alpha=0.5)
            if poly_type == 'outwall':
                patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0)
            else:
                patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0.5)
            ax1.add_patch(patch)

        ax1.set_ylim(density_map.shape[0], 0)
        ax1.set_xlim(0, density_map.shape[1])
        plt.axis('equal')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        if show:
            plt.show()
        plt.close()

        vis = PIL.Image.open(buf)
        vis = np.array(vis)
        return vis
