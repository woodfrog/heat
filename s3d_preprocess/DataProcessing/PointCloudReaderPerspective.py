import cv2
import open3d
import os
import matplotlib.pyplot as plt
from PIL import Image
import json

from misc.utils import parse_camera_info
from sem_seg_utils import *
from visualize_3d import visualize_wireframe

class PointCloudReaderPerspective():

    def __init__(self, path, resolution="full", random_level=0, generate_color=False, generate_normal=False,
                 generate_segmentation=False):
        perspective_str = "perspective"
        self.path = path
        self.random_level = random_level
        self.resolution = resolution
        self.generate_color = generate_color
        self.generate_normal = generate_normal
        self.generate_segmentation = generate_segmentation
        sections = sorted([p for p in os.listdir(os.path.join(path, "2D_rendering"))])

        sections_views = [sorted(os.listdir(os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution]))) \
                          if os.path.isdir(os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution])) \
                          else [] \
                          for p in sections]

        self.depth_paths = []
        self.rgb_paths = []
        self.seg_paths = []
        self.normal_paths = []
        self.pose_paths = []
        for p, views in zip(sections, sections_views):
            if not os.path.isdir(os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution])):
                continue

            self.depth_paths += [os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution, v, "depth.png"]) for v in views]
            self.rgb_paths += [os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution, v, "rgb_rawlight.png"]) for v in views]
            self.seg_paths += [os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution, v, "semantic.png"]) for v in views]
            self.normal_paths += [os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution, v, "normal.png"]) for v in views]
            self.pose_paths += [os.path.join(*[path, "2D_rendering", p, perspective_str, self.resolution, v, "camera_pose.txt"]) for v in views]

        self.point_cloud = self.generate_point_cloud(self.random_level, color=self.generate_color,
                                                     normal=self.generate_normal,
                                                     seg=self.generate_segmentation)


    def read_camera_center(self):
        camera_centers = []
        print(self.camera_paths)
        print(self.depth_paths)
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], 'r') as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
            print(camera_centers)
        return camera_centers

    def generate_point_cloud(self, random_level=0, color=False, normal=False, seg=False):
        coords = []
        colors = []
        segmentations = []
        normals = []
        points = {}

        # Getting Coordinates
        for i in range(len(self.depth_paths)):
            print(i)
            # i = 13
            W, H = (1280, 720)
            depth_img = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 1000.
            inv_depth_mask = depth_img < .2
            depth_img[inv_depth_mask] = .2 # Why does this fix the problem?
            # rgb_img = cv2.imread(self.rgb_paths[i])
            # plt.subplot(121)
            # plt.imshow(rgb_img)
            # plt.subplot(122)
            # plt.imshow(depth_img)
            # plt.show()

            camera_pose = np.loadtxt(self.pose_paths[i])
            rot, trans, K = parse_camera_info(camera_pose, H, W, inverse=True)

            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = trans / 1000.
            inv_pose = np.linalg.inv(pose)

            xs, ys = np.meshgrid(range(W), range(H), indexing='xy')

            # xyz_homo = np.concatenate([xyz, np.ones_like(xs)], axis=0)
            # xyz_h_global = pose.dot(xyz_homo).T
            # xyz_global = xyz_h_global[:, :3] / xyz_h_global[:, 3][:, None]

            if color:
                rgb_img = cv2.imread(self.rgb_paths[i])
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                # xs, ys = np.meshgrid(range(1280), range(720), indexing='xy')
            if seg:
                seg_img = Image.open(self.seg_paths[i])
                # xs, ys = np.meshgrid(range(1280), range(720), indexing='xy')
                seg_labels = np.array(seg_img.convert(mode="P", palette=create_color_palette()))

                def seg_grad(seg1):
                    # [-1 0 1] kernel
                    dx = np.abs(seg1[:, 2:] - seg1[:, :-2])
                    dy = np.abs(seg1[2:, :] - seg1[:-2, :])

                    grad = np.zeros_like(seg1)
                    grad[:, 1:-1] = dx
                    grad[1:-1, :] = np.maximum(grad[1:-1, :], dy)

                    grad = grad != 0
                    return grad

                def depth_grad(depth1):
                    # [-1 0 1] kernel
                    dx = np.abs(depth1[:, 2:] - depth1[:, :-2])
                    dy = np.abs(depth1[2:, :] - depth1[:-2, :])

                    grad = np.zeros_like(depth1)
                    grad[:, 1:-1] = dx
                    grad[1:-1, :] = np.maximum(grad[1:-1, :], dy)

                    grad = np.abs(grad) > 0.1
                    return grad

                grad_mask = np.logical_and(depth_grad(depth_img), seg_grad(seg_labels))
                # kern = np.ones((3, 3), np.uint8)
                # seg_mask = cv2.dilate((seg_mask).astype(np.uint8), kernel=kern, iterations=1)

                # plt.imshow(seg_mask)
                # plt.show()
                # not_windows = np.argwhere(seg_labels != class_name_to_id['window'])
                # ys = not_windows[:, 0]
                # xs = not_windows[:, 1]
                #
                # seg_labels = np.tile(np.round(seg_labels)[ys, xs].reshape(-1, 1), reps=[1, 3])
                # seg_labels = np.tile(seg_labels[:, :, None], reps=[1, 1, 3]) / 255

                # valid_mask = np.argwhere(valid_mask == 0)
                # valid_mask = np.argwhere(np.logical_and(grad_mask == 0, seg_labels != class_name_to_id['window']))
                valid_mask = np.argwhere(grad_mask == 0)

                ys = valid_mask[:, 0]
                xs = valid_mask[:, 1]

                seg_labels[inv_depth_mask] = 38
                seg_labels = np.tile(np.round(seg_labels)[ys, xs].reshape(-1, 1), reps=[1, 3])

            zs = depth_img[ys, xs]
            xs = xs.reshape(1, -1)
            ys = ys.reshape(1, -1)
            zs = zs.reshape(1, -1)

            inverse_K = np.linalg.inv(K)

            xyz = (inverse_K[:3, :3].dot(np.concatenate([xs, ys, np.ones_like(xs)], axis=0)))
            xyz = zs * (xyz / np.linalg.norm(xyz, axis=0, ord=2))
            # xyz = zs * xyz
            xyz_o3d = open3d.geometry.PointCloud()
            xyz_o3d.points = open3d.utility.Vector3dVector(xyz.T)
            xyz_o3d.transform(pose)
            xyz_global = np.asarray(xyz_o3d.points)

            segmentations += list(seg_labels)
            colors += list(rgb_img[ys, xs].reshape(-1,3))
            coords += list(xyz_global)
            # break

        points['coords'] = np.asarray(coords) * 1000.
        points['colors'] = np.asarray(colors) / 255.0
        points['segs'] = np.asarray(segmentations)




        # if normal:
        #     # Getting Normal
        #     for i in range(len(self.normal_paths)):
        #         print(self.normal_paths[i])
        #         normal_img = cv2.imread(self.normal_paths[i])
        #         for x in range(normal_img.shape[0]):
        #             for y in range(normal_img.shape[1]):
        #                 normals.append(normalize(normal_img[x, y].reshape(-1, 1)).ravel())
        #     points['normals'] = normals



        return points

    def get_point_cloud(self):
        return self.point_cloud

    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_down_sample(ind)
        # outlier_cloud = cloud.select_down_sample(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        # outlier_cloud.paint_uniform_color([1, 0, 0])
        # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        return inlier_cloud

    def visualize(self, o3d_pcd=None):
        # input("Max depth?")
        print("Visualizing...")
        pcd = open3d.geometry.PointCloud()

        if o3d_pcd is None:
            pcd.points = open3d.utility.Vector3dVector(self.point_cloud['coords'])
            # if self.generate_normal:
            #     pcd.normals = open3d.utility.Vector3dVector(self.point_cloud['normals'])

            # if False and self.generate_segmentation:
            #     pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['segs'] / 255.)
            # elif self.generate_color:
            #     pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['colors'])
            # pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['colors'])
            pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['segs'] / 255.)
        else:
            pcd = o3d_pcd

        vis = open3d.visualization.Visualizer()
        # vis.create_window(window_name="O3D")
        vis.create_window(window_name="O3D", width=1280, height=720, left=0, top=0,
                          visible=True)  # use visible=True to visualize the point cloud
        # vis.get_render_option().light_on = False
        # vis.get_render_option().point_size = 20

        vis.add_geometry(pcd)


        with open("/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/Datasets/Structured_3D_dataset/Structured3D/Structured3D_0/Structured3D/train/scene_00015/annotation_3d.json") as file:
            annos = json.load(file)

        wireframe_geo_list = visualize_wireframe(annos, vis=False, ret=True)

        vis.add_geometry(wireframe_geo_list[0])
        vis.add_geometry(wireframe_geo_list[1])

        # for view_ind in range(len(self.pose_paths)):
        #     # if view_ind != 25:
        #     #     continue
        #     W, H = (1280, 720)
        #     depth_img = cv2.imread(self.depth_paths[view_ind], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 1000.
        #
        #     # rgb_img = cv2.imread(self.rgb_paths[i])
        #     # plt.subplot(121)
        #     # plt.imshow(rgb_img)
        #     # plt.subplot(122)
        #     # plt.imshow(depth_img)
        #     # plt.show()
        #
        #     camera_pose = np.loadtxt(self.pose_paths[view_ind])
        #     rot, trans, K = parse_camera_info(camera_pose, H, W, inverse=True)
        #
        #     pose = np.eye(4)
        #     pose[:3, :3] = rot
        #     pose[:3, 3] = trans / 1000.
        #
        #     camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        #     fx, fy = camera_param.intrinsic.get_focal_length()
        #     cx = camera_param.intrinsic.intrinsic_matrix[0, 2]
        #     cy = camera_param.intrinsic.intrinsic_matrix[1, 2]
        #     camera_param.intrinsic.set_intrinsics(camera_param.intrinsic.width, camera_param.intrinsic.height,
        #                                           K[0, 0], K[1, 1], cx, cy)
        #     camera_param.extrinsic = np.linalg.inv(pose)
        #     ctr = vis.get_view_control()
        #     ctr.convert_from_pinhole_camera_parameters(camera_param)
        #     depth_render = vis.capture_depth_float_buffer(do_render=True)
        #     depth_render = np.asarray(depth_render)
        #
        #
        #     camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        #     print("My_intr", K)
        #     print("O3D_intr", camera_param.intrinsic.intrinsic_matrix)
        #     print("view ind", view_ind)
        #
        #     print("Plot")
        #     plt.subplot(131)
        #     plt.imshow(depth_img)
        #     plt.subplot(132)
        #     plt.imshow(depth_render)
        #     plt.subplot(133)
        #     plt.imshow(np.abs(depth_render - depth_img))
        #     plt.show()

        vis.run()
        vis.destroy_window()

    def generate_density(self, width=256, height=256):

        ps = self.point_cloud["coords"]

        unique_coords, unique_ind = np.unique(np.round(ps / 10) * 10., return_index=True, axis=0)

        ps = unique_coords


        image_res = np.array((width, height))

        max_coords = np.max(ps, axis=0)
        min_coords = np.min(ps, axis=0)
        max_m_min = max_coords - min_coords

        max_coords = max_coords + 0.1 * max_m_min
        min_coords = min_coords - 0.1 * max_m_min


        # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
        coordinates = \
            np.round(
                (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
        coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                    image_res - 1)

        density = np.zeros((height, width), dtype=np.float32)

        unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
        print(np.unique(counts))
        # counts = np.minimum(counts, 2e3)
        #
        unique_coordinates = unique_coordinates.astype(np.int32)

        density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
        density = density / np.max(density)
        # print(np.unique(density))

        plt.figure()
        plt.imshow(density)
        plt.show()

        return density

    def subsample_pcd(self, seg=False):
        # input("Max depth?")
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.point_cloud['coords'])
        # if self.generate_normal:
        #     pcd.normals = open3d.utility.Vector3dVector(self.point_cloud['normals'])

        if seg:
            pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['segs'] / 255.)
        else:
            pcd.colors = open3d.utility.Vector3dVector(self.point_cloud['colors'])

        final_pcd = pcd
        final_pcd, inds = pcd.remove_statistical_outlier(nb_neighbors=10,
                                                            std_ratio=3.0)
        #
        final_pcd = final_pcd.uniform_down_sample(every_k_points=10)
        return final_pcd

    def export_ply_from_o3d_pcd(self, path, pcd, seg=False):
        '''
        ply
        format ascii 1.0
        comment Mars model by Paul Bourke
        element vertex 259200
        property float x
        property float y
        property float z
        property uchar r
        property uchar g
        property uchar b
        property float nx
        property float ny
        property float nz
        end_header
        '''

        coords = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.int32)
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % coords.shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if self.generate_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")
            for i in range(coords.shape[0]):
                coord = coords[i].tolist()
                color = colors[i].tolist()
                data = coord + color
                f.write(" ".join(list(map(str,data)))+'\n')
