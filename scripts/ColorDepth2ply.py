from curses import KEY_EOL
import open3d as o3d
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
import yaml
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

depth_intrinsic  = np.array([[903.5896, 0, 631.955627],
                                 [0, 903.5896, 367.5744],
                                 [0, 0, 1]])

def color_depth2ply(rgb_img, depth_img):
    # Convert depth image to float values in meters
    depth_img = depth_img.astype(np.float32) / 1000.0

    # Create Open3D point cloud from RGB-D images
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        rgb_img.shape[1], rgb_img.shape[0], fx=depth_intrinsic[0][0], fy=depth_intrinsic[1][1], cx=depth_intrinsic[0][2], cy=depth_intrinsic[1][2])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_img),
        o3d.geometry.Image(depth_img),
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)
    return pcd

def find_center(points):
    center = np.ones((len(points), 3))
    for i, point in enumerate(points):
        center[i][0] = point[0][0]
        center[i][1] = point[0][1]
        center[i][2] = point[0][2]
    center = np.average(center, axis=0)
    return center

def find_k_mean(points, k):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return labels

def find_k_mean_plus(points, k):
    initial_centers = kmeans_plusplus_initializer(points, k).initialize()
    kmeans_instance = kmeans(points, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()
    return clusters

def find_x_mean(points, k):
    amount_initial_centers = k
    initial_centers = kmeans_plusplus_initializer(points, amount_initial_centers).initialize()
    xmeans_instance = xmeans(points, initial_centers, 4)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return clusters

def visualize(pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    ################ONE MATCH#############################
    rgb_image = cv2.imread("/home/yuzeren/CAM/amazon_ws/data/oak-D/left.png")
    depth_image = cv2.imread("/home/yuzeren/CAM/amazon_ws/data/oak-D/depth.png", cv2.IMREAD_UNCHANGED)
    depth_image[depth_image > 2000] = 0
    pcd = color_depth2ply(rgb_image, depth_image)
    visualize(pcd)
    #####################################################


    # Visualize the point cloud
    filedir_org = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples"
    store_dir = "/home/yuzeren/CAM/amazon_ws"
    # filedir_org = "/Users/YZR/Desktop/CAM/amazon_ws/AmazonPackageSamples"
    # store_dir = "/Users/YZR/Desktop/CAM/amazon_ws"

    yaml_name = "local.yaml"
    rgb_image = cv2.imread(os.path.join(filedir_org, "rgb_1680636420757.png"))
    depth_image = cv2.imread(os.path.join(filedir_org,"depth_1680636420757.png"), cv2.IMREAD_UNCHANGED)

    pcd = color_depth2ply(rgb_image, depth_image)
    # o3d.visualization.draw_geometries([pcd])

    # with open(os.path.join(store_dir, yaml_name), 'r') as f:
    #     data = yaml.safe_load(f)
    # plane = np.array(data["plane"])/1000
    # plane = plane / math.sqrt(np.sum(plane * plane))/1000

    # on_point = data["points"]
    # on_point = find_center(on_point) / 1000
    # points = np.asarray(pcd.points)
    # TODO: Draw vector
    # vec = on_point + plane
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(np.array([on_point, vec]))
    # line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    # line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(line_set.lines))])
    # TODO: Draw points
    # mesh_spheres = []
    # for point in data["points"]:
    #     center = np.array([point[0][0]/1000, point[0][1]/1000, point[0][2]/1000])
    #     radius = 0.000003
    #     mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    #     mesh_sphere.translate(center)
    #     mesh_spheres.append(mesh_sphere)
    ####################################################################################
    
    # TODO: filter points above plane
    # points = points - on_point
    # new_plane = np.zeros(points.shape)
    # new_plane[:] = plane
    # plane_magnitude = math.sqrt(np.sum(plane * plane))
    # solve = np.sum(new_plane * points/plane_magnitude, axis=1)
    # keep_indices = np.where(solve < -0.000007)
    # above_plane_pcd = pcd.select_by_index(keep_indices[0])
    # ####################################################################################

    # points = np.asarray(above_plane_pcd.points)
    # k_mean_labels = find_x_mean(points, 2)
    # keep_indices = k_mean_labels[0]
    # zero_pcd = above_plane_pcd.select_by_index(keep_indices)
    # to_view = [zero_pcd, line_set]# + mesh_spheres
    # o3d.visualization.draw_geometries(to_view)