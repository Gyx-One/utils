"""
本文件包括的是和点云有关函数 point cloud functions
"""
import copy
import torch
import numpy as np
import open3d as o3d
from .common import make_parent_dir
from .geom_op import get_param_from_rigid_matrix

# pcd io ops
def read_point_cloud(pcd_path):
    return o3d.io.read_point_cloud(pcd_path)

def write_point_cloud(pcd, pcd_path):
    make_parent_dir(pcd_path)
    return o3d.io.write_point_cloud(pcd_path, pcd)

# pcd gen ops
def get_pcd_from_mesh(mesh, compute_normal=True):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices))
    if(compute_normal):
        pcd.estimate_normals()
    return pcd

def get_pcd_from_pts(pts, compute_normal=True):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if(compute_normal):
        pcd.estimate_normals()
    return pcd

# pcd points access ops
def get_pcd_vertices(pcd_o3d):
    return np.array(pcd_o3d.points) 

def set_pcd_vertices(pcd_o3d, vertices, compute_norm=True):
    pcd_o3d.points = o3d.utility.Vector3dVector(vertices)
    if(compute_norm):
        pcd_o3d.estimate_normals()
    return pcd_o3d

# pcd warp by itksnap-format bspline grid func
def itk_warp_pcd_func(pcd_o3d, itk_transform):
    process_itk_transform = copy.copy(itk_transform)
    process_itk_transform.SetInverse()

    process_pcd = copy.copy(pcd_o3d)
    trans_pcd = copy.copy(pcd_o3d)
    pcd_vertices = get_pcd_vertices(process_pcd)

    trans_pcd_vertices_list = []
    for v_idx in range(0, pcd_vertices.shape[0]):
        pcd_vertice = list(pcd_vertices[v_idx])
        trans_pcd_vertice = process_itk_transform.TransformPoint(pcd_vertice)
        trans_pcd_vertices_list.append(trans_pcd_vertice)
    trans_pcd = set_pcd_vertices(trans_pcd, np.array(trans_pcd_vertices_list))
    trans_pcd.estimate_normals()
    return trans_pcd

# pcd registration ops
def icp_pcd_registration(src_pcd,
                         dst_pcd,
                         threshold=5,
                         max_its=2000,
                         center_x=0,
                         center_y=0,
                         center_z=0,
                         center_align=True,
                         ret_trans_matrix=False,
                         ret_trans_param=False,
                         use_rad=True):
    regis_src_pcd = copy.deepcopy(src_pcd)
    regis_dst_pcd = copy.deepcopy(dst_pcd)
    center_translation = np.array([0, 0, 0])
    # center align pre-process
    if(center_align):
        regis_src_aabb = regis_src_pcd.get_axis_aligned_bounding_box()
        regis_dst_aabb = regis_dst_pcd.get_axis_aligned_bounding_box()
        center_translation = np.array(regis_dst_aabb.get_center()) - np.array(regis_src_aabb.get_center())
        regis_src_pcd.translate(center_translation)
    # registration
    regis_trans = o3d.registration.registration_icp(
        regis_src_pcd, regis_dst_pcd, threshold, np.eye(N=4),
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration = max_its)
        )
    regis_matrix = regis_trans.transformation.copy()
    # center aligh post-process
    if(center_align):
        regis_matrix[:3, 3] += regis_matrix[:3, :3]@center_translation
    regis_pcd = copy.deepcopy(src_pcd)
    regis_pcd.transform(regis_matrix)
    if(ret_trans_param):
        regis_param = get_param_from_rigid_matrix(regis_matrix,
                                                  center_x=center_x,
                                                  center_y=center_y,
                                                  center_z=center_z,
                                                  use_rad=use_rad)
        return regis_pcd, regis_param
    if(ret_trans_matrix):
        return regis_pcd, regis_matrix
    else:
        return regis_pcd


def get_point_dist_matrix_arrayND(point_array):
    # point array shape: [N, M], N is the number of points, M is the dimension of point space
    dist_matrix = np.sqrt(np.sum(np.square(point_array[:, np.newaxis, :] - point_array[np.newaxis, :, :]),  axis=2))
    return dist_matrix
