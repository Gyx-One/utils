import copy
import torch
import numpy as np
import open3d as o3d
import SimpleITK as itk
from scipy.spatial import distance
from skimage.measure import marching_cubes
from .common import get_param_from_rigid_matrix

# mesh io ops
def read_triangle_mesh(mesh_path):
    return o3d.io.read_triangle_mesh(mesh_path)

def write_triangle_mesh(mesh, mesh_path):
    return o3d.io.write_triangle_mesh(mesh_path, mesh)

# mesh gen ops
def get_mesh_from_vol(cube,mesh_colors=[0,255,0], use_colors=False):
    # cube: 3-dimensional array
    march_cube=1-cube
    # mesh gen
    mesh=o3d.geometry.TriangleMesh()
    if(np.sum(cube)==0):
        print("get_mesh_from_vol: sum(cube)=0")
        return mesh
    mesh_verts,mesh_faces,mesh_norms,_=marching_cubes(march_cube,step_size=1,allow_degenerate=False,method="lewiner")
    mesh.vertices=o3d.utility.Vector3dVector(mesh_verts)
    mesh.triangles=o3d.utility.Vector3iVector(mesh_faces)
    mesh.vertex_normals=o3d.utility.Vector3dVector(mesh_norms)
    mesh.compute_vertex_normals()
    # mesh connect component
    triangle_clusterids,cluster_triangle_nums,_=mesh.cluster_connected_triangles()
    triangle_clusterids=np.array(triangle_clusterids)
    cluster_triangle_nums=np.array(cluster_triangle_nums)
    remove_tids=cluster_triangle_nums[triangle_clusterids]<500
    mesh.remove_triangles_by_mask(remove_tids)
    # mesh smooth
    mesh=mesh.filter_smooth_taubin(number_of_iterations=50)
    mesh.compute_vertex_normals()
    vertex_num=np.array(mesh.vertices).shape[0]
    if(use_colors):
        mesh_colors=np.repeat(np.array(mesh_colors)[:,np.newaxis],repeats=vertex_num,axis=1).transpose([1,0])/255
        mesh.vertex_colors=o3d.utility.Vector3dVector(mesh_colors)
    nan_mask=np.isnan(np.sum(np.array(mesh.vertices),axis=1))
    if(np.sum(nan_mask)!=0):
        mesh.remove_vertices_by_mask(nan_mask)
    return mesh

# registration
def icp_mesh_registration(src_mesh,
                          dst_mesh,
                          threshold=5,
                          max_its=2000,
                          center_x=0,
                          center_y=0,
                          center_z=0,
                          center_align=True,
                          ret_trans_matrix=False,
                          ret_trans_param=False,
                          use_rad=True):
    from .pcd_op import get_pcd_from_mesh, icp_pcd_registration
    src_pcd = get_pcd_from_mesh(src_mesh)
    dst_pcd = get_pcd_from_mesh(dst_mesh)
    regis_pcd, regis_matrix = icp_pcd_registration(src_pcd=src_pcd,
                                                   dst_pcd=dst_pcd,
                                                   center_x=center_x,
                                                   center_y=center_y,
                                                   center_z=center_z,
                                                   threshold=threshold,
                                                   max_its=max_its,
                                                   center_align=center_align,
                                                   ret_trans_matrix=True,
                                                   use_rad=use_rad)
    regis_mesh = o3d.geometry.TriangleMesh()
    regis_mesh.vertices = regis_pcd.points
    regis_mesh.triangles = src_mesh.triangles
    regis_mesh.compute_vertex_normals()
    if(ret_trans_param):
        regis_param = get_param_from_rigid_matrix(regis_matrix,
                                                  center_x=center_x,
                                                  center_y=center_y,
                                                  center_z=center_z,
                                                  use_rad=use_rad,
                                                  )
        return regis_mesh, regis_param
    elif(ret_trans_matrix):
        return regis_mesh, regis_matrix
    else:
        return regis_mesh

# mesh metric
def cal_msd(mesh_pd, mesh_gt):
    mesh_pd_pts = np.array(mesh_pd.vertices)
    mesh_gt_pts = np.array(mesh_gt.vertices)
    pd_pts_tensor = torch.FloatTensor(mesh_pd_pts)[:,np.newaxis,:]
    gt_pts_tensor = torch.FloatTensor(mesh_gt_pts)[np.newaxis,:,:]
    dist_matrix = torch.sqrt(torch.sum((pd_pts_tensor-gt_pts_tensor)**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    mhd = torch.mean(dist_min).numpy()
    return mhd    

def get_msd_mesh(src_mesh,ref_mesh,norm_range=5):
    msd_mesh=copy.copy(src_mesh)
    ref_vertices=np.array(ref_mesh.vertices)
    src_vertices=np.array(src_mesh.vertices)
    dis_matrix=distance.cdist(src_vertices,ref_vertices,metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)]=100000
    min_dis_vector=np.min(dis_matrix,axis=1)
    norm_dis_vector=np.clip(min_dis_vector,0,norm_range)/norm_range
    msd_colors=np.array([0,255,0])[:,np.newaxis]*(1-norm_dis_vector)+np.array([255,0,0])[:,np.newaxis]*norm_dis_vector
    msd_colors=np.clip(msd_colors,0,255).transpose([1,0])/255
    msd_mesh.vertex_colors=o3d.utility.Vector3dVector(msd_colors)
    return msd_mesh
    