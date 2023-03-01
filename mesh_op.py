import os
import copy
import torch
import numpy as np
import open3d as o3d
import SimpleITK as itk
from scipy.spatial import distance
from skimage.measure import marching_cubes
from .geom_op import get_param_from_rigid_matrix
from .common import make_parent_dir, common_json_load, common_json_dump

# mesh io ops
def read_triangle_mesh(mesh_path):
    return o3d.io.read_triangle_mesh(mesh_path)

def write_triangle_mesh(mesh, mesh_path):
    return o3d.io.write_triangle_mesh(mesh_path, mesh)

# mesh vertices access ops
def get_mesh_vertices(mesh):
    return np.array(mesh.vertices)

def set_mesh_vertices(mesh, vertices):
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

def get_mesh_vertices_pvmesh_arrayND(pv_mesh):
    pv_mesh_verts = copy.copy(np.array(pv_mesh.points))
    return pv_mesh_verts

def set_mesh_vertices_pvmesh_arrayND(pv_mesh, verts_array):
    pv_mesh.points = copy.copy(verts_array)
    return pv_mesh

def get_mesh_vertex_norm_compute_o3d(mesh):
    mesh.compute_vertex_normals()
    return mesh

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

def cpd_mesh_nonrigid_registration_probreg(
    src_mesh,
    dst_mesh,
    use_cuda=False
):
    from probreg import cpd
    
    # init cuda setting
    if use_cuda:
        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        cp = np
        to_cpu = lambda x: x
    
    # init verts
    src_mesh_verts = get_mesh_vertices(src_mesh)
    dst_mesh_verts = get_mesh_vertices(dst_mesh)
    src_cp_verts = cp.asarray(src_mesh_verts, dtype=cp.float32)
    dst_cp_verts = cp.asarray(dst_mesh_verts, dtype=cp.float32)
    # cpd nonrigid regis
    nonrigid_cpd = cpd.NonRigidCPD(src_cp_verts, use_cuda=use_cuda)
    regis_param, _, _ = nonrigid_cpd.registration(dst_cp_verts)
    regis_cp_verts = regis_param.transform(src_cp_verts)
    regis_mesh_verts = to_cpu(regis_cp_verts)
    # gen registered mesh
    regis_mesh = copy.copy(src_mesh)
    regis_mesh = set_mesh_vertices(regis_mesh, regis_mesh_verts)
    regis_mesh = get_mesh_vertex_norm_compute_o3d(regis_mesh)
    return regis_mesh

def zoom_mesh(mesh, zoom_ratio=[1.0, 1.0, 1.0]):
    mesh_vertices = np.array(mesh.vertices)
    zoom_mesh = copy.copy(mesh)
    zoom_mesh_vertices = mesh_vertices*(np.array(zoom_ratio)[np.newaxis, :])
    zoom_mesh.vertices = o3d.utility.Vector3dVector(zoom_mesh_vertices)
    return zoom_mesh

# mesh metric
def cal_msd_forward(
    mesh_src,
    mesh_dst,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd from 'mesh_src' to 'mesh_dst'.

        [Important] 

        This implementation use vertice of 'mesh_src' to find the closest vertice of 'mesh_dst', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_src' to 
        calculate the closest distance to the triangle piece of the 'mesh_dst'.
    """
    if(print_msg_flag):
        print("Calculate Msd Forward")
    mesh_src_pts = np.array(mesh_src.vertices)
    mesh_dst_pts = np.array(mesh_dst.vertices)
    src_pts_tensor = torch.FloatTensor(mesh_src_pts)[:,np.newaxis,:]
    dst_pts_tensor = torch.FloatTensor(mesh_dst_pts)[np.newaxis,:,:]
    dist_matrix = torch.sqrt(torch.sum((src_pts_tensor-dst_pts_tensor)**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    msd = torch.mean(dist_min).numpy()*scale
    return msd

def cal_msd_backward(
    mesh_src,
    mesh_dst,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd from 'mesh_dst' to 'mesh_src'.

        [Important] 
        
        This implementation use vertice of 'mesh_dst' to find the closest vertice of 'mesh_src', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_dst' to 
        calculate the closest distance to the triangle piece of the 'mesh_src'.
    """
    if(print_msg_flag):
        print("Calculate Msd Backward")
    cur_mesh_src = mesh_src
    cur_mesh_dst = mesh_dst
    msd = cal_msd_forward(
        mesh_src=cur_mesh_dst,
        mesh_dst=cur_mesh_src,
        scale=scale,
        print_msg_flag=False
    )
    return msd

def cal_msd_symmetric(
    mesh_a,
    mesh_b,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd symmetrically, average the result of 'cal_msd_forward' and 'cal_msd_backward'

        [Important] 

        This implementation use vertice of 'mesh_a/mesh_b' to find the closest vertice of 'mesh_b/mesh_a', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_a/mesh_b' to 
        calculate the closest distance to the triangle piece of the 'mesh_b/mesh_a'.
    """
    if(print_msg_flag):
        print("Calculate Msd Symmetric")
    # forward
    msd_forward = cal_msd_forward(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        print_msg_flag=False
    )
    # backward
    msd_backward = cal_msd_backward(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        print_msg_flag=False
    )
    msd_symmetric = (msd_forward + msd_backward)/2
    return msd_symmetric

def get_pv_mesh_msd_vector_arrayND(src_mesh, ref_mesh, scale=1.0):
    # src_mesh and ref_mesh are pv mesh object
    src_mesh_pts = src_mesh.points
    ref_mesh_pts = ref_mesh.points
    dis_matrix = distance.cdist(src_mesh_pts, ref_mesh_pts, metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)] = 100000
    msd_vector = np.min(dis_matrix, axis=1)*scale
    return msd_vector

def get_dist_matrix_arrayND(src_pts, dst_pts, set_nan_value=100000):
    """
    src_pts: array, [N, 3]
    dst_pts: array, [N, 3]
    """
    from scipy.spatial import distance
    input_src_pts = np.array(copy.copy(src_pts))
    input_dst_pts = np.array(copy.copy(dst_pts))
    dist_matrix = distance.cdist(input_src_pts, input_dst_pts, metric="euclidean")
    dist_matrix[np.isnan(dist_matrix)] = set_nan_value
    return dist_matrix

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

class MeshDrawer:
    def __init__(self,
                 camera_position_path,
                 show_windows=False,
                 window_size=[512,512]
                 ):
        import pyvista as pv
        self.camera_position_path = camera_position_path
        self.show_windows = show_windows
        self.window_size = window_size
        # plotter setting
        self.plotter_window_size = [self.window_size[1], self.window_size[0]]
        # set camera position
        make_parent_dir(self.camera_position_path)
    
    def load_camera_position(self, camera_position_path):
        if(os.path.exists(camera_position_path)):
            self.plotter.camera_position = common_json_load(camera_position_path)

    def save_camera_postion(self, camera_postion_path, print_camera_flag=False):
        common_json_dump(list(self.plotter.camera_position), camera_postion_path)
        if(print_camera_flag):
            print(self.plotter.camera_position)
    
    def draw_mesh(self,
                  pv_mesh,
                  background_color=None,
                  mesh_color=None,
                  save_screenshot_path=False,
                  **kwargs
                ):
        import pyvista as pv
        # init plotter
        self.plotter = pv.Plotter(off_screen=(not self.show_windows), 
                                  window_size=self.plotter_window_size)
        
        self.load_camera_position(self.camera_position_path)
        self.plotter.set_background(color=background_color)
        self.plotter.add_mesh(
            pv_mesh,
            color=mesh_color,
            **kwargs
        )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        self.save_camera_postion(self.camera_position_path)
    