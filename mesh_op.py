"""
本文件包括的是和三角形网格有关函数 triangle mesh functions
"""
import os
import copy
import math
import torch
import numpy as np
import open3d as o3d
import pyvista as pv
import SimpleITK as itk
from scipy.spatial import distance
from tqdm import tqdm
from skimage.measure import marching_cubes
from .volume_op import *
from .image_op import get_color_rgb_float_from_hex, get_color_Npoint_array_rgb_from_float
from .geom_op import get_param_from_rigid_matrix
from .common import make_parent_dir, common_json_load, common_json_dump

###############################
#       mesh io functions
###############################
# read o3d mesh function
def read_triangle_mesh(mesh_path):
    return o3d.io.read_triangle_mesh(mesh_path)

# write o3d mesh function
def write_triangle_mesh(mesh, mesh_path, make_parent_dir_flag=True):
    if(make_parent_dir_flag):
        make_parent_dir(mesh_path)
    return o3d.io.write_triangle_mesh(mesh_path, mesh)

# read pyvista mesh function
def read_triangle_mesh_pv(mesh_path):
    import pyvista as pv
    return pv.read(mesh_path)

# write pyvista mesh function
def write_triangle_mesh_pv(pv_mesh, mesh_path, make_parent_dir_flag=True):
    if(make_parent_dir_flag):
        make_parent_dir(mesh_path)
    pv_mesh.save(mesh_path, binary=False)

###################################
#  mesh elements access functions
###################################
# o3d mesh vertices access ops
def get_triangle_mesh_vertices(mesh):
    return np.array(mesh.vertices)

def set_triangle_mesh_vertices(mesh, vertices, compute_norm=True):
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if(compute_norm):
        mesh.compute_vertex_normals()
    return mesh

# pyvista mesh vertices access ops
def get_triangle_mesh_vertices_pv(pv_mesh):
    pv_mesh_verts = copy.copy(np.array(pv_mesh.points))
    return pv_mesh_verts

def set_triangle_mesh_vertices_pv(pv_mesh, verts_array):
    pv_mesh.points = copy.copy(verts_array)
    return pv_mesh

# o3d mesh faces access ops
def get_triangle_mesh_faces(mesh):
    return np.array(mesh.triangles)

def set_triangle_mesh_faces(mesh, faces):
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

# o3d mesh colors access ops
def get_triangle_mesh_color(o3d_mesh):
    color_array = np.array(o3d_mesh.vertex_colors)
    return color_array

def set_triangle_mesh_color(o3d_mesh, mesh_colors):
    """
    Inputs:
        o3d_mesh: open3d Triangle Mesh
        color: np.array, range [0, 255], length is 3
    """
    vertex_num = get_triangle_mesh_vertices(o3d_mesh).shape[0]
    mesh_colors = np.repeat(np.array(mesh_colors)[:,np.newaxis],repeats=vertex_num,axis=1).transpose([1,0])/255
    o3d_mesh.vertex_colors=o3d.utility.Vector3dVector(mesh_colors)
    return o3d_mesh

###################################
#  mesh transformation functions
###################################
# o3d mesh norm compute function
def get_mesh_vertex_norm_compute_o3dmesh(mesh):
    mesh.compute_vertex_normals()
    return mesh

# o3d mesh zoom function
def zoom_mesh(mesh, zoom_ratio=[1.0, 1.0, 1.0]):
    mesh_vertices = np.array(mesh.vertices)
    zoom_mesh = copy.copy(mesh)
    zoom_mesh_vertices = mesh_vertices*(np.array(zoom_ratio)[np.newaxis, :])
    zoom_mesh.vertices = o3d.utility.Vector3dVector(zoom_mesh_vertices)
    return zoom_mesh

# o3d mesh filter smooth functions
# o3d mesh smooth taubin
def get_filter_smooth_taubin_mesh_o3d(o3d_mesh, num_iters=50, compute_vertex_norm_flag=True):
    smooth_o3d_mesh = copy.copy(o3d_mesh).filter_smooth_taubin(number_of_iterations=num_iters)
    if(compute_vertex_norm_flag):
        smooth_o3d_mesh.compute_vertex_normals()
    return smooth_o3d_mesh

# o3d mesh smooth simple
def get_filter_smooth_simple_mesh_o3d(o3d_mesh, num_iters=50, compute_vertex_norm_flag=True):
    smooth_o3d_mesh = copy.copy(o3d_mesh).filter_smooth_simple(number_of_iterations=num_iters)
    if(compute_vertex_norm_flag):
        smooth_o3d_mesh.compute_vertex_normals()
    return smooth_o3d_mesh

# o3d mesh keep connected component
def keep_component_mesh_o3d(o3d_mesh,
                           filter_triangle_num_thresh=500
                          ):
    """
    notes:
        calculate connected component for o3d_mesh
        keep the connected-components that has triangle_num larger than filter_triangle_num_thresh
    """
    triangle_clusterids, cluster_triangle_nums, _ = o3d_mesh.cluster_connected_triangles()
    triangle_clusterids = np.array(triangle_clusterids)
    cluster_triangle_nums = np.array(cluster_triangle_nums)
    remove_tids = cluster_triangle_nums[triangle_clusterids]<filter_triangle_num_thresh
    o3d_mesh.remove_triangles_by_mask(remove_tids)
    return o3d_mesh

# remove o3d mesh nan-triangles
def get_nan_triangle_removed_mesh_o3d(o3d_mesh):
    """
    notes:
        remove nan-triangles in open3d-mesh
    """
    nan_mask = np.isnan(np.sum(np.array(o3d_mesh.vertices),axis=1))
    if(np.sum(nan_mask)!=0):
        o3d_mesh.remove_vertices_by_mask(nan_mask)
    return o3d_mesh

# o3d mesh translate ops
def translate_triangle_mesh_o3d(mesh, translation):
    """
    translate o3d mesh
    """
    mesh.translate(translation)
    return mesh

# pyvista mesh translate ops
def translate_triangle_mesh_pv(pv_mesh, translation):
    """
    translate pv mesh
    """
    pv_mesh.translate(translation)
    return pv_mesh

def get_crop_range_mesh_o3d(mesh,
                        crop_min_x=0,
                        crop_max_x=128,
                        crop_min_y=0,
                        crop_max_y=127,
                        crop_min_z=0,
                        crop_max_z=127):
    """
    对mesh的顶点裁减至一定范围内
    notes: 
        1. range boundary is not perserved. crop (min, max), rather than crop [min, max].
        2. [x, y, z] in scan coordinates is [l, w, h] in CBCT 3D volumns.
    """
    crop_mesh = copy.copy(mesh)
    crop_mesh_verts = get_triangle_mesh_vertices(crop_mesh)
    gt_x_select = crop_mesh_verts[:, 0] > crop_max_x
    lt_x_select = crop_mesh_verts[:, 0] < crop_min_x
    gt_y_select = crop_mesh_verts[:, 1] > crop_max_y
    lt_y_select = crop_mesh_verts[:, 1] < crop_min_y
    gt_z_select = crop_mesh_verts[:, 2] > crop_max_z
    lt_z_select = crop_mesh_verts[:, 2] < crop_min_z
    outof_boundary_select = np.logical_or.reduce([
        gt_x_select, lt_x_select, gt_y_select, lt_y_select, gt_z_select, lt_z_select
    ])
    outof_boundary_indexs = np.where(outof_boundary_select)[0]
    crop_mesh.remove_vertices_by_index(outof_boundary_indexs)
    crop_mesh = get_mesh_vertex_norm_compute_o3dmesh(crop_mesh)
    return crop_mesh

def itk_warp_triangle_mesh_o3d_func(mesh, itk_transform):
    """
    根据itksnap格式的bspline grid.txt对mesh进行形变
    """
    process_itk_transform = copy.copy(itk_transform)
    process_itk_transform.SetInverse()

    process_mesh = copy.copy(mesh)
    trans_mesh = copy.copy(mesh)
    mesh_vertices = get_triangle_mesh_vertices(process_mesh)

    trans_mesh_vertices_list = []
    for v_idx in range(mesh_vertices.shape[0]):
        mesh_vertice = list(mesh_vertices[v_idx])
        trans_mesh_vertice = process_itk_transform.TransformPoint(mesh_vertice)
        trans_mesh_vertices_list.append(trans_mesh_vertice)
    trans_mesh = set_triangle_mesh_vertices(trans_mesh, np.array(trans_mesh_vertices_list))
    trans_mesh.compute_vertex_normals()
    return trans_mesh


###################################
#  mesh generation functions
###################################
# o3d mesh generation by verts and faces function
def get_o3d_mesh_by_verts_and_faces(
    mesh_verts,
    mesh_faces,
    mesh_norms=None
):
    """
    Input:
        mesh_verts [np.array]: mesh vertices array, shape [N, 3]
        mesh_faces [np.array]: mesh faces array, shape[K, 3]
        mesh_norms [np.array]: mesh vertices normals array, shape [N, 3]
        N is number of vertices, K is number of triangle faces 
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
    if(mesh_norms is not None):
        mesh.vertex_normals=o3d.utility.Vector3dVector(mesh_norms)
    mesh.compute_vertex_normals()
    return mesh

def exam_mesh(mesh, name="default"):
    vertices = get_triangle_mesh_vertices(mesh)
    print(f"name: {name} vertices: {vertices.shape}")

# o3d mesh generation from volumn by skimage marching cube
def get_mesh_from_vol(cube,
                      filter_smooth_taubin_flag=True,
                      filter_smooth_taubin_iter=50,
                      filter_smooth_simple_flag=False,
                      filter_smooth_simple_iter=2,
                      filter_connected_component_flag=True,
                      filter_num_triangle_thresh=500,
                      use_colors_flag=False,
                      mesh_colors=[0,255,0],
                      step=1):
    """
    notes:
        using marching_cube from skimage library
    """
    
    # cube: 3-dimensional array
    march_cube=1-cube
    # mesh gen
    mesh=o3d.geometry.TriangleMesh()
    if(np.sum(cube)==0):
        print("get_mesh_from_vol: sum(cube)=0")
        return mesh
    # marching cube
    mesh_verts, mesh_faces, mesh_norms, _ = marching_cubes(march_cube,
                                                           step_size=step,
                                                           allow_degenerate=False,
                                                           method="lewiner")
    
    # gen o3d_mesh
    mesh = get_o3d_mesh_by_verts_and_faces(
        mesh_verts=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_norms=mesh_norms
    )
    
    # mesh connect component
    if(filter_connected_component_flag):
        mesh = keep_component_mesh_o3d(o3d_mesh=mesh, filter_triangle_num_thresh=filter_num_triangle_thresh)
        
    # mesh smooth taubin
    if(filter_smooth_taubin_flag):
        mesh = get_filter_smooth_taubin_mesh_o3d(o3d_mesh=mesh, num_iters=filter_smooth_taubin_iter)
        
    # mesh smooth simple
    if(filter_smooth_simple_flag):
        mesh = get_filter_smooth_simple_mesh_o3d(o3d_mesh=mesh, num_iters=filter_smooth_simple_iter)
    
    # mesh color
    if(use_colors_flag):
        mesh = set_triangle_mesh_color(o3d_mesh=mesh, mesh_colors=mesh_colors)
        
    # mesh nan mask
    mesh = get_nan_triangle_removed_mesh_o3d(o3d_mesh=mesh)
    return mesh

# o3d mesh generation from volumn by mcube library marching cube
def get_mesh_from_vol_mcube(cube,
                            filter_smooth_taubin_flag=False,
                            filter_smooth_taubin_iter=50,
                            filter_smooth_simple_flag=False,
                            filter_smooth_simple_iter=2,
                            filter_connected_component_flag=True,
                            filter_num_triangle_thresh=500,
                            use_colors_flag=False,
                            mesh_colors=[0,255,0],
                            flip_flag=False):
    """
    notes:
        using marching_cube from mcubes library
        the generated mesh is more smooth than function `get_mesh_from_vol` which uses marching_cubes from skimage library
    """
    
    import mcubes
    # marching cube
    mesh_verts, mesh_faces = mcubes.marching_cubes(cube, 0.5)
    if(flip_flag):
        mesh_verts[:, 0] = 230 - mesh_verts[:, 0]
    
    # gen o3d_mesh
    mesh = get_o3d_mesh_by_verts_and_faces(
        mesh_verts=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_norms=None
    )
    
    # mesh connect component
    if(filter_connected_component_flag):
        mesh = keep_component_mesh_o3d(o3d_mesh=mesh, filter_triangle_num_thresh=filter_num_triangle_thresh)
        
    # mesh smooth taubin
    if(filter_smooth_taubin_flag):
        mesh = get_filter_smooth_taubin_mesh_o3d(o3d_mesh=mesh, num_iters=filter_smooth_taubin_iter)
        
    # mesh smooth simplt
    if(filter_smooth_simple_flag):
        mesh = get_filter_smooth_simple_mesh_o3d(o3d_mesh=mesh, num_iters=filter_smooth_simple_iter)

    # mesh color
    if(use_colors_flag):
        mesh = set_triangle_mesh_color(o3d_mesh=mesh, mesh_colors=mesh_colors)

    # mesh nan mask
    mesh = get_nan_triangle_removed_mesh_o3d(o3d_mesh=mesh)

    return mesh

# o3d mesh generation from volumn routine with smooth for visualization
def get_mesh_from_vol_smoothvis(cube):
    mesh = get_mesh_from_vol_mcube(cube, flip_flag=False)
    mesh = get_filter_smooth_taubin_mesh_o3d(o3d_mesh=mesh, num_iters=50)
    mesh = get_filter_smooth_simple_mesh_o3d(o3d_mesh=mesh, num_iters=2)
    return mesh

# get_mesh_from_vol function variant using mesh path as input 
def get_mesh_from_vol_from_path(cube_path,
                                mesh_path,
                                filter_smooth_taubin_flag=True,
                                filter_smooth_taubin_iter=50,
                                filter_smooth_simple_flag=False,
                                filter_smooth_simple_iter=2,
                                filter_connected_component_flag=True,
                                filter_num_triangle_thresh=500,
                                use_colors_flag=False,
                                mesh_colors=[0,255,0],
                                step=1):
    cube = read_mha_array3D(cube_path)
    mesh = get_mesh_from_vol(
        cube=cube,
        filter_smooth_taubin_flag=filter_smooth_taubin_flag,
        filter_smooth_taubin_iter=filter_smooth_taubin_iter,
        filter_smooth_simple_flag=filter_smooth_simple_flag,
        filter_smooth_simple_iter=filter_smooth_simple_iter,
        filter_connected_component_flag=filter_connected_component_flag,
        filter_num_triangle_thresh=filter_num_triangle_thresh,
        use_colors_flag=use_colors_flag,
        mesh_colors=mesh_colors,
        step=step
    )
    ret = write_triangle_mesh(mesh=mesh, mesh_path=mesh_path)
    return ret

# get_mesh_from_vol_mcube function variant using mesh path as input 
def get_mesh_from_vol_mcube_from_path(cube_path,
                                      mesh_path,
                                      filter_smooth_taubin_flag=False,
                                      filter_smooth_taubin_iter=50,
                                      filter_smooth_simple_flag=False,
                                      filter_smooth_simple_iter=2,
                                      filter_connected_component_flag=True,
                                      filter_num_triangle_thresh=500,
                                      keep_component_flag=False,
                                      keep_component_num=100000,
                                      use_colors_flag=False,
                                      mesh_colors=[0,255,0],
                                      flip_flag=False):
    cube = read_mha_array3D(cube_path)
    if(keep_component_flag):
        cube = keep_component_3D(cube, keep_num=keep_component_num)
    mesh = get_mesh_from_vol_mcube(
        cube=cube,
        filter_smooth_taubin_flag=filter_smooth_taubin_flag,
        filter_smooth_taubin_iter=filter_smooth_taubin_iter,
        filter_smooth_simple_flag=filter_smooth_simple_flag,
        filter_smooth_simple_iter=filter_smooth_simple_iter,
        filter_connected_component_flag=filter_connected_component_flag,
        filter_num_triangle_thresh=filter_num_triangle_thresh,
        use_colors_flag=use_colors_flag,
        mesh_colors=mesh_colors,
        flip_flag=flip_flag
    )
    ret = write_triangle_mesh(mesh=mesh, mesh_path=mesh_path)
    return ret

# get_mesh_from_vol_smoothvis function variant using mesh path as input 
def get_mesh_from_vol_smoothvis_from_path(label_path, mesh_path):
    label = read_mha_array3D(label_path)
    mesh = get_mesh_from_vol_smoothvis(label)
    write_triangle_mesh(mesh, mesh_path)
    return True

######################################################################
#  specific pyvista mesh generation functions for visualization
######################################################################
# generate pyvista line mesh
def get_pv_mesh_line(src_point,
                     dst_point,
                     resolution=1,
                     cvt_tube_flag=False,
                     cvt_tube_radius=0.2):
    """
    Notes:
        generate a pyvista `Line` object with 2 end points to draw a straight line.
        
        `Line` object is straight line, if need to draw a multiple line, use `get_pv_mesh_multi_line`
        
        `Line` object can has alias and not smooth, if need to draw a smooth line, use `get_pv_mesh_spline`

    Input:
        src_point:  start point, array, [x,y,z]
        dst_point:  destination point, array, [x,y,z]
        resolution: Number of pieces to divide line into.
        cvt_cube_flag:  whether return the `tube` object generated from the current line.
        cvt_cube_radius:    the radius of the generated tube
    """
    line_mesh = pv.Line(pointa=src_point, pointb=dst_point, resolution=resolution)
    if(cvt_tube_flag):
        line_mesh = line_mesh.tube(radius=cvt_tube_radius)
    return line_mesh

# generate pyvista multiple line mesh
def get_pv_mesh_multi_line(point_list,
                           cvt_tube_flag=False,
                           cvt_tube_radius=0.2):
    """
    Notes:
        generate a pyvista `MultiLine` object with multiple points to draw a flet line.
        
        `MultiLine` object can has alias and not smooth, if need to draw a smooth line, use `get_pv_mesh_spline`

    Input:
        point_list: list of points, each point is an array [x,y,z]
        cvt_cube_flag:  whether return the `tube` object generated from the current multiline.
        cvt_cube_radius:    the radius of the generated tube
    """
    multi_line_mesh = pv.MultipleLines(points=point_list)
    if(cvt_tube_flag):
        multi_line_mesh = multi_line_mesh.tube(radius=cvt_tube_radius)
    return multi_line_mesh

# generate pyvista spline mesh, more smooth than `line` and `multi-line` 
def get_pv_mesh_spline(point_list,
                       sample_num=1000,
                       cvt_tube_flag=False,
                       cvt_tube_radius=0.2):
    """
    Notes:
        generate a pyvista `Spline` object with multiple points to draw a smooth curve
        
    Input:
        point_list: list of points, each point is an array [x,y,z]
        sample_num: number of sampling points in the created spline
        cvt_cube_flag:  whether return the `tube` object generated from the current spline.
        cvt_cube_radius:    the radius of the generated tube
        
    """
    point_array = np.array(point_list)
    spline_mesh = pv.Spline(point_array, sample_num)
    if(cvt_tube_flag):
        spline_mesh = spline_mesh.tube(radius=cvt_tube_radius)
    return spline_mesh

# generate pyvista sphere mesh
def get_pv_mesh_sphere(center, radius):
    sphere_mesh = pv.Sphere(radius=radius, center=center)
    return sphere_mesh

# generate pyvista a list of sphere mesh
def get_pv_mesh_shpere_list(center_list, radius):
    sphere_mesh_list = []
    for center in center_list:
        sphere_mesh_list.append(pv.Sphere(radius=radius, center=center))
    return sphere_mesh_list


##################################
#  mesh registration functions
##################################
# o3d mesh icp registration function
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
    """
    Mesh Rigid registration using ICP from open3d
    """
    
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

# o3d mesh probreg non-rigid registration (from git repository)
def cpd_mesh_nonrigid_registration_probreg(
    src_mesh,
    dst_mesh,
    use_cuda=False
):
    """
    Mesh Non-rigid registration using probreg (from a github repository)
    """
    
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
    src_mesh_verts = get_triangle_mesh_vertices(src_mesh)
    dst_mesh_verts = get_triangle_mesh_vertices(dst_mesh)
    src_cp_verts = cp.asarray(src_mesh_verts, dtype=cp.float32)
    dst_cp_verts = cp.asarray(dst_mesh_verts, dtype=cp.float32)
    # cpd nonrigid regis
    nonrigid_cpd = cpd.NonRigidCPD(src_cp_verts, use_cuda=use_cuda)
    regis_param, _, _ = nonrigid_cpd.registration(dst_cp_verts)
    regis_cp_verts = regis_param.transform(src_cp_verts)
    regis_mesh_verts = to_cpu(regis_cp_verts)
    # gen registered mesh
    regis_mesh = copy.copy(src_mesh)
    regis_mesh = set_triangle_mesh_vertices(regis_mesh, regis_mesh_verts)
    regis_mesh = get_mesh_vertex_norm_compute_o3dmesh(regis_mesh)
    return regis_mesh

# cpd_mesh_nonrigid_registration_probreg function variant using mesh path as input
def cpd_mesh_nonrigid_registration_probreg_by_path(
    src_mesh_path,
    dst_mesh_path,
    regis_mesh_path,
    use_cuda=False
):
    src_mesh = read_triangle_mesh(src_mesh_path)
    dst_mesh = read_triangle_mesh(dst_mesh_path)
    regis_mesh = cpd_mesh_nonrigid_registration_probreg(
        src_mesh=src_mesh,
        dst_mesh=dst_mesh,
        use_cuda=use_cuda
    )
    write_triangle_mesh(regis_mesh, regis_mesh_path)

##############################################
#           mesh metric functions
##############################################
# naive msd metric (forward msd)
def cal_msd_init_version(mesh_pd, mesh_gt):
    """
    注意： 此函数由于只考虑单向forward的msd, 不合理, 已被遗弃, 为legacy
    """
    mesh_pd_pts = np.array(mesh_pd.vertices)
    mesh_gt_pts = np.array(mesh_gt.vertices)
    pd_pts_tensor = torch.FloatTensor(mesh_pd_pts)[:,np.newaxis,:]
    gt_pts_tensor = torch.FloatTensor(mesh_gt_pts)[np.newaxis,:,:]
    dist_matrix = torch.sqrt(torch.sum((pd_pts_tensor-gt_pts_tensor)**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    mhd = torch.mean(dist_min).numpy()
    return mhd

# naive msd metric with low memery footprint optimization (forward msd)
def cal_msd_lowmem_init_version(mesh_pd, mesh_gt):
    """
    注意： 此函数由于只考虑单向forward的msd, 不合理, 已被遗弃, 为legacy
    """
    mesh_pd_pts = np.array(mesh_pd.vertices)
    mesh_gt_pts = np.array(mesh_gt.vertices)
    pd_pts_tensor = torch.FloatTensor(mesh_pd_pts)[:,np.newaxis,:].cuda()
    gt_pts_tensor = torch.FloatTensor(mesh_gt_pts)[np.newaxis,:,:].cuda()
    dist_min = torch.zeros(size=[pd_pts_tensor.shape[0]])
    pd_pts_num = pd_pts_tensor.shape[0]
    step_size = 1000
    step_num = int(np.ceil(pd_pts_num/step_size))
    for step_idx in tqdm(range(0, step_num+10)):
        pd_idx_start = step_idx*step_size
        pd_idx_end = min( (step_idx+1)*step_size,  pd_pts_num)
        if(pd_idx_start>=pd_pts_num):
            break
        pd_pts_single = pd_pts_tensor[pd_idx_start:pd_idx_end, :, :]
        dist_matrix_single = torch.sqrt(torch.sum((pd_pts_single-gt_pts_tensor)**2, dim=2))
        dist_min_single = torch.min(dist_matrix_single, dim=1).values
        dist_min[pd_idx_start:pd_idx_end] = dist_min_single
    mhd = torch.mean(dist_min).cpu().numpy()
    return mhd

# msd calculation function, 以tensor为输入, 单向forward计算msd, 作为双向symmetric计算msd的基础，用于training的metric记录
def metric_msd_lowmem_tensor_core(
    src_pts_tensor,
    dst_pts_tensor,
    scale = 1.0,
    block_size = 4096,
    debug = False
):
    src_block_num = math.ceil(src_pts_tensor.shape[0]/block_size)
    dst_block_num = math.ceil(dst_pts_tensor.shape[0]/block_size)
    # init min vector
    dist_min_vector = torch.ones(size=[src_pts_tensor.shape[0]])*100000000
    # traverse src_pts by block
    for src_block_idx in range(0, src_block_num):
        src_start_idx = src_block_idx * block_size
        src_end_idx = (src_block_idx + 1) * block_size
        block_src_pts_tensor = src_pts_tensor[src_start_idx:src_end_idx, :]
        slice_min_vector = None
        # traverse dst_pts by block
        for dst_block_idx in range(0, dst_block_num):
            dst_start_idx = dst_block_idx * block_size
            dst_end_idx = (dst_block_idx + 1) * block_size
            block_dst_pts_tensor = dst_pts_tensor[dst_start_idx:dst_end_idx, :]
            # calculate block dist matrix
            block_dist_matrix = torch.sqrt(
                    torch.sum((block_src_pts_tensor[:,np.newaxis,:] - block_dst_pts_tensor[np.newaxis,:,:])**2, dim=2)
                )
            block_min_vector = torch.min(block_dist_matrix, dim=1).values
            # save min value from block_min_vector into slice_min_vector
            if(slice_min_vector is None):
                slice_min_vector = block_min_vector
            else:
                slice_min_vector = torch.min(slice_min_vector, block_min_vector)
            
            if(debug):
                print(f"handling src_start_idx:{src_start_idx} src_end_idx:{src_end_idx} dst_start_idx:{dst_start_idx} dst_end_idx:{dst_end_idx} block_src_pts_tensor:{block_src_pts_tensor.shape} block_dst_pts_tensor:{block_dst_pts_tensor.shape} block_dist_matrix:{block_dist_matrix.shape} block_min_vector:{block_min_vector.shape} slice_min_vector:{slice_min_vector.shape}\n")
        # integrate slice_min_vector into dist_min_vector
        dist_min_vector[src_start_idx:src_end_idx] = slice_min_vector
    
    # calculate msd
    msd = torch.mean(dist_min_vector).cpu().numpy()*scale
    return msd

# msd calculation function, 以tensor为输入, 双向forward计算msd, 用于training的metric记录
def metric_msd_lowmem_tensor_symmetric(
    src_pcd_tensor,
    dst_pcd_tensor,
    scale = 1.0,
    block_size = 4096,
    debug = False
):
    """
    Input:
        src_pcd_tensor, Tensor, shape [B, C, N]
        dst_pcd_tenosr, Tensor, shape [B, C, N]
    Return:
        Average Msd between Batches, taken physical scale `scale` parameter into account.
    """
    
    batch_size = src_pcd_tensor.shape[0]
    # prm_src_pcd_tensor shape [B, N, C]
    prm_src_pcd_tensor = src_pcd_tensor.permute(0, 2, 1)[0, :, :]
    # prm_dst_pcd_tensor shape [B, N, C]
    prm_dst_pcd_tensor = dst_pcd_tensor.permute(0, 2, 1)[0, :, :]
    
    metric_msd_list = []
    for batch_idx in range(0, batch_size):
        # forward msd
        cur_fwd_metric_msd = metric_msd_lowmem_tensor_core(
            src_pts_tensor=prm_src_pcd_tensor,
            dst_pts_tensor=prm_dst_pcd_tensor,
            scale = scale,
            block_size = block_size,
            debug = debug
        )
        # backward msd
        cur_bkd_metric_msd = metric_msd_lowmem_tensor_core(
            src_pts_tensor=prm_dst_pcd_tensor,
            dst_pts_tensor=prm_src_pcd_tensor,
            scale = scale,
            block_size = block_size,
            debug = debug
        )
        cur_metric_msd = (cur_fwd_metric_msd + cur_bkd_metric_msd)/2
        metric_msd_list.append(cur_metric_msd)
    
    metric_msd = np.mean(metric_msd_list)
    return metric_msd


# 标准的msd单向forward msd的计算函数, 作为双向symmetric msd计算函数的基础
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

# 标准的msd单向backward msd的计算函数, 作为双向symmetric msd计算函数的基础
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

# 标准的msd双向symmetric msd的计算函数, 用于evaluation
def cal_msd_symmetric(
    mesh_a,
    mesh_b,
    scale=1.0,
    print_msg_flag=True
):
    """
    标准双向MSD计算函数, 当前evaluation中MSD metric计算的首选
    注意: 需要设定每单位长度对应的物理尺寸, 也即函数的scale参数
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

# 标准的msd单向forward msd的计算函数, 作为双向symmetric msd的计算函数,
# 加入了low memory footprint的优化和cuda的优化, 应对大量顶点的情况
def cal_msd_forward_lowmem_general(
    mesh_src,
    mesh_dst,
    scale=1.0,
    block_size=2000,
    print_msg_flag=True,
    use_cuda=False,
    debug=False
):
    if(print_msg_flag):
        print("Calculate Msd Forward")
    
    # src pts
    mesh_src_pts = np.array(mesh_src.vertices)
    src_pts_tensor = torch.FloatTensor(mesh_src_pts)
    if(use_cuda):
        src_pts_tensor = src_pts_tensor.cuda()
    src_block_num = math.ceil(src_pts_tensor.shape[0]/block_size)
    
    # dst pts
    mesh_dst_pts = np.array(mesh_dst.vertices)
    dst_pts_tensor = torch.FloatTensor(mesh_dst_pts)
    if(use_cuda):
        dst_pts_tensor = dst_pts_tensor.cuda()
    dst_block_num = math.ceil(dst_pts_tensor.shape[0]/block_size)
    
    # init min vector
    dist_min_vector = torch.ones(size=[src_pts_tensor.shape[0]])*100000000
    # traverse src_pts by block
    for src_block_idx in range(0, src_block_num):
        src_start_idx = src_block_idx * block_size
        src_end_idx = (src_block_idx + 1) * block_size
        block_src_pts_tensor = src_pts_tensor[src_start_idx:src_end_idx, :]
        slice_min_vector = None
        # traverse dst_pts by block
        for dst_block_idx in range(0, dst_block_num):
            dst_start_idx = dst_block_idx * block_size
            dst_end_idx = (dst_block_idx + 1) * block_size
            block_dst_pts_tensor = dst_pts_tensor[dst_start_idx:dst_end_idx, :]
            # calculate block dist matrix
            block_dist_matrix = torch.sqrt(
                    torch.sum((block_src_pts_tensor[:,np.newaxis,:] - block_dst_pts_tensor[np.newaxis,:,:])**2, dim=2)
                )
            block_min_vector = torch.min(block_dist_matrix, dim=1).values
            # save min value from block_min_vector into slice_min_vector
            if(slice_min_vector is None):
                slice_min_vector = block_min_vector
            else:
                slice_min_vector = torch.min(slice_min_vector, block_min_vector)
            if(debug):
                print(f"handling src_start_idx:{src_start_idx} src_end_idx:{src_end_idx} dst_start_idx:{dst_start_idx} dst_end_idx:{dst_end_idx} block_src_pts_tensor:{block_src_pts_tensor.shape} block_dst_pts_tensor:{block_dst_pts_tensor.shape} block_dist_matrix:{block_dist_matrix.shape} block_min_vector:{block_min_vector.shape} slice_min_vector:{slice_min_vector.shape}\n")
        # integrate slice_min_vector into dist_min_vector
        dist_min_vector[src_start_idx:src_end_idx] = slice_min_vector
    
    # calculate msd
    msd = torch.mean(dist_min_vector).cpu().numpy()*scale
    return msd

# 标准的msd单向backward msd的计算函数, 作为双向symmetric msd的计算函数,
# 加入了low memory footprint的优化和cuda的优化, 应对大量顶点的情况
def cal_msd_backward_lowmem_general(
    mesh_src,
    mesh_dst,
    scale=1.0,
    block_size=2000,
    print_msg_flag=True,
    use_cuda=False,
    debug=False
):  
    if(print_msg_flag):
        print("Calculate Msd Backward")
        
    cur_mesh_src = mesh_src
    cur_mesh_dst = mesh_dst
    msd = cal_msd_forward_lowmem_general(
        mesh_src=cur_mesh_dst,
        mesh_dst=cur_mesh_src,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        use_cuda=use_cuda,
        debug=debug
    )
    return msd

# 标准的msd双向symmetric msd的计算函数, 用于evaluation
# 加入了low memory footprint的优化和cuda的优化, 应对大量顶点的情况
def cal_msd_symmetric_lowmem_general(
    mesh_a,
    mesh_b,
    scale=1.0,
    block_size=2000,
    print_msg_flag=True,
    use_cuda=False,
    debug=False
):
    """
    标准双向MSD计算函数, 当前evaluation中MSD metric计算的首选; 加入了low memory footprint的优化和cuda的优化, 应对大量顶点的情况
    注意: 需要设定每单位长度对应的物理尺寸, 也即函数的scale参数
    """
    if(print_msg_flag):
        print("Calculate Msd Symmetric")
    # calculate msd forward
    msd_forward = cal_msd_forward_lowmem_general(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        use_cuda=use_cuda,
        debug=debug
    )
    # calculate msd backward
    msd_backward = cal_msd_backward_lowmem_general(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        use_cuda=use_cuda,
        debug=debug
    )
    # calculate msd
    msd_symmetric = (msd_forward + msd_backward)/2
    return msd_symmetric

def get_pv_mesh_forward_msd_vec_arrayND(src_mesh,
                                    ref_mesh,
                                    scale=1.0):
    # src_mesh and ref_mesh are pv mesh object
    src_mesh_pts = src_mesh.points
    ref_mesh_pts = ref_mesh.points
    dis_matrix = distance.cdist(src_mesh_pts, ref_mesh_pts, metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)] = 100000
    msd_vector = np.min(dis_matrix, axis=1)*scale
    return msd_vector

def get_pv_mesh_symmetric_msd_byregismesh_arrayND(src_mesh,
                                                  ref_mesh,
                                                  regis_src_mesh,
                                                  scale=1.0,
                                                  dist_matrix_nan_value=100000):
    src_mesh_pts = get_triangle_mesh_vertices_pv(src_mesh)
    dst_mesh_pts = get_triangle_mesh_vertices_pv(ref_mesh)
    regis_mesh_pts = get_triangle_mesh_vertices_pv(regis_src_mesh)
    # init
    input_src_mesh = src_mesh
    input_ref_mesh = ref_mesh
    # forward msd vec
    fwd_msd_vec = get_pv_mesh_forward_msd_vec_arrayND(src_mesh=input_src_mesh, ref_mesh=input_ref_mesh, scale=scale)
    # backward msd vec
    bkd_msd_vec = get_pv_mesh_forward_msd_vec_arrayND(src_mesh=input_ref_mesh, ref_mesh=input_src_mesh, scale=scale)
    # find correspondence
    dist_matrix = get_dist_matrix_arrayND(src_pts=regis_mesh_pts, dst_pts=dst_mesh_pts, set_nan_value=dist_matrix_nan_value)
    corres_idxs = np.argmin(dist_matrix, axis=1)
    corres_bkd_msd_vec = bkd_msd_vec[corres_idxs]
    # print(f"test shape fwd_msd_vec:{fwd_msd_vec.shape} bkd_msd_vec:{bkd_msd_vec.shape} dist_matrix:{dist_matrix.shape} corres_idxs:{corres_idxs.shape} corres_bkd_msd_vec:{corres_bkd_msd_vec.shape}")
    # symmetric msd vec
    sym_msd_vec = (fwd_msd_vec + corres_bkd_msd_vec)/2
    return sym_msd_vec

def get_pv_mesh_corres_msd_byregismesh_arrayND(src_mesh,
                                                  ref_mesh,
                                                  regis_src_mesh,
                                                  scale=1.0,
                                                  dist_matrix_nan_value=100000):
    """
    Calculate Msd based on the correspondence found between regis_src_mesh and ref_mesh
    """
    src_mesh_pts = get_triangle_mesh_vertices_pv(src_mesh)
    dst_mesh_pts = get_triangle_mesh_vertices_pv(ref_mesh)
    regis_mesh_pts = get_triangle_mesh_vertices_pv(regis_src_mesh)
    # init
    input_src_mesh = src_mesh
    input_ref_mesh = ref_mesh
    # find correspondence
    dist_matrix = get_dist_matrix_arrayND(src_pts=regis_mesh_pts, dst_pts=dst_mesh_pts, set_nan_value=dist_matrix_nan_value)
    corres_idxs = np.argmin(dist_matrix, axis=1)
    # calculate correspond distance
    corres_dst_mesh_pts = dst_mesh_pts[corres_idxs, :]
    # print(f"test shape src_mesh_pts.shape:{src_mesh_pts.shape} dst_mesh_pts.shape:{dst_mesh_pts.shape}")
    # print(f"test shape corres_idxs.shape:{corres_idxs.shape} corres_dst_mesh_pts.shape:{corres_dst_mesh_pts.shape}")
    corres_dist = np.sqrt(np.sum((src_mesh_pts - corres_dst_mesh_pts)**2, axis=1))
    # print(f"test shape corres_dist.shape:{corres_dist.shape}")
    corres_msd_vec = corres_dist*scale
    return corres_msd_vec

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

# pv mesh msd single-forward calculate function 
def get_pv_mesh_msd_vector(src_mesh, ref_mesh, scale=1.0):
    # src_mesh and ref_mesh are pv mesh object
    src_mesh_pts = src_mesh.points
    ref_mesh_pts = ref_mesh.points
    dis_matrix = distance.cdist(src_mesh_pts, ref_mesh_pts,metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)] = 100000
    msd_vector = np.min(dis_matrix, axis=1)*scale
    return msd_vector

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

# mesh visualize

# common visualization setting (below are from inpainting)
# Individual Mesh Visualize
St_Pv_Indiv_Mesh_Draw_Color = "#f1f1f1"
St_Pv_Background_Draw_Color = [1.0, 1.0, 1.0]

# Gt and Pd Mesh Overlap
# Gt Mesh
St_Pv_Overlap_Gt_Mesh_Color = "#BABEBE"
St_Pv_Overlap_Gt_Mesh_Ambient = 0.2
St_Pv_Overlap_Gt_Mesh_Metallic = 0.1
St_Pv_Overlap_Gt_Mesh_Opacity = 0.5
St_Pv_Overlap_Gt_Mesh_ShowEdge = False
St_Pv_Overlap_Gt_Mesh_EdgeColor = "#01B3FB"
St_Pv_Overlap_Gt_Mesh_Silhouette = False
# Pd Mesh
St_Pv_Overlap_Pd_Mesh_Color = "#7CFC00"
St_Pv_Overlap_Pd_Mesh_Ambient = 0.1
St_Pv_Overlap_Pd_Mesh_Metallic = 0.8
St_Pv_Overlap_Pd_Mesh_Opacity = 0.4
St_Pv_Overlap_Pd_Mesh_ShowEdge = False
St_Pv_Overlap_Pd_mesh_EdgeColor = "#B3FB01"
St_Pv_Overlap_Pd_Mesh_Silhouette = False

# Maxi mesh draw color
St_Pv_Maxi_Mesh_Color = "#cdcec9"
St_Pv_Maxi_Mesh_Color_Light = "#f2f2f0"
# Cleft mesh draw color
St_Pv_Cleft_Mesh_Color = "#7cfc00"
# Third mesh draw color for 3 mesh overlaping
St_Pv_Third_Mesh_Color = "#00BBFC"
St_Pv_Overlap_Third_Mesh_Color = "#00BBFC"
# other overlap common color
St_Pv_Overlap_Mesh_Color_Red = "#d9480f"
St_Pv_Overlap_Mesh_Color_Blue = "#5acdfa"
St_Pv_Overlap_Mesh_Color_Yellow = "#ebf538"
St_Pv_Overlap_Mesh_Color_Green = "#49e012"
St_Pv_Overlap_Mesh_Color_Black = "#000000"
St_Pv_Overlap_Mesh_Color_DeepBlue = "#5654c7"
St_Pv_Overlap_Mesh_Color_LightRed = "#d46b20"

# draw skeleton line common color
St_Pv_Skeleton_Line_Mesh_Color_Blue = "#60a1e6"
St_Pv_Skeleton_Line_Mesh_Color_Red = "#db6148"
# spline sample num
St_Pv_Skeleton_Line_Tube_SampleNum = 100
# default size
St_Pv_Skeleton_Line_Line_Width = 8
St_Pv_Skeleton_Line_Tube_Radius = 0.64
St_Pv_Skeleton_Line_Sphere_Radius = 1.12
# opacity size
St_Pv_Skeleton_Line_Tube_Opacity = 1.0
St_Pv_Skeleton_Line_Shpere_Opacity = 1.0

# 图像切片绘制参数
# draw image slice default param
St_Pv_Norm_Image_Slice_Crop_Size = 200
St_Pd_Cleft_Slice_Color = ""

# Msd 绘制参数
# Msd Draw Config Default Set
St_Mesh_Msd_Scale = 0.5
St_Mesh_Msd_Clim = [0, 3]
St_Mesh_Msd_Cmap = "CET_L18"

# Polyscope 绘制参数
# Polyscope render color
St_Ps_Maxi_Mesh_Color = "#CEE5E2"
St_Ps_Cleft_Mesh_Color = "#519D05"
# Polyscope render opacity
St_Ps_Maxi_Mesh_Opacity = 0.43
St_Ps_Cleft_Mesh_Opacity = 1.0
# draw skeleton line common color
St_Ps_Skeleton_Line_Mesh_Color_Blue = "#4000FF"
St_Ps_Skeleton_Line_Mesh_Color_Red = "#FF0500"

# Mesh Drawer
class MeshDrawer:
    def __init__(self,
                 camera_position_path,
                 show_windows=False,
                 window_size=[512,512],
                 light_system_type="preset_1",
                 ):
        """
        available light system: defualt, preset_1.
        
            `default` mode is used to draw overlap mesh.    
            
            `preset_1` mode is used to draw individual mesh. 
        """
        import pyvista as pv
        self.camera_position_path = camera_position_path
        self.show_windows = show_windows
        self.window_size = window_size
        # plotter setting
        self.plotter_window_size = [self.window_size[1], self.window_size[0]]
        # set camera position
        make_parent_dir(self.camera_position_path)
        # depth peeling config
        self.set_depth_peeling_flag = False
        self.set_depth_peeling_number_of_peels = 10
        # anti aliasing config
        self.set_anti_aliasing_flag = False
        self.set_anti_aliasing_method = "fxaa"
        
        # camera position button
        self.show_camera_position_save_button_flag = False
        self.camera_position_save_dir = None
        self.camera_position_save_perfix = None
        self.camera_position_save_idx = -1
        self.camera_position_save_button_start_pos = None
        self.camera_position_save_button_size = None
        self.camera_position_save_print_info_flag = False
        
        
        
        # initialize light system
        self.light_system_type = light_system_type
        default_plotter = pv.Plotter()
        default_light_list = default_plotter.renderer.lights
        # default
        if(self.light_system_type == "default"):
            self.light_list = default_light_list
        # preset_1
        if(self.light_system_type == "preset_1"):
            preset_1_light_list = copy.copy(default_light_list)
            for preset_light in preset_1_light_list:
                preset_light.intensity -= 0.13
                preset_light.intensity = np.clip(preset_light.intensity, 0, 100)
            downside_light = pv.Light(position=(0, -10, 0), focal_point=[0, 10, 0], intensity=0.4, light_type='camera light')
            frontside_light = pv.Light(position=(0, 0, 10), focal_point=[0, 0, -10], intensity=0.3, light_type='camera light')
            leftside_light = pv.Light(position=(-10, 0, 0), focal_point=[10, 0, 0], intensity=0.2, light_type='camera light')
            # rightside_light = pv.Light(position=(10, 0, 0), focal_point=[-10, 0, 0], intensity=0.1, light_type='camera light')
            preset_1_light_list.append(downside_light)
            preset_1_light_list.append(frontside_light)
            preset_1_light_list.append(leftside_light)
            # preset_1_light_list.append(rightside_light)
            self.light_list = preset_1_light_list
        
        # draw mesh by stages
        self.draw_stage_mesh_list = []
        self.draw_stage_mesh_color_list = []
        self.draw_stage_kwargsdict_list = []
    
    def enable_depth_peeling(
        self, number_of_peels=10
    ):
        self.set_depth_peeling_flag = True
        self.set_depth_peeling_number_of_peels = number_of_peels
    
    def enable_anti_aliasing(
        self, anti_aliasing_method = "fxaa"
    ):
        self.set_anti_aliasing_flag = True
        self.set_anti_aliasing_method = anti_aliasing_method
    
    def init_pv_plotter(self):
        self.plotter = pv.Plotter(lighting="none",
                                  off_screen=(not self.show_windows), 
                                  window_size=self.plotter_window_size)
        # configure light system
        for preset_light in self.light_list:
            self.plotter.add_light(preset_light)
        # config depth peeling
        if(self.set_depth_peeling_flag):
            self.plotter.enable_depth_peeling(number_of_peels=self.set_depth_peeling_number_of_peels)
        # config anti aliasing
        if(self.set_anti_aliasing_flag):
            self.plotter.enable_anti_aliasing(aa_type=self.set_anti_aliasing_method)
        
        return self.plotter
    
    def load_camera_position(self, camera_position_path):
        if(os.path.exists(camera_position_path)):
            self.plotter.camera_position = common_json_load(camera_position_path)

    def save_camera_position(self, camera_postion_path, print_camera_flag=False):
        common_json_dump(list(self.plotter.camera_position), camera_postion_path)
        if(print_camera_flag):
            print(self.plotter.camera_position)
    
    def get_cur_camera_position(self):
        return list(self.plotter.camera_position)
    
    def del_camera_position(self, camera_position_path):
        os.remove(camera_position_path)
        return True
    
    # set camera position functions
    def activate_camera_positon_save_ability(self):
        self.show_camera_position_save_button_flag = True
        return True

    def deactivate_camera_position_save_ability(self):
        self.show_camera_position_save_button_flag = False
    
    def activate_show_windows_ability(self):
        self.show_windows = True
    
    def deactivate_show_windows_ability(self):
        self.show_windows = False
        
    def set_camera_position_save_param(self,
                                       camera_position_save_dir,
                                       camera_position_save_perfix="default",
                                       camera_position_save_button_start_pos=(10, 10),
                                       camera_position_save_button_size=50,
                                       camera_position_save_print_info_flag=False):
        """
        notice:
            `blue button` is for camera position save.
            
            `red button` is for camera position delete.
        """
        self.camera_position_save_dir = camera_position_save_dir
        self.camera_position_save_perfix = camera_position_save_perfix
        self.camera_position_save_button_start_pos = camera_position_save_button_start_pos
        self.camera_position_save_button_size = camera_position_save_button_size
        self.camera_position_save_print_info_flag = camera_position_save_print_info_flag
    
    def get_cur_camera_position_save_path(self):
        return f"{self.camera_position_save_dir}/{self.camera_position_save_perfix}_{self.camera_position_save_idx}.json"
    
    def get_prev_camera_position_save_path(self, save_idx):
        return f"{self.camera_position_save_dir}/{self.camera_position_save_perfix}_{save_idx}.json"
    
    def draw_mesh(self,
                  pv_mesh,
                  background_color=[1.0, 1.0, 1.0],
                  mesh_color=None,
                  save_screenshot_path=False,
                  cam_pos_path=None,
                  auto_load_pos_flag=True,
                  **kwargs
                ):
        import pyvista as pv
        
        # init plotter
        self.plotter = self.init_pv_plotter()
        
        # camera position save func
        def save_camera_position_callback(button_flag):
            if(button_flag==True):
                self.camera_position_save_idx += 1
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.save_camera_position(camera_postion_path=cur_camera_position_save_path)
                    print(f"saved camera position at {cur_camera_position_save_path}")
                    if(self.camera_position_save_print_info_flag):
                        print(f"saved camera position:{self.get_cur_camera_position()}")
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
                return True
            return False
        
        def del_camera_position_callback(button_flag):
            if(button_flag==True):
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.camera_position_save_idx -= 1
                    self.del_camera_position(cur_camera_position_save_path)
                    print(f"delete camera position at {cur_camera_position_save_path}")
                    return True
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
            return False
                        
        if(self.show_camera_position_save_button_flag):
            # save call back
            save_button_start_pos = [
                self.camera_position_save_button_start_pos[0],
                self.camera_position_save_button_start_pos[1] + self.camera_position_save_button_size*11//10
            ]
            self.plotter.add_checkbox_button_widget(
                callback=save_camera_position_callback,
                value=False,
                color_on="blue",
                color_off="grey",
                position=save_button_start_pos,
                size=self.camera_position_save_button_size
            )
            # del call back
            del_button_start_pos = self.camera_position_save_button_start_pos
            self.plotter.add_checkbox_button_widget(
                callback=del_camera_position_callback,
                value=False,
                color_on="red",
                color_off="grey",
                position=del_button_start_pos,
                size=self.camera_position_save_button_size
            )
        
        if(cam_pos_path is not None):
            self.load_camera_position(cam_pos_path)
        if(auto_load_pos_flag):
            self.load_camera_position(self.camera_position_path)
        self.plotter.set_background(color=background_color)
        self.plotter.add_mesh(
            pv_mesh,
            color=mesh_color,
            **kwargs
        )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        if(auto_load_pos_flag and self.show_windows):
            self.save_camera_position(self.camera_position_path)
    
    def draw_mesh_list(self,
                        pv_mesh_list,
                        mesh_color_list=None,
                        background_color=[1.0, 1.0, 1.0],
                        save_screenshot_path=False,
                        **kwargs):
        import pyvista as pv
        
        # init plotter
        self.plotter = self.init_pv_plotter()
        
        self.load_camera_position(self.camera_position_path)
        self.plotter.set_background(color=background_color)
        for pv_mesh, mesh_color in zip(pv_mesh_list, mesh_color_list):
            self.plotter.add_mesh(
                pv_mesh,
                color=mesh_color,
                **kwargs
            )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        self.save_camera_position(self.camera_position_path)
    
    def add_mesh_stage(self,
                    pv_mesh,
                    mesh_color,
                    **kwargs
                ):
        self.draw_stage_mesh_list.append(pv_mesh)
        self.draw_stage_mesh_color_list.append(mesh_color)
        self.draw_stage_kwargsdict_list.append(kwargs)
    
    def clean_mesh_stage(self):
        self.draw_stage_mesh_list = []
        self.draw_stage_mesh_color_list = []
        self.draw_stage_kwargsdict_list = []
    
    def draw_mesh_stage(self,
                        background_color=[1.0, 1.0, 1.0],
                        save_screenshot_path=False,
                        auto_load_pos_flag=True
                        ):
        
        import pyvista as pv
        # init plotter
        self.plotter = self.init_pv_plotter()
        
        # camera position save func
        def save_camera_position_callback(button_flag):
            if(button_flag==True):
                self.camera_position_save_idx += 1
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.save_camera_position(camera_postion_path=cur_camera_position_save_path)
                    print(f"saved camera position at {cur_camera_position_save_path}")
                    if(self.camera_position_save_print_info_flag):
                        print(f"saved camera position:{self.get_cur_camera_position()}")
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
                return True
            return False
        
        def del_camera_position_callback(button_flag):
            if(button_flag==True):
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.camera_position_save_idx -= 1
                    self.del_camera_position(cur_camera_position_save_path)
                    print(f"delete camera position at {cur_camera_position_save_path}")
                    return True
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
            return False
                        
        if(self.show_camera_position_save_button_flag):
            # save call back
            save_button_start_pos = [
                self.camera_position_save_button_start_pos[0],
                self.camera_position_save_button_start_pos[1] + self.camera_position_save_button_size*11//10
            ]
            self.plotter.add_checkbox_button_widget(
                callback=save_camera_position_callback,
                value=False,
                color_on="blue",
                color_off="grey",
                position=save_button_start_pos,
                size=self.camera_position_save_button_size
            )
            # del call back
            del_button_start_pos = self.camera_position_save_button_start_pos
            self.plotter.add_checkbox_button_widget(
                callback=del_camera_position_callback,
                value=False,
                color_on="red",
                color_off="grey",
                position=del_button_start_pos,
                size=self.camera_position_save_button_size
            )
        
        # load camera position
        if(auto_load_pos_flag):
            self.load_camera_position(self.camera_position_path)
        # draw mesh
        self.plotter.set_background(color=background_color)
        for pv_mesh, mesh_color, kwargs_dict in zip(self.draw_stage_mesh_list, self.draw_stage_mesh_color_list, \
            self.draw_stage_kwargsdict_list):
            if("save_screenshot_path" in kwargs_dict):
                kwargs_dict.remove("save_screenshot_path")
            self.plotter.add_mesh(
                pv_mesh,
                color=mesh_color,
                **kwargs_dict
            )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        # save camera position
        if(auto_load_pos_flag and self.show_windows):
            self.save_camera_position(self.camera_position_path)

# Mesh Drawer Polyscope
class MeshDrawer_Polyscope:
    def __init__(self):
        # draw mesh by stages
        self.mesh_list = []
        self.color_list = []
        self.opacity_list = []
        self.translation_list = []
        self.smooth_flag_list = []

    def add_mesh_stage(
        self,
        o3d_mesh,
        o3d_mesh_color,
        o3d_mesh_opacity,
        o3d_mesh_translation=None,
        ps_smooth_flag=False
    ):
        self.mesh_list.append(o3d_mesh)
        self.color_list.append(o3d_mesh_color)
        self.opacity_list.append(o3d_mesh_opacity)
        self.translation_list.append(o3d_mesh_translation)
        self.smooth_flag_list.append(ps_smooth_flag)
    
    def draw_mesh_stage(
        self,
        draw_image_path=None,
        show_windows_flag=None
    ):
        import polyscope as ps
        ps.init()
        ps.set_transparency_mode('pretty')
        ps.set_ground_plane_mode("none")
        ps.set_navigation_style("free")
        
        for mesh_idx, (cur_mesh, cur_color, cur_opacity, cur_translation, cur_smooth_flag) in \
            (zip(self.mesh_list, self.color_list, self.opacity_list, self.translation_list, self.smooth_flag_list)):
            if(cur_translation is not None):
                cur_mesh = translate_triangle_mesh_o3d(mesh=cur_mesh, translation=cur_translation)
            # register mesh
            cur_mesh_verts = get_triangle_mesh_vertices(cur_mesh)
            cur_mesh_faces = get_triangle_mesh_faces(cur_mesh)
            ps_cur_mesh = ps.register_surface_mesh(
                                            name=f"mesh_{mesh_idx}",
                                            vertices=cur_mesh_verts,
                                            faces=cur_mesh_faces, 
                                            transparency=cur_opacity,
                                            smooth_shade=cur_smooth_flag)
            # add color
            background_color_float_list = get_color_rgb_float_from_hex(cur_color)
            ps_color_background = get_color_Npoint_array_rgb_from_float(
                rgb_float_list=background_color_float_list,
                point_num=cur_mesh_verts.shape[0]
            )
            ps_cur_mesh.add_color_quantity(name="verts_color",values=ps_color_background, defined_on='vertices', enabled=True)
            mesh_idx += 1
    
        if(show_windows_flag is not None):
            ps.show()
        if(draw_image_path is not None):
            ps.screenshot(draw_image_path, transparent_bg=False)
