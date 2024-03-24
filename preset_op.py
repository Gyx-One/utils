"""
本脚本收纳的函数是根据已有的函数, 封装成的预设的绘制等API, 用于在实验中以统一的参数调用已有的函数
例如已统一的参数对Cleft进行绘制
"""

from .dependencies import *
from .mesh_op import *
from .image_op import *
from .volume_op import *

# basic gen mesh function
def basic_preset_get_mesh_from_vol_cleft(cube,
                                         keep_component_num=2):
    mesh = get_mesh_from_vol_mcube(
        cube=cube,
        filter_smooth_taubin_flag=True,
        filter_smooth_taubin_iter=50,
        filter_smooth_simple_flag=True,
        filter_smooth_simple_iter=2,
        filter_connected_component_flag=True,
        filter_num_triangle_thresh=500,
        keep_component_flag=True,
        keep_component_num=keep_component_num   
    )
    return mesh

def basic_preset_get_mesh_from_vol_cleft_from_path(cube_path,
                                                   mesh_path,
                                                   keep_component_num=2):
    get_mesh_from_vol_mcube_from_path(
        cube_path=cube_path,
        mesh_path=mesh_path,
        filter_smooth_taubin_flag=True,
        filter_smooth_taubin_iter=50,
        filter_smooth_simple_flag=True,
        filter_smooth_simple_iter=2,
        filter_connected_component_flag=True,
        filter_num_triangle_thresh=500,
        keep_component_flag=True,
        keep_component_num=keep_component_num
    )

# basic draw mesh functions
def basic_draw_mesh_func(
    mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    mesh_color=St_Pv_Indiv_Mesh_Draw_Color,
    window_size=[512, 512],
    show_windows_flag=False
):
    """
    notes:
        Draw Single Mesh
    """
    
    pv_mesh = read_triangle_mesh_pv(mesh_path)
    mesh_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="preset_1"
    )
    mesh_drawer.draw_mesh(
        pv_mesh=pv_mesh,
        background_color=St_Pv_Background_Draw_Color,
        mesh_color=mesh_color,
        save_screenshot_path=draw_image_path
    )
    return True

def basic_draw_transparent_mesh_func(
    mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    mesh_color=St_Pv_Indiv_Mesh_Draw_Color,
    mesh_ambient = 0.01,
    mesh_metallic = 0.8,
    mesh_opacity = 0.4,
    mesh_silhouette = False,
    window_size=[512, 512],
    show_windows_flag=False
):
    """
    notes:
        Draw Single Mesh
    """
    
    pv_mesh = read_triangle_mesh_pv(mesh_path)
    mesh_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="preset_1"
    )
    mesh_drawer.draw_mesh(
        pv_mesh=pv_mesh,
        background_color=St_Pv_Background_Draw_Color,
        mesh_color=mesh_color,
        ambient = mesh_ambient,
        metallic = mesh_metallic,
        opacity = mesh_opacity,
        silhouette = mesh_silhouette,
        save_screenshot_path=draw_image_path
    )
    return True

def basic_draw_overlap_mesh_func(
    background_mesh_path,
    foreground_mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    background_color=St_Pv_Overlap_Gt_Mesh_Color,
    foreground_color=St_Pv_Overlap_Pd_Mesh_Color,
    background_opacity=0.4,
    foreground_opacity=0.6,
    window_size=[512, 512],
    show_windows_flag=False,
    ret_drawer_flag=False
):
    """
    notes:
        Basic function for drawing foreground_mesh overlap background_mesh
    """
    
    maxi_mesh_default_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="default"
    )
    
    pv_background_mesh = read_triangle_mesh_pv(background_mesh_path)
    pv_foreground_mesh = read_triangle_mesh_pv(foreground_mesh_path)
    maxi_mesh_default_drawer.clean_mesh_stage()
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_background_mesh,
        mesh_color=background_color,
        ambient = 0.01,
        metallic = 0.8,
        opacity = background_opacity,
        silhouette = False
    )
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_foreground_mesh,
        mesh_color=foreground_color,
        opacity = foreground_opacity
    )
    if(ret_drawer_flag):
        return maxi_mesh_default_drawer
    else:
        maxi_mesh_default_drawer.draw_mesh_stage(
            save_screenshot_path=draw_image_path
        )
        return True

def basic_draw_overlap_mesh_func_addtrans_depthpeel(
    background_mesh_path,
    foreground_mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    background_color=St_Pv_Overlap_Gt_Mesh_Color,
    foreground_color=St_Pv_Overlap_Pd_Mesh_Color,
    background_opacity=0.55,
    foreground_opacity=1.0,
    window_size=[512, 512],
    trans_background_list=None,
    trans_foreground_list=None,
    show_windows_flag=False,
    ret_drawer_flag=False
):
    """
    Draw background mesh and foreground mesh overlap
    1. Use Depth peeling for better transparent mesh overlapping
    
    """
    
    maxi_mesh_default_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="default"
    )
    
    maxi_mesh_default_drawer.enable_depth_peeling(10)
    pv_background_mesh = read_triangle_mesh_pv(background_mesh_path)
    pv_foreground_mesh = read_triangle_mesh_pv(foreground_mesh_path)
    
    # translate background mesh for better visualize
    if(trans_background_list is not None):
        pv_background_mesh = pv_background_mesh.translate(trans_background_list)
    # translate foregroud  mesh for better visualize
    if(trans_foreground_list is not None):
        pv_foreground_mesh = pv_foreground_mesh.translate(trans_foreground_list)
    
    maxi_mesh_default_drawer.clean_mesh_stage()
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_background_mesh,
        mesh_color=background_color,
        opacity=background_opacity,
        silhouette=False
    )
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_foreground_mesh,
        mesh_color=foreground_color,
        opacity=foreground_opacity,
        silhouette=False
    )
    
    if(ret_drawer_flag):
        return maxi_mesh_default_drawer
    else:
        ret_value = maxi_mesh_default_drawer.draw_mesh_stage(
            save_screenshot_path=draw_image_path
        )
        return ret_value

def basic_draw_overlap_mesh_func_double_overlap(
    background_mesh_path,
    foreground_mesh_path,
    third_mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    background_color=St_Pv_Overlap_Gt_Mesh_Color,
    foreground_color=St_Pv_Overlap_Pd_Mesh_Color,
    third_color=St_Pv_Overlap_Pd_Mesh_Color,
    window_size=[512, 512],
    show_windows_flag=False
):
    """
    notes:
        Baisc function for drawing 2 foreground_mesh overlap background_mesh
    """
    
    maxi_mesh_default_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="default"
    )
    
    pv_background_mesh = read_triangle_mesh_pv(background_mesh_path)
    pv_foreground_mesh = read_triangle_mesh_pv(foreground_mesh_path)
    pv_third_mesh = read_triangle_mesh_pv(third_mesh_path)
    
    maxi_mesh_default_drawer.clean_mesh_stage()
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_background_mesh,
        mesh_color=background_color,
        ambient=0.01,
        metallic=0.8,
        opacity=0.4,
        silhouette=False
    )
    
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_foreground_mesh,
        mesh_color=foreground_color,
        opacity=0.55
    )
    
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=pv_third_mesh,
        mesh_color=third_color,
        opacity=0.6
    )
    
    maxi_mesh_default_drawer.draw_mesh_stage(
        save_screenshot_path=draw_image_path
    )
    return True

def basic_draw_overlap_mesh_func_from_mesh_list(
    background_mesh,
    foreground_mesh_list,
    mesh_cam_pos_path,
    draw_image_path,
    background_color=St_Pv_Overlap_Gt_Mesh_Color,
    background_ambient=0.01,
    background_metallic=0.8,
    background_opacity=0.4,
    foreground_color_list=[],
    foreground_ambient_list=[],
    foreground_metallic_list=[],
    foreground_opacity_list=[],
    default_foreground_color=St_Pv_Overlap_Pd_Mesh_Color,
    default_foreground_ambient=None,
    default_foreground_metallic=None,
    default_foreground_opacity=0.5,
    window_size=[512, 512],
    show_windows_flag=False
):
    """
    notes:
        Baisc function for drawing a list of foreground_mesh overlap background_mesh
    """
    
    def local_add_dict_filter_none(attr_dict, attr_key, attr_value):
        if(attr_value is not None):
            attr_dict[attr_key] = attr_value
        return attr_dict
    
    maxi_mesh_default_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="default"
    )
    
    maxi_mesh_default_drawer.clean_mesh_stage()
    maxi_mesh_default_drawer.add_mesh_stage(
        pv_mesh=background_mesh,
        mesh_color=background_color,
        ambient=background_ambient,
        metallic=background_metallic,
        opacity=background_opacity
    )
    
    for f_idx, foreground_mesh in enumerate(foreground_mesh_list):
        draw_dict = {}
        local_add_dict_filter_none(draw_dict, "pv_mesh", foreground_mesh)
        # color
        if(f_idx<len(foreground_color_list)):
            cur_color = foreground_color_list[f_idx]
        else:
            cur_color = default_foreground_color
        local_add_dict_filter_none(draw_dict, "mesh_color", cur_color)
        # ambient
        if(f_idx<len(foreground_ambient_list)):
            cur_ambient = foreground_ambient_list[f_idx]
        else:
            cur_ambient = default_foreground_ambient
        local_add_dict_filter_none(draw_dict, "ambient", cur_ambient)
        # metallic
        if(f_idx<len(foreground_metallic_list)):
            cur_metallic = foreground_metallic_list[f_idx]
        else:
            cur_metallic = default_foreground_metallic
        local_add_dict_filter_none(draw_dict, "metallic", cur_metallic)
        # opacity
        if(f_idx<len(foreground_opacity_list)):
            cur_opacity = foreground_opacity_list[f_idx]
        else:
            cur_opacity = default_foreground_opacity
        local_add_dict_filter_none(draw_dict, "opacity", cur_opacity)
        # drwa
        maxi_mesh_default_drawer.add_mesh_stage(
            **draw_dict
        )
        
    maxi_mesh_default_drawer.draw_mesh_stage(
        save_screenshot_path=draw_image_path
    )
    return True

def basic_draw_overlap_mesh_func_polyscope(
    background_mesh_path,
    foreground_mesh_path,
    draw_image_path=None,
    background_color=St_Ps_Cleft_Mesh_Color,
    foreground_color=St_Ps_Maxi_Mesh_Color,
    background_opacity=St_Ps_Cleft_Mesh_Opacity,
    foreground_opacity=St_Ps_Maxi_Mesh_Opacity,
    background_translation_list=None,
    foreground_translation_list=None,
    background_smooth_flag=False,
    foreground_smooth_flag=False,
    show_windows_flag=False
):
    """
    Notes:
        Background_color: Hex Color String 
        Foreground_color: Hex Color String
    
    """
    import polyscope as ps
    ps.init()
    ps.set_transparency_mode('pretty')
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")
    
    # background mesh
    mesh_background = read_triangle_mesh(background_mesh_path)
    if(background_translation_list is not None):
        mesh_background = translate_triangle_mesh_o3d(mesh=mesh_background, translation=background_translation_list)
    mesh_background_verts = get_triangle_mesh_vertices(mesh_background)
    mesh_background_faces = get_triangle_mesh_faces(mesh_background)
    ps_mesh_background = ps.register_surface_mesh(
                                    name="mesh_background",
                                    vertices=mesh_background_verts,
                                    faces=mesh_background_faces, 
                                    transparency=background_opacity,
                                    smooth_shade=background_smooth_flag)
    # add color
    background_color_float_list = get_color_rgb_float_from_hex(background_color)
    ps_color_background = get_color_Npoint_array_rgb_from_float(
        rgb_float_list=background_color_float_list,
        point_num=mesh_background_verts.shape[0]
    )
    ps_mesh_background.add_color_quantity(name="verts_color",values=ps_color_background, defined_on='vertices', enabled=True)
    
    # foreground mesh
    mesh_foreground = read_triangle_mesh(foreground_mesh_path)
    if(foreground_translation_list is not None):
        mesh_foreground = translate_triangle_mesh_o3d(mesh=mesh_foreground, translation=foreground_translation_list)
    mesh_foreground_verts = get_triangle_mesh_vertices(mesh_foreground)
    mesh_foreground_faces = get_triangle_mesh_faces(mesh_foreground)
    ps_mesh_foreground = ps.register_surface_mesh(
                                    name="mesh_foreground",
                                    vertices=mesh_foreground_verts,
                                    faces=mesh_foreground_faces, 
                                    transparency=foreground_opacity,
                                    smooth_shade=foreground_smooth_flag)
    # add color
    foreground_color_float_list = get_color_rgb_float_from_hex(foreground_color)
    ps_color_foreground = get_color_Npoint_array_rgb_from_float(
        rgb_float_list=foreground_color_float_list,
        point_num=mesh_foreground_verts.shape[0]
    )
    ps_mesh_foreground.add_color_quantity(name="verts_color",values=ps_color_foreground, defined_on='vertices', enabled=True)
    
    if(show_windows_flag):
        ps.show()
    elif(draw_image_path is not None):
        ps.screenshot(draw_image_path, transparent_bg=False)


def basic_draw_mesh_func_polyscope(
    mesh_path,
    draw_image_path=None,
    mesh_color=St_Ps_Cleft_Mesh_Color,
    mesh_opacity=St_Ps_Cleft_Mesh_Opacity,
    mesh_smooth_flag=False,
    show_windows_flag=False
):
    """
    Notes:
        Background_color: Hex Color String 
        Foreground_color: Hex Color String
    
    """
    import polyscope as ps
    ps.init()
    ps.set_transparency_mode('pretty')
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")
    
    # mesh
    mesh = read_triangle_mesh(mesh_path)
    mesh_verts = get_triangle_mesh_vertices(mesh)
    mesh_faces = get_triangle_mesh_faces(mesh)
    ps_mesh = ps.register_surface_mesh(
                                    name="mesh",
                                    vertices=mesh_verts,
                                    faces=mesh_faces, 
                                    transparency=mesh_opacity,
                                    smooth_shade=mesh_smooth_flag)
    # add color
    color_float_list = get_color_rgb_float_from_hex(mesh_color)
    ps_color = get_color_Npoint_array_rgb_from_float(
        rgb_float_list=color_float_list,
        point_num=mesh_verts.shape[0]
    )
    ps_mesh.add_color_quantity(name="verts_color",values=ps_color, defined_on='vertices', enabled=True)
    
    if(show_windows_flag):
        ps.show()
    elif(draw_image_path is not None):
        ps.screenshot(draw_image_path, transparent_bg=False)

# basic preset draw mesh func 
def basic_preset_draw_pd_gt_cleft_overlap_func(
    gt_mesh_path,
    pd_mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    window_size=[512, 512],
    show_windows_flag=False,
    ret_drawer_flag=False
):
    """
    专用于绘制 预测裂隙 和 真值裂隙 重叠的函数
    起源于2023年3月份绘制 MICCAI 和 ICCV 的对比试验Mesh绘制图
    """
    # Gt and Pd Mesh Overlap
    # Gt Mesh
    local_Pv_Overlap_Gt_Mesh_Color = "#BABEBE"
    local_Pv_Overlap_Gt_Mesh_Ambient = 0.2
    local_Pv_Overlap_Gt_Mesh_Metallic = 0.1
    local_Pv_Overlap_Gt_Mesh_Opacity = 0.5
    local_Pv_Overlap_Gt_Mesh_ShowEdge = False
    local_Pv_Overlap_Gt_Mesh_EdgeColor = "#01B3FB"
    local_Pv_Overlap_Gt_Mesh_Silhouette = False
    # Pd Mesh
    local_Pv_Overlap_Pd_Mesh_Color = "#7CFC00"
    local_Pv_Overlap_Pd_Mesh_Ambient = 0.1
    local_Pv_Overlap_Pd_Mesh_Metallic = 0.8
    local_Pv_Overlap_Pd_Mesh_Opacity = 0.4
    local_Pv_Overlap_Pd_Mesh_ShowEdge = False
    local_Pv_Overlap_Pd_mesh_EdgeColor = "#B3FB01"
    local_Pv_Overlap_Pd_Mesh_Silhouette = False
    mesh_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        show_windows=show_windows_flag,
        window_size=window_size,
        light_system_type="default"
    )
    mesh_drawer.clean_mesh_stage()
    pv_gt_cleft_mesh = read_triangle_mesh_pv(gt_mesh_path)
    pv_pd_cleft_mesh = read_triangle_mesh_pv(pd_mesh_path)
    mesh_drawer.add_mesh_stage(
        pv_mesh = pv_gt_cleft_mesh,
        mesh_color = local_Pv_Overlap_Gt_Mesh_Color,
        ambient = local_Pv_Overlap_Gt_Mesh_Ambient,
        metallic = local_Pv_Overlap_Gt_Mesh_Metallic,
        opacity = local_Pv_Overlap_Gt_Mesh_Opacity,
        show_edges = local_Pv_Overlap_Gt_Mesh_ShowEdge,
        edge_color = local_Pv_Overlap_Gt_Mesh_EdgeColor,
        silhouette = local_Pv_Overlap_Gt_Mesh_Silhouette
    )
    mesh_drawer.add_mesh_stage(
        pv_mesh = pv_pd_cleft_mesh,
        mesh_color = local_Pv_Overlap_Pd_Mesh_Color,
        ambient = local_Pv_Overlap_Pd_Mesh_Ambient,
        metallic = local_Pv_Overlap_Pd_Mesh_Metallic,
        opacity = local_Pv_Overlap_Pd_Mesh_Opacity,
        show_edges = local_Pv_Overlap_Pd_Mesh_ShowEdge,
        edge_color = local_Pv_Overlap_Pd_mesh_EdgeColor,
        silhouette = local_Pv_Overlap_Pd_Mesh_Silhouette
    )
    mesh_drawer.draw_mesh_stage(
        save_screenshot_path=draw_image_path
    )
    if(ret_drawer_flag):
        return mesh_drawer

def basic_preset_draw_pd_gt_cleft_regis_sym_msd_func(
    gt_mesh_path,
    pd_mesh_path,
    regis_mesh_path,
    mesh_cam_pos_path,
    draw_image_path,
    use_cuda_flag=True,
    window_size=[512, 512],
    msd_clim=[0, 3],
    show_windows_flag=False,
    ret_drawer_flag=False,
    show_scalar_bar_flag=False
):
    """
    专用于绘制 预测裂隙 和 真值裂隙 MSD的函数
    本函数用regis_prob函数将 预测裂隙 配准到 仿真裂隙, 生成 regis_mesh 配准Mesh
    将配准Mesh和真值Mesh之间根据最近邻计算对应, 根据计算的对应计算 预测裂隙 和 真值裂隙 之间的Msd
    (因为配准Mesh的点和预测裂隙的点是一一对应的)
    
    起源于2023年3月份绘制 MICCAI 和 ICCV 的 单双侧/仿真-临床 填补效果Mesh绘制图
    
    注意: 本函数绘制Msd时, 对Msd进行了物理距离的scale, scale大小为0.5;
          同时对Msd进行了clip, clip 范围为 [0, 3] (scale后)
    """
    local_Mesh_Msd_Scale = 0.5
    local_Mesh_Msd_Cmap = "CET_L18"
    local_Mesh_Msd_Clim = msd_clim
    if(not os.path.exists(regis_mesh_path)):
        cpd_mesh_nonrigid_registration_probreg_by_path(src_mesh_path=pd_mesh_path,
                                                        dst_mesh_path=gt_mesh_path,
                                                        regis_mesh_path=regis_mesh_path,
                                                        use_cuda=use_cuda_flag)
    mesh_drawer = MeshDrawer(
        camera_position_path=mesh_cam_pos_path,
        window_size=window_size,
        show_windows=show_windows_flag
    )
    
    pv_gt_mesh = read_triangle_mesh_pv(gt_mesh_path)
    pv_pd_mesh = read_triangle_mesh_pv(pd_mesh_path)
    pv_regis_mesh = read_triangle_mesh_pv(regis_mesh_path)
    
    pd_sym_msd_vec = get_pv_mesh_symmetric_msd_byregismesh_arrayND(
            src_mesh = pv_pd_mesh,
            ref_mesh = pv_gt_mesh,
            regis_src_mesh = pv_regis_mesh,
            scale = local_Mesh_Msd_Scale
    )
    pv_pd_mesh.point_data["sym_msd"] = pd_sym_msd_vec
    mesh_drawer.draw_mesh(
        pv_mesh = pv_pd_mesh,
        background_color = [1.0, 1.0, 1.0],
        scalars = "sym_msd",
        cmap = local_Mesh_Msd_Cmap,
        clim = local_Mesh_Msd_Clim,
        show_scalar_bar = show_scalar_bar_flag,
        scalar_bar_args = {"title" : "", "vertical" : True, "color" : "black"},
        save_screenshot_path = draw_image_path
    )
    if(ret_drawer_flag):
        return mesh_drawer


# basic preset draw gt-contour pd-region label slice overlap func

def basic_preset_render_contour_slice_func(func_image_slice_input,
                            func_gt_label_slice_input,
                            func_pd_label_slice_input,
                            pd_alpha = 0.5,
                            gt_alpha = 1.0,
                            pd_color_vec = [0, 1, 0],
                            gt_color_vec = [0/255, 69/255, 255/255],
                            render_gt_label_contour_flag=True,
                            thickness = 1):
    
    render_overlap_slice = copy.copy(func_image_slice_input)
    
    # draw pd label
    render_overlap_slice = blend_image_with_color_mask(
        image = render_overlap_slice,
        mask = func_pd_label_slice_input,
        blend_color = pd_color_vec,
        alpha = pd_alpha
    )
    
    # draw gt label contour
    if(render_gt_label_contour_flag):
        gt_label_slice_contour_mask = get_contours_mask_binary_image_float(
            binary_image=func_gt_label_slice_input,
            thickness=thickness
        )
        render_overlap_slice = blend_image_with_color_mask(
            image = render_overlap_slice,
            mask = gt_label_slice_contour_mask,
            blend_color = gt_color_vec,
            alpha = gt_alpha
        )
    return render_overlap_slice

def basic_preset_draw_crop_slice_label_overlap_func(
    image_origin,
    gt_label,
    pd_label,
    slice_type,
    slice_idx,
    gt_color="#ff0800",
    pd_color="#00ff00",
    pd_render_alpha=0.5,
    gt_render_alpha=1.0,
    slice_crop_size=180
):
    """
    给定原始图像-origin_image, 真值label图像-gt_label, 预测label图像-pd_label
    
    根据gt_label裁剪出裂隙图像区域, 将真值画成轮廓, 将预测画成区域, 绘制预测区域和真值区域在原始图像上的重叠
    
    1. 自动根据gt_label计算裁剪的中心点
    2. 裁剪出crop_origin_image, crop_gt_label, crop_pd_label
    3. 绘制重叠slice图像返回
    
    注意: gt_color和pd_color输入格式为16进制字符串
    """
    
    
    gt_color_vec = get_color_bgr_float_from_hex(gt_color)
    pd_color_vec = get_color_bgr_float_from_hex(pd_color)
    image2D_center_h, image2D_center_w = get_2D_image_crop_center_from_3D_binary_image(
        binary_3D_mask=gt_label,
        slice_type=slice_type
    )
    
    image_slice = get_slice_image_from_cube_array3D(cube=image_origin, slice_idx=slice_idx, slice_type=slice_type, cvt_color_flag=True)
    gt_label_slice = get_slice_image_from_cube_array3D(cube=gt_label, slice_idx=slice_idx, slice_type=slice_type)
    pd_label_slice = get_slice_image_from_cube_array3D(cube=pd_label, slice_idx=slice_idx, slice_type=slice_type)
    crop_image_slice = get_crop_image2D_by_center(
                input_image=image_slice, center_h=image2D_center_h, center_w=image2D_center_w,
                range_h=slice_crop_size, range_w=slice_crop_size, crop_type="fit"
    )
    crop_gt_label_slice = get_crop_image2D_by_center(
                input_image=gt_label_slice, center_h=image2D_center_h, center_w=image2D_center_w,
                range_h=slice_crop_size, range_w=slice_crop_size, crop_type="fit"
    )
    crop_pd_label_slice = get_crop_image2D_by_center(
                input_image=pd_label_slice, center_h=image2D_center_h, center_w=image2D_center_w,
                range_h=slice_crop_size, range_w=slice_crop_size, crop_type="fit"
    )
    crop_render_slice = basic_preset_render_contour_slice_func(
        func_image_slice_input = crop_image_slice,
        func_gt_label_slice_input = crop_gt_label_slice,
        func_pd_label_slice_input = crop_pd_label_slice,
        pd_alpha = pd_render_alpha,
        gt_alpha = gt_render_alpha,
        pd_color_vec = pd_color_vec,
        gt_color_vec = gt_color_vec
    )
    return crop_render_slice

# preset get mesh skeleton line 
def basic_preset_get_mesh_skeleton_line(
    point_list,
    sample_num=100,
    cvt_tube_flag=True,
    tube_radius=St_Pv_Skeleton_Line_Tube_Radius,
    sphere_radius=St_Pv_Skeleton_Line_Sphere_Radius
):
    """
    Input:
        point_list: List of Points, each point is a 3-number vec [x,y,z]
    
    Output:
        3 `Pyvista` Mesh
        
        `pv_mesh_line` is a tube mesh for skeleton trace
        
        `pv_mesh_src_pt` is a sphere mesh for start point of skeleton
        
        `pv_mesh_dst_pt` is a sphere mesh for dst point of skeleton
    """
    src_pt = point_list[0]
    dst_pt = point_list[-1]
    pv_mesh_line = get_pv_mesh_spline(point_list=point_list, sample_num=sample_num,
                                cvt_tube_flag=cvt_tube_flag, cvt_tube_radius=tube_radius)
    pv_mesh_src_pt = get_pv_mesh_sphere(center=src_pt, radius=sphere_radius)
    pv_mesh_dst_pt = get_pv_mesh_sphere(center=dst_pt, radius=sphere_radius)
    
    return pv_mesh_line, pv_mesh_src_pt, pv_mesh_dst_pt