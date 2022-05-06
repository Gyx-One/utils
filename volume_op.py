import shutil
from utils.common import get_name, get_suffix, make_parent_dir
from utils.tensor_op import get_inv_rigid_matrix
from .dependencies import *
import cc3d
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import measure

# volume io operations

# read and write image
def read_mha_image3D(image_path):
    return itk.ReadImage(image_path)

def write_mha_image3D(image, image_path):
    return itk.WriteImage(image, image_path)

# read and write dicoms
def read_dicom_image(dicom_dir, ret_image=False):
    series_ids = itk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    series_file_names = itk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir,series_ids[0])
    series_reader = itk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    cube=itk.GetArrayFromImage(image3D)
    if(ret_image):
        return  cube, image3D
    return cube

# read and write arrays
def read_mha_array3D(image_path, norm=False, ret_param_dict=False):
    image = read_mha_image3D(image_path)
    image_array = itk.GetArrayFromImage(image).astype(np.float32)
    if(norm):
        image_array=image_array/2500
    if(ret_param_dict):
        param_dict = {}
        param_dict["Origin"] = image.GetOrigin()
        param_dict["Spacing"] = image.GetSpacing()
        param_dict["Direction"] = image.GetDirection()
        return image_array, param_dict
    
    return image_array

def read_mha_array4D(image_path,norm=False):
    return read_mha_array3D(image_path=image_path,norm=norm)[np.newaxis,:,:,:]

def write_mha_array3D(array,
                      image_path,
                      Origin=None,
                      Spacing=None,
                      Direction=None,
                      param_dict=None,
                      norm=False):
    if(norm):
        array=array*2500
    image = itk.GetImageFromArray(array)
    if(param_dict is not None):
        if("Origin" in param_dict):
            Origin = param_dict["Origin"]
        if("Spacing" in param_dict):
            Spacing = param_dict["Spacing"]
        if("Direction" in param_dict):
            Direction = param_dict["Direction"]
        
    if(Origin is not None):
        image.SetOrigin(Origin)
    if(Spacing is not None):
        image.SetSpacing(Spacing)
    if(Direction is not None):
        image.SetDirection(Direction)
    write_mha_image3D(image, image_path)

def write_mha_array4D(array,image_path,norm=False):
    write_mha_array3D(array=array[0],image_path=image_path,norm=norm)

# read and write tensors
def read_mha_tensor3D(image_path,norm=False):
    return torch.FloatTensor(read_mha_array3D(image_path=image_path,norm=norm))

def read_mha_tensor4D(image_path,norm=False):
    return torch.FloatTensor(read_mha_array4D(image_path=image_path,norm=norm))

def write_mha_tensor3D(tensor,image_path,norm=False):
    array=tensor.detach().cpu().numpy()
    write_mha_array3D(array,image_path,norm)

def write_mha_tensor4D(tensor,image_path,norm=False):
    array=tensor.detach().cpu().numpy()
    write_mha_array4D(array,image_path,norm)

# volumne generation


# connected-component operations
def get_component_3D(cube,keep_num=1,connectivity=2,only_cube=False):
    cube=cube.astype(np.int32)
    label_seg, label_num = measure.label(input=cube, return_num=True, connectivity=connectivity)
    label_count = np.bincount(label_seg.flatten())
    keep_labels = label_count.argsort()[::-1][1:keep_num+1]
    keep_components = [label_seg == label_idx for label_idx in keep_labels]
    if(only_cube):
        return keep_components
    return list(zip(keep_components, keep_labels))

def keep_component_3D(cube,keep_num=1,connectivity=2):
    cube_value=np.max(cube)
    keep_cube=np.zeros_like(cube)
    keep_list=get_component_3D(cube,keep_num,connectivity=connectivity)
    for com,vol in keep_list:
        keep_cube=np.logical_or(keep_cube,com)
    return np.where(keep_cube,cube_value,0).astype(np.int32)

def get_component_2D(slice,keep_num=100000,only_slice=False):
    com_list=[]
    labels_out,N=cc3d.connected_components(slice, connectivity=8,return_N=True)
    for n in range(1,N+1):
        com=(labels_out==n)
        vol=np.sum(com)
        com_list.append([com,vol])
    keep_list=sorted(com_list,key=lambda x:x[1],reverse=True)
    keep_list=keep_list[:keep_num]
    if(only_slice):
        keep_list=[com for com,vol in keep_list]
    return keep_list

def keep_component_2D(slice,keep_num=1):
    slice_value=np.max(slice)
    keep_slice=np.zeros_like(slice)
    keep_list=get_component_2D(slice,keep_num)
    for com,vol in keep_list:
        keep_slice=np.logical_or(keep_slice,com)
    return np.where(keep_slice,slice_value,0).astype(np.int32)

# morphology operations
def open_operation(bin_cube):
    from scipy.ndimage import morphology
    errode_cube=morphology.binary_erosion(bin_cube,iterations=1)
    dilation_cube=morphology.binary_dilation(errode_cube,iterations=1)
    open_bin_cube=np.where(dilation_cube,1,0).astype(np.int32)
    return open_bin_cube

def close_operation(bin_cube):
    from scipy.ndimage import morphology
    dilation_cube=morphology.binary_dilation(bin_cube,iterations=1)
    errode_cube=morphology.binary_erosion(dilation_cube,iterations=1)
    close_bin_cube=np.where(errode_cube,1,0).astype(np.int32)
    return close_bin_cube

def clean_segments(segment):
    value_volumns=[]
    values=np.unique(segment)
    for value in values:
        value_volumn=np.where(segment==value,segment,0)
        value_volumn=keep_component_3D(value_volumn,keep_num=1)
        value_volumns.append(value_volumn)
    clean_segment=np.zeros_like(segment)
    for value_volumn in value_volumns:
        clean_segment=np.where(value_volumn!=0,value_volumn,clean_segment)
    return clean_segment

# zoom operations
def zoom_to_shape(x,dst_shape):
    from scipy.ndimage import zoom
    x_shape=x.shape
    zoom_x=zoom(x,zoom=[dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]])
    return zoom_x

def zoom_to_shape_binary(x,dst_shape):
    from scipy.ndimage import zoom
    x_value=np.round(np.max(x))
    x_shape=x.shape
    zoom_rate = [dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]]
    zoom_x = zoom_binary(cube=x, zoom_rate=zoom_rate).astype(np.float32)
    return zoom_x

def zoom_binary(cube,zoom_rate):
    from scipy.ndimage import zoom
    x_value = np.round(np.max(cube))
    x_shape = cube.shape
    dst_shape = [round(x_shape[0]*zoom_rate[0]), round(x_shape[1]*zoom_rate[1]), round(x_shape[2]*zoom_rate[2])]
    tensor_cube = torch.FloatTensor(cube).unsqueeze(0).unsqueeze(0)
    zoom_tensor_cube = F.interpolate(input=tensor_cube, size=dst_shape, mode="nearest",).squeeze()
    zoom_cube = zoom_tensor_cube.numpy()
    return zoom_cube.astype(np.float32)

def zoom_segments(x,zoom_rate):
    value_volumns=[]
    values=np.unique(x)
    for value in values:
        part_origin_seg=np.where(x==value,value,0)
        part_zoom_seg=zoom_binary(part_origin_seg,zoom_rate=zoom_rate)
        value_volumns.append(part_zoom_seg)
    zoom_seg=np.zeros_like(value_volumns[0])
    for value_volumn in value_volumns:
        zoom_seg=np.where(value_volumn!=0,value_volumn,zoom_seg)
    return zoom_seg.astype(np.int32)

def zoom_segments_to_shape(x,dst_shape):
    x_shape=x.shape
    zoom_rate = [dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]]
    return zoom_binary(x, zoom_rate=zoom_rate)

def plasti_zoom_segments_to_shape(src_path, dst_path, dst_shape, temp_dir="./temp_dir"):
    src_suffix = get_suffix(src_path)
    src_name = get_name(src_path)
    dst_suffix = get_suffix(dst_path)
    dst_name = get_name(dst_path)
    cvt_flag = False
    
    # path
    os.makedirs(temp_dir, exist_ok=True)
    make_parent_dir(dst_path)
    zoom_src_path = src_path
    zoom_dst_path = dst_path
    
    if((src_suffix != "mha") or (dst_suffix != "mha")):
        cvt_flag = True
        zoom_src_path = f"{temp_dir}/{src_name}.mha"
        zoom_dst_path = f"{temp_dir}/{dst_name}.mha"
        src_cube, src_param_dict = read_mha_array3D(src_path, ret_param_dict=True)
        write_mha_array3D(src_cube, zoom_src_path, param_dict=src_param_dict)
    
    zoom_src_path = os.path.abspath(zoom_src_path)
    zoom_dst_path = os.path.abspath(zoom_dst_path)
    cmd = f"plastimatch resample --input {zoom_src_path} --output {zoom_dst_path} --interpolation=nn --dim=\"{dst_shape[0]} {dst_shape[1]} {dst_shape[2]}\""
    os.system(cmd)
    
    if(cvt_flag):
        zoom_dst_cube, zoom_dst_param_dict = read_mha_array3D(zoom_dst_path, ret_param_dict=True)
        write_mha_array3D(zoom_dst_cube, dst_path, param_dict=zoom_dst_param_dict)
        
    os.system(f"rm -r {temp_dir}")
    

# boundary determination operations
def detect_top_array3D(cube,thresh=0.01):
    h,w,l=cube.shape
    top_slice_h=h-1
    for slice_h in range(h-1,0,-1):
        if(np.sum(cube[slice_h,:,:])>=thresh):
            top_slice_h=slice_h
            break
    return top_slice_h

def detect_bottom_array3D(cube,thresh=0.01):
    h,w,l=cube.shape
    bottom_slice_h=0
    for slice_h in range(0,h):
        if(np.sum(cube[slice_h,:,:])>=thresh):
            bottom_slice_h=slice_h
            break
    return bottom_slice_h

# bouding box operations
def get_mask_bbx3D_array(label_cube,margin=4):
    #get bounding box of the binary mask
    #input [h,w,l] tensor
    mask=np.zeros_like(label_cube)
    if(np.sum(label_cube)!=0):
        img_h,img_w,img_l=label_cube.shape
        hs,ws,ls=np.where(label_cube>0)
        mask_hmin=np.clip(np.min(hs)-margin,0,img_h)
        mask_hmax=np.clip(np.max(hs)+margin,0,img_h)
        mask_wmin=np.clip(np.min(ws)-margin,0,img_w)
        mask_wmax=np.clip(np.max(ws)+margin,0,img_w)
        mask_lmin=np.clip(np.min(ls)-margin,0,img_l)
        mask_lmax=np.clip(np.max(ls)+margin,0,img_l)
        mask[mask_hmin:mask_hmax,mask_wmin:mask_wmax,mask_lmin:mask_lmax]=1
    mask=mask.astype(np.float32)
    return mask

def get_mask_bbx4D_array(bin_mask,pad_h=5,pad_w=5,pad_l=5):
    #get bounding box of the binary mask
    #input [c,h,w,l] tensor
    mask=np.zeros_like(bin_mask)
    if(np.sum(bin_mask>0)!=0):
        _,h,w,l=bin_mask.shape
        _,hs,ws,ls=np.where(bin_mask>0)
        min_h,max_h=np.min(hs),np.max(hs)
        min_w,max_w=np.min(ws),np.max(ws)
        min_l,max_l=np.min(ls),np.max(ls)
        min_h=np.clip(min_h-pad_h,0,h)
        max_h=np.clip(max_h+pad_h,0,h)
        min_w=np.clip(min_w-pad_w,0,w)
        max_w=np.clip(max_w+pad_w,0,w)
        min_l=np.clip(min_l-pad_l,0,l)
        max_l=np.clip(max_l+pad_l,0,l)
        mask[:,min_h:max_h,min_w:max_w,min_l:max_l]=1
    return mask

def thresh_histgram(input_image,std_image,thresh=500/2500):
    from skimage.exposure import match_histograms
    high_std_image_idxs=std_image>=thresh
    low_std_image_idxs=std_image<thresh
    high_input_image_idxs=input_image>=thresh
    low_input_image_idxs=input_image<thresh
    output_image=np.zeros_like(input_image)
    output_image[high_input_image_idxs]=match_histograms(image=input_image[high_input_image_idxs],\
        reference=std_image[high_std_image_idxs])
    output_image[low_input_image_idxs]=input_image[low_input_image_idxs]
    return output_image

def crop_and_pad(image,dst_shape,pad_value=0):
    # crop and pad shape to dst shape for 3D image
    h,w,l=image.shape
    dst_h,dst_w,dst_l=dst_shape
    pad_image=np.ones(shape=[dst_h,dst_w,dst_l])*pad_value
    crop_h,crop_w,crop_l=min(h,dst_h),min(w,dst_w),min(l,dst_l)
    pad_image[:crop_h,:crop_w,:crop_l]=image[:crop_h,:crop_w,:crop_l]
    return pad_image

def crop_and_pad_consist(image_list,dst_shape,pad_value=0):
    # crop and pad shape to dst shape for 3D image
    h,w,l = image_list[0].shape
    dst_h, dst_w, dst_l = dst_shape
    # determine crop_h, crop_w, crop_l
    crop_h, crop_w, crop_l = dst_h, dst_w, dst_l
    for image in image_list:
        image_h, image_w, image_l = image.shape
        crop_h, crop_w, crop_l = min(image_h, crop_h), min(image_w, crop_w), min(image_l, crop_l)
    # crop images
    pad_image_list = []
    for image in image_list:
        pad_image=np.ones(shape=[dst_h,dst_w,dst_l])*pad_value
        pad_image[:crop_h,:crop_w,:crop_l]=image[:crop_h,:crop_w,:crop_l]
        pad_image_list.append(pad_image)
    return pad_image_list

def spatial_transform_by_param_array3D(image,
                                       trans_x=0,
                                       trans_y=0,
                                       trans_z=0,
                                       roll=0,
                                       pitch=0,
                                       yaw=0,
                                       center_x=0,
                                       center_y=0,
                                       center_z=0,
                                       mode="bilinear",
                                       padding_mode="zeros",
                                       ret_matrix=False,
                                       debug=False):
    from .tensor_op import get_rigid_matrix_from_param, get_deform_from_rigid_matrix, spatial_transform
    print(f"test spatial_transform trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}")
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    rigid_matrix = get_rigid_matrix_from_param(trans_x, trans_y, trans_z, roll, pitch, yaw, center_x, center_y, center_z)
    if(debug):
        print(f"[Debug] Spatial transform by param rigid_matrix:\n{rigid_matrix}")
    inv_rigid_matrix = get_inv_rigid_matrix(rigid_matrix)
    deform = get_deform_from_rigid_matrix(inv_rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    if(ret_matrix):
        return trans_image, rigid_matrix
    return trans_image

def spatial_transform_by_param_ZYX_array3D(image,
                                           trans_x=0,
                                           trans_y=0,
                                           trans_z=0,
                                           roll=0,
                                           pitch=0,
                                           yaw=0,
                                           center_x=0,
                                           center_y=0,
                                           center_z=0,
                                           mode="bilinear",
                                           padding_mode="zeros"):
    from .tensor_op import get_rigid_matrix_from_param_ZYX, get_deform_from_rigid_matrix, spatial_transform
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    rigid_matrix = get_rigid_matrix_from_param_ZYX(trans_x, trans_y, trans_z, roll, pitch, yaw, center_x, center_y, center_z)
    inv_rigid_matrix = get_inv_rigid_matrix(rigid_matrix)
    deform = get_deform_from_rigid_matrix(inv_rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    return trans_image

def spatial_transform_by_matrix_array3D(image,
                                        rigid_matrix,
                                        mode="bilinear",
                                        padding_mode="zeros",
                                        debug=False):
    from .tensor_op import get_inv_rigid_matrix, get_deform_from_rigid_matrix, spatial_transform
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    if(debug):
        print(f"[Debug] Spatial transform by matrix rigid_matrix:\n{rigid_matrix}")
    inv_rigid_matrix = get_inv_rigid_matrix(rigid_matrix)
    deform = get_deform_from_rigid_matrix(inv_rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    return trans_image

def Legacy_spatial_transform_by_param_array3D(image,
                                       trans_x=0,
                                       trans_y=0,
                                       trans_z=0,
                                       roll=0,
                                       pitch=0,
                                       yaw=0,
                                       center_x=0,
                                       center_y=0,
                                       center_z=0,
                                       mode="bilinear",
                                       padding_mode="zeros",
                                       ret_matrix=False):
    from .tensor_op import get_rigid_matrix_from_param, get_deform_from_rigid_matrix, spatial_transform
    # print(f"test spatial_transform trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}")
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    trans_x, trans_y, trans_z = -trans_x, -trans_y, -trans_z
    roll, pitch, yaw = -roll, -pitch, -yaw
    rigid_matrix = get_rigid_matrix_from_param(trans_x, trans_y, trans_z, roll, pitch, yaw, center_x, center_y, center_z)
    # print(f"test spatial_transform matrix:\n{rigid_matrix}\n")
    deform = get_deform_from_rigid_matrix(rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    if(ret_matrix):
        return trans_image, rigid_matrix
    return trans_image

def Legacy_spatial_transform_by_param_ZYX_array3D(image,
                                           trans_x=0,
                                           trans_y=0,
                                           trans_z=0,
                                           roll=0,
                                           pitch=0,
                                           yaw=0,
                                           center_x=0,
                                           center_y=0,
                                           center_z=0,
                                           mode="bilinear",
                                           padding_mode="zeros"):
    from .tensor_op import get_rigid_matrix_from_param_ZYX, get_deform_from_rigid_matrix, spatial_transform
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    rigid_matrix = get_rigid_matrix_from_param_ZYX(trans_x, trans_y, trans_z, roll, pitch, yaw, center_x, center_y, center_z)
    deform = get_deform_from_rigid_matrix(rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    return trans_image

def Legacy_spatial_transform_by_matrix_array3D(image,
                                               rigid_matrix,
                                               mode="bilinear",
                                               padding_mode="zeros"):
    from .tensor_op import get_inv_rigid_matrix, get_deform_from_rigid_matrix, spatial_transform
    inv_rigid_matrix = get_inv_rigid_matrix(rigid_matrix)
    tensor_image = torch.FloatTensor(image).unsqueeze(0)
    deform = get_deform_from_rigid_matrix(inv_rigid_matrix, shape=image.shape)
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode)
    trans_image = trans_tensor_image[0].numpy()
    return trans_image

# metrics oprations
def cal_dice(pd_label, gt_label):
    cal_gt_label=np.where(gt_label!=0,1,0)
    cal_pd_label=np.where(pd_label!=0,1,0)
    intersec=np.logical_and(cal_gt_label,cal_pd_label)
    dice=2*np.sum(intersec)/(np.sum(cal_gt_label)+np.sum(cal_pd_label))
    return dice

def cal_dice_clip(pd_label, 
                  gt_label, 
                  margin_max_h=0, 
                  margin_min_h=0, 
                  margin_max_w=0, 
                  margin_min_w=0,
                  margin_max_l=0,
                  margin_min_l=0):
    cal_gt_label=np.where(gt_label!=0,1,0)
    cal_pd_label=np.where(pd_label!=0,1,0)
    gt_hs, gt_ws, gt_ls = np.where(cal_gt_label>0)
    # determin clip index
    clip_max_h = np.max(gt_hs) - margin_max_h
    clip_min_h = np.min(gt_hs) + margin_min_h
    clip_max_w = np.max(gt_ws) - margin_max_w
    clip_min_w = np.min(gt_ws) + margin_min_w
    clip_max_l = np.max(gt_ls) - margin_max_l
    clip_min_l = np.min(gt_ls) + margin_min_l
    clip_min_h = min(clip_min_h, clip_max_h-1)
    clip_min_w = min(clip_min_w, clip_max_w-1)
    clip_min_l = min(clip_min_l, clip_max_l-1)
    # print(f"test clip min_h:{clip_min_h} max_h:{clip_max_h} min_w:{clip_min_w} max_w:{clip_max_w} min_l:{clip_min_l} max_l:{clip_max_l}")
    # clip 
    cal_gt_label = cal_gt_label[clip_min_h:clip_max_h, clip_min_w:clip_max_w, clip_min_l:clip_max_l]
    cal_pd_label = cal_pd_label[clip_min_h:clip_max_h, clip_min_w:clip_max_w, clip_min_l:clip_max_l]
    intersec=np.logical_and(cal_gt_label,cal_pd_label)
    dice=2*np.sum(intersec)/(np.sum(cal_gt_label)+np.sum(cal_pd_label))
    return dice

def cal_dice_commonshape(pd_label, gt_label):
    pd_h, pd_w, pd_l = pd_label.shape
    gt_h, gt_w, gt_l = gt_label.shape
    com_h, com_w, com_l = max(pd_h, gt_h), max(pd_w, gt_w), max(pd_l, gt_l)
    com_pd_label = np.zeros(shape=[com_h, com_w, com_l])
    com_pd_label[:pd_h, :pd_w, :pd_l] = pd_label
    com_gt_label = np.zeros(shape=[com_h, com_w, com_l])
    com_gt_label[:gt_h, :gt_w, :gt_l] = gt_label
    cal_gt_label=np.where(com_gt_label!=0,1,0)
    cal_pd_label=np.where(com_pd_label!=0,1,0)
    intersec=np.logical_and(cal_gt_label,cal_pd_label)
    dice=2*np.sum(intersec)/(np.sum(cal_gt_label)+np.sum(cal_pd_label))
    return dice

def cal_ahd(pd_label, gt_label):
    pd_pts = torch.transpose(torch.stack(torch.where(torch.FloatTensor(pd_label>0))),1,0)
    gt_pts = torch.transpose(torch.stack(torch.where(torch.FloatTensor(gt_label>0))),1,0)
    dist_matrix = torch.sqrt(torch.sum((pd_pts[:,np.newaxis,:] - gt_pts[np.newaxis,:,:])**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    hsd = torch.mean(dist_min).numpy()
    return hsd