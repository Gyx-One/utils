from dependencies import *
from .common import *
from .tensor_op import get_inv_rigid_matrix
from .image_op import cvt_image_gray_to_color
from .image_op import Tsne_embed, write_image_float_gray
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from collections import Counter
from skimage import measure
import cc3d

# volume io operations
# read and write image
def read_mha_image3D(image_path):
    return itk.ReadImage(image_path)

def write_mha_image3D(image, image_path):
    make_parent_dir(image_path)
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

def read_dicom_array3D(dicom_dir, norm=False, ret_param_dict=False):
    image_array, image = read_dicom_image(dicom_dir, ret_image=True)
    if(norm):
        image_array = image_array/2500
    if(ret_param_dict):
        param_dict = {}
        param_dict["Origin"] = image.GetOrigin()
        param_dict["Spacing"] = image.GetSpacing()
        param_dict["Direction"] = image.GetDirection()
        return image_array, param_dict
    return image_array    

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
                      norm=False,
                      make_parent_dir_flag=True
                      ):
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
    if(make_parent_dir_flag):
        make_parent_dir(image_path)
    write_mha_image3D(image, image_path)

def write_mha_array4D(array,image_path,norm=False):
    make_parent_dir(image_path)
    write_mha_array3D(array=array[0],image_path=image_path,norm=norm)

# read and write tensors
def read_mha_tensor3D(image_path,norm=False):
    return torch.FloatTensor(read_mha_array3D(image_path=image_path,norm=norm))

def read_mha_tensor4D(image_path,norm=False):
    return torch.FloatTensor(read_mha_array4D(image_path=image_path,norm=norm))

def write_mha_tensor3D(tensor,image_path,norm=False):
    array=tensor.detach().cpu().numpy()
    make_parent_dir(image_path)
    write_mha_array3D(array,image_path,norm)

def write_mha_tensor4D(tensor,image_path,norm=False, squeeze=True):
    array=tensor.detach().cpu().numpy()
    make_parent_dir(image_path)
    write_mha_array4D(array,image_path,norm)

# array exam function
def exam_array(array, name="default"):
    print(f"array {name}: type:{type(array)} shape:{array.shape} dtype:{array.dtype} max:{np.max(array)} min:{np.min(array)} mean:{np.mean(array)} sum:{np.sum(array)}")

# volumne generation
# connected-component operations
def get_component_3D(cube,keep_num=1,connectivity=2,only_cube=False):
    cube=cube.astype(np.int32)
    label_seg, label_num = measure.label(input=cube, return_num=True, connectivity=connectivity)
    label_count = np.bincount(label_seg.flatten())
    keep_labels = label_count.argsort()[::-1][1:keep_num+1]
    keep_components = [label_seg == label_idx for label_idx in keep_labels]
    keep_vols = [np.sum(keep_com) for keep_com in keep_components]
    if(only_cube):
        return keep_components
    return list(zip(keep_components, keep_vols))

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
    zoom_x=zoom(x,zoom=[dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]],)
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
    # print(f"test zoom_rate:{zoom_rate} zoom_rate[0]:{zoom_rate[0]}")
    # print(f"zoom_rate[1]:{zoom_rate[1]} zoom_rate[2]:{zoom_rate[2]}")
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

# bounding box operations
def get_mask_bbx3D_array(label_cube,margin=4):
    # get bounding box of the binary mask
    # input [h,w,l] tensor
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
        mask[mask_hmin:mask_hmax+1,mask_wmin:mask_wmax+1, mask_lmin:mask_lmax+1]=1
    mask=mask.astype(np.float32)
    return mask

def get_mask_bbx_array2D(label_slice, margin=4):
    # get bounding box of the binary mask
    # input [w, l] tensor
    mask_slice = np.zeros_like(label_slice)
    if(np.sum(label_slice)!=0):
        img_h, img_w = label_slice.shape
        hs, ws =np.where(label_slice>0)
        mask_hmin = np.clip(np.min(hs)-margin,0,img_h)
        mask_hmax = np.clip(np.max(hs)+margin,0,img_h)
        mask_wmin = np.clip(np.min(ws)-margin,0,img_w)
        mask_wmax = np.clip(np.max(ws)+margin,0,img_w)
        mask_slice[mask_hmin:mask_hmax+1, mask_wmin:mask_wmax+1]=1
    mask_slice = mask_slice.astype(np.float32)
    return mask_slice


# bouding box operations
def get_mask_bbx3D_param(label_cube,margin=4):
    #get bounding box of the binary mask
    #input [h,w,l] tensor
    param_dict = {
        "min_h" : None,
        "max_h" : None,
        "range_h" : 0,
        "min_w" : None,
        "max_w" : None,
        "range_w" : 0,
        "min_l" : None,
        "max_l" : None,
        "range_l" : 0
    }
    if(np.sum(label_cube)!=0):
        img_h,img_w,img_l=label_cube.shape
        hs,ws,ls=np.where(label_cube>0)
        mask_hmin=np.clip(np.min(hs)-margin,0,img_h)
        mask_hmax=np.clip(np.max(hs)+margin,0,img_h)
        mask_wmin=np.clip(np.min(ws)-margin,0,img_w)
        mask_wmax=np.clip(np.max(ws)+margin,0,img_w)
        mask_lmin=np.clip(np.min(ls)-margin,0,img_l)
        mask_lmax=np.clip(np.max(ls)+margin,0,img_l)
        param_dict["min_h"] = mask_hmin
        param_dict["max_h"] = mask_hmax
        param_dict["range_h"] = mask_hmax - mask_hmin + 1
        param_dict["min_w"] = mask_wmin
        param_dict["max_w"] = mask_wmax
        param_dict["range_w"] = mask_wmax - mask_wmin + 1
        param_dict["min_l"] = mask_lmin
        param_dict["max_l"] = mask_lmax
        param_dict["range_l"] = mask_lmax - mask_lmin + 1
    crop_info = CropInfo3D(**param_dict)
    return crop_info

# bounding box operations
def get_crop_image_by_bbx_mask_array3D(image, bbx_mask, margin=0, ret_crop_info=True):
    # crop image based on the bounary of the binary mask
    # input [h,w,l] array
    crop_info = get_mask_bbx3D_param(label_cube=bbx_mask, margin=margin)
    min_h, max_h = crop_info.min_h, crop_info.max_h
    min_w, max_w = crop_info.min_w, crop_info.max_w
    min_l, max_l = crop_info.min_l, crop_info.max_l
    crop_cube = image[min_h:max_h+1, min_w:max_w+1, min_l:max_l+1]
    if(ret_crop_info):
        return crop_cube, crop_info
    return crop_cube

# bouding box operations
def get_crop_image_by_param_array3D(image, crop_info):
    # get bounding box crop of the binary mask
    # input [h,w,l] array
    min_h, max_h = crop_info.min_h, crop_info.max_h
    min_w, max_w = crop_info.min_w, crop_info.max_w
    min_l, max_l = crop_info.min_l, crop_info.max_l
    crop_cube = image[min_h:max_h+1, min_w:max_w+1, min_l:max_l+1]
    return crop_cube

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

def get_crop_and_pad_consist_shape(image_list,dst_shape,pad_value=0):
    # crop and pad shape to dst shape for 3D image
    h,w,l = image_list[0].shape
    dst_h, dst_w, dst_l = dst_shape
    # determine crop_h, crop_w, crop_l
    crop_h, crop_w, crop_l = dst_h, dst_w, dst_l
    for image in image_list:
        image_h, image_w, image_l = image.shape
        crop_h, crop_w, crop_l = min(image_h, crop_h), min(image_w, crop_w), min(image_l, crop_l)
    return crop_h, crop_w, crop_l

def fill_component_3Dlayer(image_cube,seg_cube,black_thresh=300/2300):
    seg_value = np.max(seg_cube)
    h,w,l=seg_cube.shape
    if(type(image_cube)==np.ndarray):
        black_cube=np.where(image_cube<black_thresh,1,0)
    else:
        black_cube=np.zeros_like(seg_cube)
    process_cube=(seg_cube==0)
    fill_seg_cube=seg_cube.copy()
    for slice_h in range(0, h):
        seg_slice=process_cube[slice_h,:,:]
        largest_slice=keep_component_2D(seg_slice,keep_num=1)
        res_slice=seg_slice-largest_slice
        fill_seg_cube[slice_h,:,:]+=res_slice
    fill_seg_cube=fill_seg_cube*(1-black_cube)
    fill_seg_cube=np.where(fill_seg_cube!=0,seg_value,0).astype(np.int32)
    return fill_seg_cube

def label_prune_3D(label_cube,errode_num=1,dilation_num=3,keep_num=2):
    from scipy.ndimage import morphology
    label_value=np.max(label_cube)
    origin_cube=np.where(label_cube!=0,1,0).astype(np.int32)
    errode_cube=origin_cube.copy()
    for eid in range(0,errode_num):
        errode_cube=morphology.binary_erosion(errode_cube,iterations=1)
        errode_cube=np.where(errode_cube,1,0).astype(np.int32)
        keep_cube=keep_component_3D(errode_cube,keep_num=keep_num)
        keep_cube=np.where(keep_cube,1,0).astype(np.int32)
        dilation_cube=morphology.binary_dilation(keep_cube,iterations=dilation_num*(eid+1))
        dilation_cube=np.where(dilation_cube,1,0).astype(np.int32)
        final_cube=np.where(np.logical_and(origin_cube!=0,dilation_cube!=0),1,0).astype(np.int32)
    final_cube=np.where(final_cube!=0,label_value,0).astype(np.int32)
    return final_cube

def label_prune_3Dlayer(label_cube,vol_thresh=30,keep_num=2):
    label_value=np.max(label_cube)
    prune_label=np.zeros_like(label_cube)
    h,w,l=label_cube.shape
    for hidx in range(0,h):
        label_layer=label_cube[hidx,:,:]
        if(np.sum(label_layer)>=2):
            keep_list=get_component_2D(label_layer)
            for com,vol in keep_list:
                if(vol>=vol_thresh):
                    prune_label[hidx,:,:]=np.logical_or(prune_label[hidx,:,:],com)
    prune_label=keep_component_3D(prune_label,keep_num=keep_num)
    prune_label=np.where(prune_label!=0,label_value,0).astype(np.int32)
    return prune_label

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
    """
        available mode: bilinear, nearest, bicubic
    """
    from .tensor_op import get_rigid_matrix_from_param, get_deform_from_rigid_matrix, spatial_transform
    # print(f"test spatial_transform trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}")
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

def write_dicom_image(image,
                      dicom_dir,
                      patient_name="DefaultName",
                      patient_id="1000016240",
                      rescale_intercept=-1024,
                      rescale_slope=1,
                      dtype="float"):
    import time
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = image.GetDirection()
    basic_tag_value_list = [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0020|000e",
            "1.2.826.0.1.3680043.2.1125."
            + modification_date + ".1" + modification_time,
        ),  # Series Instance UID
        ("0020|0011", "1"), 
        ("0010|0010", patient_name),  # Patients name
        ("0010|0020", patient_id),  # Patients name
        ("0020|0052", "1.1.698.642632.4.2.4545644516.9769046754"), # Frame of Reference UID
        ("0018|0050", "0.3"),
        ("0028|0100", "16"),  # bits allocated
        ("0028|0101", "16"),  # bits stored
        ("0028|0102", "15"),  # high bit
        ("0028|0103", "1"), # pixel representation,
        ("0028|1052", f"{rescale_intercept}"), # rescale intercept,
        ("0028|1053", f"{rescale_slope}"), # rescale slope
        ("0028|1054", "HU"),
        ("0020|0037",
            "\\".join(
                map(str, (direction[0],direction[3],direction[6],direction[1],direction[4],direction[7],))
            ),
        )  # Image Orientation
    ]
    
    # float point configuration
    # print(f"test detype:{dtype}")
    """
    if dtype == "float":
        # If we want to write floating point values, we need to use the rescale
        # slope, "0028|1053", to select the number of digits we want to keep. We
        # also need to specify additional pixel storage and representation
        # information.
        rescale_slope = 0.001  # keep three digits after the decimal point
        basic_tag_value_list = basic_tag_value_list + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|0103", "1"),
        ]  # pixel representation
    """
    # print(f"test basic_tag_value_list:\n{basic_tag_value_list}")
    writer = itk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for slice_idx in range(0, image.GetDepth()):
        image_slice = image[:, :, slice_idx]
        # set basic tag value
        for basic_tag_value in basic_tag_value_list:
            basic_tag, basic_value = basic_tag_value[0], basic_tag_value[1]
            # print(f"setting basic_tag:{basic_tag} basic_value:{basic_value}")
            image_slice.SetMetaData(basic_tag, basic_value)
        # set series-specific tag value
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        # Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        # Setting the type to CT so that the slice location is preserved and the thickness is carried over.
        image_slice.SetMetaData("0008|0060", "CT")
        # (0020, 0032) image position (patient) determines the 3D spacing betweenslices.
        image_slice.SetMetaData("0020|0032",
            "\\".join(map(str, image.TransformIndexToPhysicalPoint((0, 0, slice_idx)))),
        )
        #   Instance Number
        image_slice.SetMetaData("0020,0013", str(slice_idx))
        # write
        image_slice_path = os.path.join(dicom_dir, f"{slice_idx}.dcm")
        writer.SetFileName(image_slice_path)
        writer.Execute(image_slice)
    return True

def write_dicom_array3D(array, dicom_dir, dtype="float"):
    if(dtype=="int"):
        array = array.astype(np.int32)
    else:
        array = array.astype(np.float64)
    image = itk.GetImageFromArray(array)
    ret_result = write_dicom_image(image=image, dicom_dir=dicom_dir, dtype=dtype)
    return ret_result

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

# image reconstruction functions
def cal_mse_array3D(image_a,
                    image_b,
                    norm_value=1,
                    renorm_value=1):
    cur_image_a = (image_a/norm_value)*renorm_value
    cur_image_b = (image_b/norm_value)*renorm_value
    mse = np.mean((cur_image_a-cur_image_b)**2)
    return mse

def cal_psnr_array3D(image_a,
                     image_b,
                     norm_value=1,
                     maxI=1.0):
    cur_image_a = image_a/norm_value
    cur_image_b = image_b/norm_value
    mse = np.mean((cur_image_a-cur_image_b)**2)
    psnr = 10*np.log10((maxI**2)/mse)
    return psnr
    
def cal_ssim_array3D(image_a,
                     image_b,
                     norm_value=1,
                     L=1.0,
                     k1=0.01,
                     k2=0.03):
    cur_image_a = image_a/norm_value
    cur_image_b = image_b/norm_value
    mu_x = np.mean(cur_image_a)
    mu_y = np.mean(cur_image_b)
    std_x = np.std(cur_image_a)
    std_y = np.std(cur_image_b)
    std_xy = np.mean(cur_image_a*cur_image_b)-np.mean(cur_image_a)*np.mean(cur_image_b)
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    ssim_nume = (2*mu_x*mu_y+c1)*(2*std_xy+c2)
    ssim_deno = ((mu_x**2)+(mu_y**2)+c1)*((std_x**2)+(std_y**2)+c2)
    ssim = ssim_nume/ssim_deno
    return ssim

def cal_ncc_array3D(image_gt, image_pd):
    mu_gt = np.mean(image_gt)
    sigma_gt = np.std(image_gt)
    mu_pd = np.mean(image_pd)
    sigma_pd = np.std(image_pd)
    n = image_gt.shape[0]*image_gt.shape[1]*image_gt.shape[2]
    corr_image = ((image_gt-mu_gt)*(image_pd-mu_pd))/(sigma_gt*sigma_pd)
    ncc = np.sum(corr_image)/n
    return ncc

# get label cube centroid functions 
def get_centroid(cube, round=False):
    hs, ws, ls = np.where(cube>1e-2)
    mean_h = np.mean(hs)
    mean_w = np.mean(ws)
    mean_l = np.mean(ls)
    if(round):
        mean_h = int(np.round(mean_h))
        mean_w = int(np.round(mean_w))
        mean_l = int(np.round(mean_l))
    return mean_h, mean_w, mean_l

def get_centroid_array3D(cube, round=False):
    return get_centroid(cube=cube, round=round)

def get_centroid_array2D(cube, round=False):
    hs, ws = np.where(cube>1e-2)
    mean_h = np.mean(hs)
    mean_w = np.mean(ws)
    if(round):
        mean_h = int(np.round(mean_h))
        mean_w = int(np.round(mean_w))
    return mean_h, mean_w


def get_mincoor(cube, round=False):
    hs, ws, ls = np.where(cube>1e-2)
    min_h = np.min(hs)
    min_w = np.min(ws)
    min_l = np.min(ls)
    if(round):
        min_h = int(np.round(min_h))
        min_w = int(np.round(min_w))
        min_l = int(np.round(min_l))
    return min_h, min_w, min_l

def possion_image_edit_array2d(src_image, dst_image, mask):
    """
    src_image: array, [h,w,l] , the image that to copy patch from
    dst_image: array, [h,w,l] , the image to be filled,
    mask: array, [h,w,l] , the binary image indicates the region to be filled, is of value {0,1}.
    
    notes: refer from William Emmanuel, wemmanuel3@gatech.edu
    """
    import numpy as np
    from scipy.sparse import linalg as linalg
    from scipy.sparse import lil_matrix as lil_matrix
    # Helper enum
    OMEGA = 0
    DEL_OMEGA = 1
    OUTSIDE = 2

    # Determine if a given index is inside omega, on the boundary (del omega),
    # or outside the omega region
    def point_location(index, mask):
        if in_omega(index,mask) == False:
            return OUTSIDE
        if edge(index,mask) == True:
            return DEL_OMEGA
        return OMEGA

    # Determine if a given index is either outside or inside omega
    def in_omega(index, mask):
        return mask[index] == 1

    # Deterimine if a given index is on del omega (boundary)
    def edge(index, mask):
        if in_omega(index,mask) == False: return False
        for pt in get_surrounding(index):
            # If the point is inside omega, and a surrounding point is not,
            # then we must be on an edge
            if in_omega(pt,mask) == False: return True
        return False

    # Apply the Laplacian operator at a given index
    def lapl_at_index(source, index):
        i,j = index
        val = (4 * source[i,j])    \
            - (1 * source[i+1, j]) \
            - (1 * source[i-1, j]) \
            - (1 * source[i, j+1]) \
            - (1 * source[i, j-1])
        return val

    # Find the indicies of omega, or where the mask is 1
    def mask_indicies(mask):
        nonzero = np.nonzero(mask)
        return list(zip(nonzero[0], nonzero[1]))

    # Get indicies above, below, to the left and right
    def get_surrounding(index):
        i,j = index
        return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

    # Create the A sparse matrix
    def poisson_sparse_matrix(points):
        # N = number of points in mask
        N = len(points)
        A = lil_matrix((N,N))
        # Set up row for each point in mask
        for i,index in enumerate(points):
            # Should have 4's diagonal
            A[i,i] = 4
            # Get all surrounding points
            for x in get_surrounding(index):
                # If a surrounding point is in the mask, add -1 to index's
                # row at correct position
                if x not in points: continue
                j = points.index(x)
                A[i,j] = -1
        return A

    # Main method
    # Does Poisson image editing on one channel given a source, target, and mask
    def process(source, target, mask):
        indicies = mask_indicies(mask)
        N = len(indicies)
        # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
        A = poisson_sparse_matrix(indicies)
        # Create B matrix
        b = np.zeros(N)
        for i,index in enumerate(indicies):
            # Start with left hand side of discrete equation
            b[i] = lapl_at_index(source, index)
            # If on boundry, add in target intensity
            # Creates constraint lapl source = target at boundary
            if point_location(index, mask) == DEL_OMEGA:
                for pt in get_surrounding(index):
                    if in_omega(pt,mask) == False:
                        b[i] += target[pt]

        # Solve for x, unknown intensities
        x = linalg.cg(A, b)
        # Copy target photo, make sure as int
        composite = np.copy(target)
        # Place new intensity on target at given index
        for i,index in enumerate(indicies):
            composite[index] = x[0][i]
        # print(f"test source.shape:{source.shape} target.shape:{target.shape} mask:{mask.shape} composite:{composite.shape}")
        return composite
    
    return process(source=src_image, target=dst_image, mask=mask)

def possion_image_edit_array3d(src_image, dst_image, mask):
    """
    src_image: array, [h,w,l] , the image that to copy patch from
    dst_image: array, [h,w,l] , the image to be filled,
    mask: array, [h,w,l] , the binary image indicates the region to be filled, is of value {0,1}.
    
    notes: modify from William Emmanuel, wemmanuel3@gatech.edu.
    """
    import numpy as np
    from scipy.sparse import linalg as linalg
    from scipy.sparse import lil_matrix as lil_matrix

    # Helper enum
    OMEGA = 0
    DEL_OMEGA = 1
    OUTSIDE = 2

    # Determine if a given index is inside omega, on the boundary (del omega),
    # or outside the omega region
    def point_location(index, mask):
        if in_omega(index,mask) == False:
            return OUTSIDE
        if edge(index,mask) == True:
            return DEL_OMEGA
        return OMEGA

    # Determine if a given index is either outside or inside omega
    def in_omega(index, mask):
        return mask[index] == 1

    # Deterimine if a given index is on del omega (boundary)
    def edge(index, mask):
        if in_omega(index,mask) == False: return False
        for pt in get_surrounding(index):
            # If the point is inside omega, and a surrounding point is not,
            # then we must be on an edge
            if in_omega(pt,mask) == False: return True
        return False

    # Apply the Laplacian operator at a given index
    def lapl_at_index(source, index):
        i,j,k = index
        val = (6 * source[i,j,k])    \
            - (1 * source[i+1, j, k]) \
            - (1 * source[i-1, j, k]) \
            - (1 * source[i, j+1, k]) \
            - (1 * source[i, j-1, k]) \
            - (1 * source[i, j, k+1]) \
            - (1 * source[i, j, k-1])
        return val

    # Find the indicies of omega, or where the mask is 1
    def mask_indicies(mask):
        nonzero = np.nonzero(mask)
        return list(zip(nonzero[0], nonzero[1], nonzero[2]))

    # Get indicies above, below, to the left and right
    def get_surrounding(index):
        i,j,k = index
        return [(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k-1),(i,j,k+1)]

    # Create the A sparse matrix
    def poisson_sparse_matrix(points):
        # N = number of points in mask
        N = len(points)
        A = lil_matrix((N,N))
        # Set up row for each point in mask
        for i,index in enumerate(points):
            # Should have 4's diagonal
            A[i,i] = 6
            # Get all surrounding points
            for x in get_surrounding(index):
                # If a surrounding point is in the mask, add -1 to index's
                # row at correct position
                if x not in points: continue
                j = points.index(x)
                A[i,j] = -1
        return A

    # Main method
    # Does Poisson image editing on one channel given a source, target, and mask
    def process3d(source, target, mask):
        indicies = mask_indicies(mask)
        N = len(indicies)
        # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
        A = poisson_sparse_matrix(indicies)
        # Create B matrix
        b = np.zeros(N)
        for i,index in enumerate(indicies):
            # Start with left hand side of discrete equation
            b[i] = lapl_at_index(source, index)
            # If on boundry, add in target intensity
            # Creates constraint lapl source = target at boundary
            if point_location(index, mask) == DEL_OMEGA:
                for pt in get_surrounding(index):
                    if in_omega(pt,mask) == False:
                        b[i] += target[pt]

        # Solve for x, unknown intensities
        # print(f"test shape A:{A.shape} b:{b.shape}")
        x = linalg.cg(A, b)
        # print(f"test x.shape: {x[0].shape}")
        # Copy target photo, make sure as int
        composite = np.copy(target)
        # Place new intensity on target at given index
        for i,index in enumerate(indicies):
            composite[index] = x[0][i]
        # print(f"test source.shape:{source.shape} target.shape:{target.shape} mask:{mask.shape} composite:{composite.shape}")
        return composite
    
    return process3d(source=src_image, target=dst_image, mask=mask)


def possion_image_edit_by_condmask_array3d(src_image,
                                          dst_image,
                                          mask,
                                          src_cond_mask,
                                          tgt_cond_mask):
    """
    src_image: array, [h,w,l] , the image that to copy patch from
    dst_image: array, [h,w,l] , the image to be filled,
    mask: array, [h,w,l] , the binary image indicates the region to be filled, is of value {0,1}.
    
    notes: modify from William Emmanuel, wemmanuel3@gatech.edu.
    """
    import numpy as np
    from scipy.sparse import linalg as linalg
    from scipy.sparse import lil_matrix as lil_matrix

    # Helper enum
    OMEGA = 0
    DEL_OMEGA = 1
    OUTSIDE = 2

    # Determine if a given index is inside omega, on the boundary (del omega),
    # or outside the omega region
    def point_location(index, mask):
        if in_omega(index,mask) == False:
            return OUTSIDE
        if edge(index,mask) == True:
            return DEL_OMEGA
        return OMEGA

    # Determine if a given index is either outside or inside omega
    def in_omega(index, mask):
        return mask[index] == 1

    # Deterimine if a given index is on del omega (boundary)
    def edge(index, mask):
        if in_omega(index,mask) == False: return False
        for pt in get_surrounding(index):
            # If the point is inside omega, and a surrounding point is not,
            # then we must be on an edge
            if in_omega(pt,mask) == False: return True
        return False

    # Apply the Laplacian operator at a given index
    def lapl_at_index(source, index):
        i,j,k = index
        val = (6 * source[i,j,k])    \
            - (1 * source[i+1, j, k]) \
            - (1 * source[i-1, j, k]) \
            - (1 * source[i, j+1, k]) \
            - (1 * source[i, j-1, k]) \
            - (1 * source[i, j, k+1]) \
            - (1 * source[i, j, k-1])
        return val

    # Find the indicies of omega, or where the mask is 1
    def mask_indicies(mask):
        nonzero = np.nonzero(mask)
        return list(zip(nonzero[0], nonzero[1], nonzero[2]))

    # Get indicies above, below, to the left and right
    def get_surrounding(index):
        i,j,k = index
        return [(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k-1),(i,j,k+1)]

    # Create the A sparse matrix
    def poisson_sparse_matrix(points):
        # N = number of points in mask
        N = len(points)
        A = lil_matrix((N,N))
        # Set up row for each point in mask
        for i,index in enumerate(points):
            # Should have 4's diagonal
            A[i,i] = 6
            # Get all surrounding points
            for x in get_surrounding(index):
                # If a surrounding point is in the mask, add -1 to index's
                # row at correct position
                if x not in points: continue
                j = points.index(x)
                A[i,j] = -1
        return A

    # Main method
    # Does Poisson image editing on one channel given a source, target, and mask
    def process3d(source, target, mask, src_cond_mask, tgt_cond_mask):
        indicies = mask_indicies(mask)
        N = len(indicies)
        # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
        A = poisson_sparse_matrix(indicies)
        # Create B matrix
        b = np.zeros(N)
        for i,index in enumerate(indicies):
            # Start with left hand side of discrete equation
            b[i] = lapl_at_index(source, index)
            # If on boundry, add in target intensity
            # Creates constraint lapl source = target at boundary
            if point_location(index, mask) == DEL_OMEGA:
                for pt in get_surrounding(index):
                    if in_omega(pt,mask) == False:
                        if(src_cond_mask[pt]):
                            b[i] += source[pt]
                        elif(tgt_cond_mask[pt]):
                            b[i] += target[pt]
                        else:
                            b[i] += target[pt]

        # Solve for x, unknown intensities
        x = linalg.cg(A, b)
        # Copy target photo, make sure as int
        composite = np.copy(target)
        # Place new intensity on target at given index
        for i,index in enumerate(indicies):
            composite[index] = x[0][i]
        # print(f"test source.shape:{source.shape} target.shape:{target.shape} mask:{mask.shape} composite:{composite.shape}")
        return composite
    
    return process3d(source=src_image, target=dst_image, mask=mask, src_cond_mask=src_cond_mask, tgt_cond_mask=tgt_cond_mask)

# mask contour and periphery functions
def get_mask_contour_array3D(mask, border_width=1):
    """
    get the inner contour voxel of a binary mask,
    mask: array [h,w,l], is of value {0,1} 
    
    notes: the inner contour voxels are inside the mask!
    """
    errode_mask = binary_erosion(mask, iterations=border_width).astype(np.float32)
    contour = np.where(np.logical_and(mask>1e-1, errode_mask<1e-1), 1, 0).astype(np.float32)
    return contour

def get_mask_contour_array2D(mask, border_width=1):
    return get_mask_periphery_array3D(mask=mask, border_width=border_width)

def get_mask_periphery_array3D(mask, border_width=1):
    """
    get the outer periphery voxel of a binary mask.
    mask: array [h,w,l], is of value {0,1} 
    notes: the periphery voxels are outside the binary mask!
    """
    dilate_mask = binary_dilation(mask, iterations=border_width).astype(np.float32)
    periphery = np.where(np.logical_and(dilate_mask>1e-1, mask<1e-1), 1, 0).astype(np.float32)
    return periphery

def get_mask_front_contour_array3D(mask, border_width=1):
    h, w, l = mask.shape
    mask_front_contour = np.zeros_like(mask)
    for h_idx in range(0, h):
        for l_idx in range(0, l):
                w_slice = mask[h_idx, :, l_idx]
                if(np.sum(w_slice)!=0):
                    ws = np.where(w_slice>1e-1)[0]
                    min_w = int(np.min(ws))
                    mask_front_contour[h_idx, min_w, l_idx] = 1
    return mask_front_contour


def check_contour_intersect(base_mask, mask_b, border_width=1):
    base_contour = get_mask_periphery_array3D(base_mask, border_width=border_width)
    intersec = np.where(np.logical_and(base_contour>1e-1, mask_b>1e-1), 1, 0)
    if(np.sum(intersec)>1e-1):
        return True
    else:
        return False

def get_union_mask(base_mask, add_mask):
    union_mask = np.where(add_mask>1e-1, add_mask, base_mask)
    return union_mask

def get_dif_mask(base_mask, minus_mask):
    dif_mask = np.where(minus_mask>1e-1, 0, base_mask)
    return dif_mask

# bounding box related function
def get_bbx_boundry_array3D(bbx_mask):
    """
    get the boundry of the given bbx_mask

    Args:
        bbx_mask (array): shape [h,w,l], binary.
    
    Returns:
        min_h (array): min_h.
        max_h (array): max_h.
        min_w (array): min_w.
        max_w (array): max_w.
        min_l (array): min_l.
        max_l (array): max_l.
    """
    hs, ws, ls = np.where(bbx_mask>0)
    # hs, ws, ls
    min_h, max_h = np.min(hs),np.max(hs)
    min_w, max_w = np.min(ws),np.max(ws)
    min_l, max_l = np.min(ls),np.max(ls)
    return min_h, max_h, min_w, max_w, min_l, max_l

def get_bbx_boundry_dict_array3D(bbx_mask):
    """
    get the boundry of the given bbx_mask, ret in a dict format.
    keys of the dict are `min_h`, `max_h`, `min_w`, `max_w`, `min_l`, `max_l`

    Args:
        bbx_mask (array): shape [h,w,l], binary.
    
    Returns:
        boundry_dict
    """
    min_h, max_h, min_w, max_w, min_l, max_l = get_bbx_boundry_array3D(bbx_mask)
    boundry_dict = {
        "min_h" : min_h,
        "max_h" : max_h,
        "min_w" : min_w,
        "max_w" : max_w,
        "min_l" : min_l,
        "max_l" : max_l
    }
    return boundry_dict


def get_bbx_range_array3D(bbx_mask):
    """
    get the boundry of the given bbx_mask

    Args:
        bbx_mask (array): shape [h,w,l], binary.
    
    Returns:
        range_h (array): range of bbx in h-dimension.
        range_w (array): range of bbx in w-dimension.
        range_l (array): range of bbx in l-dimension
    """
    min_h, max_h, min_w, max_w, min_l, max_l = get_bbx_boundry_array3D(bbx_mask=bbx_mask)
    range_h = max_h - min_h + 1
    range_w = max_w - min_w + 1
    range_l = max_l - min_l + 1
    return range_h, range_w, range_l

def cvt_seg_value_array3D(seg, thresh_value=1e-1, cvt_value=1.0):
    """
    convert all values in seg that are greater than thresh_value into cvt_value 
    """
    cvt_seg = np.where(seg>thresh_value, cvt_value, 0.0).astype(np.float32)
    return cvt_seg

# crop image functions
class CropInfo3D:
    def __init__(self,
                 min_h,
                 max_h,
                 min_w,
                 max_w,
                 min_l,
                 max_l,
                 *args,
                 **kwargs
                 ):
        # notes:
        # crop range: [min_h, max_h], is not [min_h, max) !
        self.min_h = min_h
        self.max_h = max_h
        self.range_h = self.max_h - self.min_h + 1
        self.min_w = min_w
        self.max_w = max_w
        self.range_w = self.max_w - self.min_w + 1
        self.min_l = min_l
        self.max_l = max_l
        self.range_l = self.max_l - self.min_l + 1
    
    def crop_image_func(self, image):
        crop_image = image[self.min_h:self.max_h+1, self.min_w:self.max_w+1, self.min_l:self.max_l+1]
        return crop_image
    
    def filter_image_func(self, image):
        filter_image = np.zeros_like(image)
        filter_image[self.min_h:self.max_h+1, self.min_w:self.max_w+1, self.min_l:self.max_l+1] = \
            image[self.min_h:self.max_h+1, self.min_w:self.max_w+1, self.min_l:self.max_l+1]
        return filter_image

def get_crop_image_array3D(image,
                           start_h,
                           start_w,
                           start_l,
                           crop_range_h,
                           crop_range_w,
                           crop_range_l):
    start_h = round_int(start_h)
    start_w = round_int(start_w)
    start_l = round_int(start_l)
    crop_range_h = round_int(crop_range_h)
    crop_range_w = round_int(crop_range_w)
    crop_range_l = round_int(crop_range_l)
    crop_image = copy.copy(image[start_h:start_h+crop_range_h, start_w:start_w+crop_range_w, start_l:start_l+crop_range_l])
    return crop_image

def get_crop_image_array3D_dict(image_dict,
                                start_h,
                                start_w,
                                start_l,
                                crop_range_h,
                                crop_range_w,
                                crop_range_l):
    crop_image_dict = {}
    for image_key in image_dict.keys():
        image = image_dict[image_key]
        crop_image = get_crop_image_array3D(
            image = image,
            start_h = start_h,
            start_w = start_w,
            start_l = start_l,
            crop_range_h = crop_range_h,
            crop_range_w = crop_range_w,
            crop_range_l = crop_range_l
        )
        crop_image_dict[image_key] = crop_image
    return crop_image_dict

def get_crop_image_array3D_list(image_list,
                                start_h,
                                start_w,
                                start_l,
                                crop_range_h,
                                crop_range_w,
                                crop_range_l):
    crop_image_list = []
    for image in image_list:
        crop_image = get_crop_image_array3D(
            image = image,
            start_h = start_h,
            start_w = start_w,
            start_l = start_l,
            crop_range_h = crop_range_h,
            crop_range_w = crop_range_w,
            crop_range_l = crop_range_l
        )
        crop_image_list.append(crop_image)
    return crop_image_list

def get_2D_image_crop_center_from_3D_binary_image(
    binary_3D_mask,
    slice_type
):
    """
        给定二值掩码图像, 输出裁剪2D图像区域的范围
    """
    
    center_h, center_w, center_l = get_centroid_array3D(binary_3D_mask)
    crop_center_h = binary_3D_mask.shape[0] - center_h
    crop_center_w = center_w
    crop_center_l = binary_3D_mask.shape[2]//2
    if(slice_type=="slice_h"):
        return crop_center_w, crop_center_l
    elif(slice_type=="slice_w"):
        return  crop_center_h, crop_center_l
    elif(slice_type=="slice_l"):
        return crop_center_h, crop_center_w
    else:
        raise NotImplementedError(f"invalid slice_type: {slice_type}")

# get slice 2D image from 3D cube functions
def get_slice_h_image_from_cube_array3D(
    cube,
    slice_idx,
    cvt_color_flag=False
):
    slice_h = cube[slice_idx, :, :]
    if(cvt_color_flag):
        slice_h = cvt_image_gray_to_color(slice_h)
    return slice_h

def get_slice_w_image_from_cube_array3D(
    cube,
    slice_idx,
    cvt_color_flag=False
):
    slice_w = cube[:, slice_idx, :][::-1, :]
    if(cvt_color_flag):
        slice_w = cvt_image_gray_to_color(slice_w)
    return slice_w

def get_slice_l_image_from_cube_array3D(
    cube,
    slice_idx,
    cvt_color_flag=False
):
    slice_l = cube[:, :, slice_idx][::-1, :]
    if(cvt_color_flag):
        slice_l = cvt_image_gray_to_color(slice_l)
    return slice_l

def get_slice_image_from_cube_array3D(
    cube,
    slice_idx,
    slice_type,
    cvt_color_flag=False
):
    """
    available slice_type:
        slice_h, slice_w, slice_l
    """
    gen_slice_func = None
    if(slice_type == "slice_h"):
        gen_slice_func = get_slice_h_image_from_cube_array3D
    elif(slice_type == "slice_w"):
        gen_slice_func = get_slice_w_image_from_cube_array3D
    elif(slice_type == "slice_l"):
        gen_slice_func = get_slice_l_image_from_cube_array3D
    else:
        raise NotImplementedError(f"Error Slice Type: {slice_type}")
    slice_image = gen_slice_func(
        cube = cube,
        slice_idx = slice_idx,
        cvt_color_flag = cvt_color_flag
    )
    return slice_image

# array unsqueeze functions
def unsqueeze_array4D_to_array3D(array4D):
    return array4D[0]

def unsqueeze_array4D_to_array3D_list(array4D_list):
    array3D_list = [array4D[0] for array4D in array4D_list]
    return array3D_list

# embedding functions
class Embed_Info:
    def __init__(
        self,
        embed_idx=-1,
        embed_coor=None,
        class_label=None,
        data_name=None
    ):
        self.embed_idx = embed_idx
        self.embed_coor = embed_coor
        self.class_label = class_label
        self.data_name = data_name
    
    def to_dict(self):
        embed_coor_list = list(self.embed_coor)
        cvt_embed_coor_list = [float(single_coor) for single_coor in embed_coor_list]
        ret_dict =  {
            "embed_idx": self.embed_idx,
            "embed_coor": cvt_embed_coor_list,
            "class_label": self.class_label,
            "data_name": self.data_name
        }
        return ret_dict
    
    def from_dict(self, load_dict):
        self.embed_idx = load_dict["embed_idx"]
        self.embed_coor = np.array(load_dict["embed_coor"])
        self.class_label = load_dict["class_label"]
        self.data_name = load_dict["data_name"]
        return True        
    
    def set_embed_coor(self, embed_coor):
        self.embed_coor = embed_coor
    
    def get_embed_coor(self):
        return self.embed_coor
    
    def set_class_label(self, class_label):
        self.class_label = class_label
    
    def get_class_label(self):
        return self.class_label

    def set_data_name(self, data_name):
        self.data_name = data_name
    
    def get_data_name(self):
        return self.data_name
    
    def get_info_str(self):
        info_str = f"embed_idx: {self.embed_idx}\t data_name:{self.data_name}\t class_label: {self.class_label}\t embed_coor: {self.embed_coor}"
        return info_str

class Embed_Manager:
    def __init__(self,
                 save_dir,
                 default_color="#2f2c30"):
        self._embed_type = "TSNE_embed"
        self._embed_func = Tsne_embed
        self.save_dir = save_dir 
        self.default_color = default_color
        make_dir(self.save_dir)
        # embed attrs
        self.data_num = -1
        self.embed_raw_data = None
        self.embed_raw_label = None
        self.embed_matrix = None
        self.embed_info_dict = {}
        self.label_color_dict = {}
        # save attrs
        self.save_embed_info_dict_path = f"{self.save_dir}/embed_info_dict.json"
        self.save_embed_matrix_path = f"{self.save_dir}/embed_matrix.txt"
        self.save_label_color_dict_path = f"{self.save_dir}/label_color_dict.json"
    
    def set_label_color_dict(self, label_color_dict):
        self.label_color_dict = label_color_dict
    
    def get_embed_coor(self, embed_idx):
        return self.embed_info_dict[embed_idx].get_embed_coor()
    
    def get_embed_idx_by_data_name(self, input_data_name):
        data_name_dict = {}
        for embed_idx in self.embed_info_dict.keys():
            embed_info = self.embed_info_dict[embed_idx]
            data_name = embed_info.get_data_name()
            if(data_name in data_name_dict):
                raise NotImplementedError(f"Repeate Data Name, Error! data_name:{data_name}")
            else:
                data_name_dict[data_name] = embed_idx
        return data_name_dict[input_data_name]
    
    def get_label_color(self, label):
        if(self.label_color_dict is None):
            return None
        else:
            return self.label_color_dict[label]
    
    def get_label_list_by_embed_info_dict(self, temp_embed_info_dict):
        cur_label_list = []
        for embed_idx in range(0, self.embed_num):
            embed_info = temp_embed_info_dict[embed_idx]
            embed_class_label = embed_info.get_class_label()
            if(embed_class_label is not None):
                cur_label_list.append(embed_class_label)
        return cur_label_list
    
    def get_cur_label_list(self):
        return self.get_label_list_by_embed_info_dict(self.embed_info_dict)
    
    def get_class_embed_list_by_embed_info_dict(self, temp_embed_info_dict, label):
        class_embed_idx_list = []
        class_embed_x_list = []
        class_embed_y_list = []
        for embed_idx in range(0, self.embed_num):
            embed_info = temp_embed_info_dict[embed_idx]
            if(embed_info.get_class_label() == label):
                class_embed_idx_list.append(embed_idx)
                embed_coor = self.embed_matrix[embed_idx, :]
                class_embed_x_list.append(embed_coor[0])
                class_embed_y_list.append(embed_coor[1])
        return class_embed_idx_list, class_embed_x_list, class_embed_y_list
    
    def get_cur_class_embed_list(self, label):
        return self.get_class_embed_list_by_embed_info_dict(
            temp_embed_info_dict=self.embed_info_dict,
            label=label
        )
    
    def set_embed_raw(
        self,
        embed_raw_data,
        embed_raw_label=None,
        embed_raw_name=None,
        label_color_dict=None
    ):
        self.embed_raw_data = embed_raw_data
        self.embed_raw_label = embed_raw_label
        self.embed_raw_name = embed_raw_name
        self.label_color_dict = label_color_dict
        self.embed_num = self.embed_raw_data.shape[0]
        
        for embed_idx in range(0, self.embed_num):
            self.embed_info_dict[embed_idx] = Embed_Info(
                    embed_idx = embed_idx
            )
            if(self.embed_raw_label is not None):
                self.embed_info_dict[embed_idx].set_class_label(self.embed_raw_label[embed_idx])
            if(self.embed_raw_name is not None):
                self.embed_info_dict[embed_idx].set_data_name(self.embed_raw_name[embed_idx])
        return True
    
    def exec_embedding(self, embed_dim=2):
        """
        embed_raw_data:
            array2D, [n, k], n is the number of data, k is the dim of feature
        embed_raw_label:
            array3D, [n], n is the number of data
        """
        # print(f"test self.embed_raw_data: type:{type(self.embed_raw_data)} shape:{self.embed_raw_data.shape}")
        self.embed_matrix = self._embed_func(x=self.embed_raw_data,
                                             embed_dim=embed_dim)
        
        for embed_idx in range(0, self.embed_num):
            # embed coor
            self.embed_info_dict[embed_idx].set_embed_coor(self.embed_matrix[embed_idx])
    
    
    def plot_embedding(
        self,
        plot_path,
        plot_chosen_label_dict=None,
        title="",
        print_info_flag=False
    ):
        """
        structure of chosen_label_dict
            key: embed_idx
            value: label
        """
        plot_embed_x = copy.deepcopy(self.embed_matrix)
        plot_embed_info_dict = copy.deepcopy(self.embed_info_dict)
        
        # set labels of chosen_data
        if(plot_chosen_label_dict is not None):
            for chosen_idx in plot_chosen_label_dict.keys():
                chosen_label = plot_chosen_label_dict[chosen_idx]
                plot_embed_info_dict[chosen_idx].set_class_label(chosen_label)
                # print(f"test set chosen_idx:{chosen_idx} chosen_label:{chosen_label}")
        
        if(print_info_flag):
            self.exam_label_dist(temp_embed_info_dict=plot_embed_info_dict)
            self.exam_label_color_dict()
        
        fig, ax = plt.subplots()
        cord_min = np.min(plot_embed_x,axis=0)
        cord_max = np.max(plot_embed_x,axis=0)
        plot_embed_x = (plot_embed_x-cord_min)/(cord_max-cord_min)*1.4
        plot_label_list = np.unique(self.get_label_list_by_embed_info_dict(temp_embed_info_dict=plot_embed_info_dict))
        plot_label_list = plot_label_list[::-1]
        
        s_legend_list = []
        name_legend_list = []
        for label in plot_label_list:
            class_embed_idx_list, class_embed_x_list, class_embed_y_list = \
                self.get_class_embed_list_by_embed_info_dict(temp_embed_info_dict=plot_embed_info_dict, label=label)
            class_idx = np.array(class_embed_idx_list)
            class_embed_x = np.array(class_embed_x_list)
            class_embed_y = np.array(class_embed_y_list)
            class_color = self.get_label_color(label)
            class_name = label
            class_s = ax.scatter(x=class_embed_x, y=class_embed_y, c=class_color)
            s_legend_list.append(class_s)
            name_legend_list.append(class_name)
        plt.legend(s_legend_list, name_legend_list, loc="best")
        plt.title(title)
        make_parent_dir(plot_path)
        plt.savefig(plot_path)
        plt.close()
    
    def save_status(self):
        # embed_info_dict
        print("Embed Manager Save embed_info_dict")
        save_embed_info_dict = {}
        for embed_idx in list(self.embed_info_dict.keys()):
            save_embed_info_dict[embed_idx] = self.embed_info_dict[embed_idx].to_dict()
            # print(f"test save_embed_info_dict[embed_idx]: {save_embed_info_dict[embed_idx]} type(embed_coor):{type(embed_coor)} {type(embed_coor[0])} {type(1.2124)} {type(float(embed_coor[1]))}")
        common_json_dump(save_embed_info_dict, self.save_embed_info_dict_path)
        # embed matrix
        print("Embed Manager Save embed_matrix")
        common_array_dump(self.embed_matrix, self.save_embed_matrix_path)
        # label color dict
        print("Embed Manager Save color_dict")
        if(self.label_color_dict is not None):
            common_json_dump(self.label_color_dict, self.save_label_color_dict_path)
        return True
    
    def load_status(self):
        print("[Loading Embeding Status]")
        # embed info dict
        print("\t Loading Embed Info Dict")
        if(os.path.exists(self.save_embed_info_dict_path)):
            save_embed_info_dict = common_json_load(self.save_embed_info_dict_path)
            for embed_idx in list(save_embed_info_dict.keys()):
                self.embed_info_dict[int(embed_idx)] = Embed_Info(embed_idx=embed_idx)
                self.embed_info_dict[int(embed_idx)].from_dict(save_embed_info_dict[embed_idx])
            self.embed_num = len(self.embed_info_dict.keys())
            print("\t\t Embed Info Dict Load Success")
        else:
            print("\t\t Embed Info Dict Path not exists! Load Error, Ignore.")
        # embed matrix
        print("\t Loading Embed Matrix")
        if(os.path.exists(self.save_embed_matrix_path)):
            self.embed_matrix = common_array_load(self.save_embed_matrix_path)
            print("\t\t Embed Matrix Load Success")
        else:
            print("\t\t Embed Matrix Path not exists! Load Error, Ignore.")
        # label color dict
        print("\t Loading Label Color Dict")
        if(os.path.exists(self.save_label_color_dict_path)):
            self.label_color_dict = common_json_load(self.save_label_color_dict_path)
            print("\t\t Label Color Dict Load Success")
        else:
            print("\t\t Label Color Dict Path not exists! Load Error, Ignore.")
    
    def exam_label_dist(self, temp_embed_info_dict):
        label_list = self.get_label_list_by_embed_info_dict(temp_embed_info_dict=temp_embed_info_dict)
        dist_label_dict = dict(Counter(label_list))
        for label in dist_label_dict.keys():
            print(f"\t\t label: {label} count: {dist_label_dict[label]}")
    
    def exam_label_color_dict(self):
        if(self.label_color_dict is None):
            print("Error! label_color_dict is None")
        else:
            for label in self.label_color_dict.keys():
                print(f"\t\t label: {label} color: {self.label_color_dict[label]}")
    
    def exam_status(self, print_details_flag=False):
        print("[Exam Status]")
        # check correspondence between embed_matrix and embed_info_dict
        correspond_flag = True
        if(len(self.embed_info_dict.keys())!=self.embed_matrix.shape[0]):
            correspond_flag = False
        for embed_idx in self.embed_info_dict.keys():
            embed_info = self.embed_info_dict[embed_idx]
            embed_info_coor = embed_info.get_embed_coor()
            embed_matrix_coor = self.embed_matrix[embed_idx, :]
            if(np.sum(embed_info_coor!=embed_matrix_coor)>0):
                correspond_flag = False
        print(f"\t Exam Correspond: {correspond_flag}")
        
        # check embed_num
        print(f"\t Embed Num: {self.embed_num}")
        
        # check label dist
        print("\t Check Label Dist:")
        self.exam_label_dist(temp_embed_info_dict=self.embed_info_dict)
        
        # check color dict
        print("\t Check Color Dict")
        self.exam_label_color_dict()
            
        # check details
        if(print_details_flag):
            print("\t Exam Details Info")
            for embed_idx in self.embed_info_dict.keys():
                embed_info = self.embed_info_dict[embed_idx]
                print(f"\t\t embed_idx:{embed_idx}\t {embed_info.get_info_str()}")
        return True

def get_circle_seg_array3D(
    shape,
    center_array,
    radius
):
    """
    center_array [3], 1-dimension array with length equals 3.
    """
    shape_h, shape_w, shape_l = shape
    mesh_grid = np.stack(np.mgrid[0:shape_h:1, 0:shape_w:1, 0:shape_l:1])
    dist = (mesh_grid - center_array[:, np.newaxis, np.newaxis, np.newaxis])
    dist = np.sqrt(np.sum(dist**2, axis=0))
    circle_seg = np.where(dist<radius, 1, 0).astype(np.float32)
    return circle_seg