from .dependencies import *
import cc3d
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import measure

#connected-component
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

#morphology operation
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

#binary volume prune
def fill_component_3Dlayer(image_cube,seg_cube,black_thresh=300/2300):
    seg_value=np.max(seg_cube)
    h,w,l=seg_cube.shape
    if(type(image_cube)==np.ndarray):
        black_cube=np.where(image_cube<black_thresh,1,0)
    else:
        black_cube=np.zeros_like(seg_cube)
    process_cube=(seg_cube==0)
    fill_seg_cube=seg_cube.copy()
    for slice_h in range(0,h):
        seg_slice=process_cube[slice_h,:,:]
        largest_slice=keep_component_2D(seg_slice,keep_num=1)
        res_slice=seg_slice-largest_slice
        fill_seg_cube[slice_h,:,:]+=res_slice
    fill_seg_cube=fill_seg_cube*(1-black_cube)
    fill_seg_cube=np.where(fill_seg_cube!=0,seg_value,0).astype(np.int32)
    return fill_seg_cube

def label_prune_3D(label_cube,errode_num=1,dilation_num=3):
    from scipy.ndimage import morphology
    label_value=np.max(label_cube)
    origin_cube=np.where(label_cube!=0,1,0).astype(np.int32)
    errode_cube=origin_cube.copy()
    for eid in range(0,errode_num):
        errode_cube=morphology.binary_erosion(errode_cube,iterations=1)
        errode_cube=np.where(errode_cube,1,0).astype(np.int32)
        keep_cube=keep_component_3D(errode_cube,keep_num=1)
        keep_cube=np.where(keep_cube,1,0).astype(np.int32)
        dilation_cube=morphology.binary_dilation(keep_cube,iterations=dilation_num*(eid+1))
        dilation_cube=np.where(dilation_cube,1,0).astype(np.int32)
        final_cube=np.where(np.logical_and(origin_cube!=0,dilation_cube!=0),1,0).astype(np.int32)
    final_cube=np.where(final_cube!=0,label_value,0).astype(np.int32)
    return final_cube

def label_prune_3Dlayer(label_cube,vol_thresh=30):
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
    prune_label=keep_component_3D(prune_label,keep_num=1)
    prune_label=np.where(prune_label!=0,label_value,0).astype(np.int32)
    return prune_label

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

#zoom
def zoom_to_shape(x,dst_shape):
    from scipy.ndimage import zoom
    x_shape=x.shape
    zoom_x=zoom(x,zoom=[dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]])
    return zoom_x

def zoom_to_shape_binary(x,dst_shape,thresh=0.5):
    from scipy.ndimage import zoom
    x_value=np.round(np.max(x))
    x_shape=x.shape
    zoom_x=zoom(x,zoom=[dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]])
    zoom_x=(np.where(zoom_x>thresh*x_value,1,0)*x_value).astype(np.int32)
    return zoom_x

def zoom_binary(cube,zoom_rate,thresh=0.5):
    from scipy.ndimage import zoom
    x_value=np.round(np.max(cube))
    return np.where(zoom(cube,zoom=zoom_rate)>thresh,x_value,0).astype(np.int32)

def zoom_segments(x,zoom_rate,thresh=0.5):
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

def zoom_segments_to_shape(x,dst_shape,thresh=0.5):
    x_shape=x.shape
    zoom_rate = [dst_shape[0]/x_shape[0],dst_shape[1]/x_shape[1],dst_shape[2]/x_shape[2]]
    return zoom_segments(x, zoom_rate=zoom_rate, thresh=thresh)

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

#bbx
#bouding box
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

def get_mask_bbx(label_cube,margin=4):
    return get_mask_bbx3D_array(label_cube,margin)

def cal_dice(gt_label,pd_label):
    cal_gt_label=np.where(gt_label!=0,1,0)
    cal_pd_label=np.where(pd_label!=0,1,0)
    intersec=np.logical_and(cal_gt_label,cal_pd_label)
    dice=2*np.sum(intersec)/(np.sum(cal_gt_label)+np.sum(cal_pd_label))
    return dice

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

def pad_shape(image,dst_shape,pad_value=0):
    # pad shape for 3D image
    h,w,l=image.shape
    dst_h,dst_w,dst_l=dst_shape
    pad_image=np.ones(shape=[dst_h,dst_w,dst_l])*pad_value
    crop_h,crop_w,crop_l=min(h,dst_h),min(w,dst_w),min(l,dst_l)
    pad_image[:crop_h,:crop_w,:crop_l]=image[:crop_h,:crop_w,:crop_l]
    return pad_image

def degree(angle):
    return angle/180*np.pi