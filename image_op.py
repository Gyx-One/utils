"""
本文件封装的是2维图像有关的函数, image functions
"""
from .dependencies import *
from .common import *

# interpolation method dicts
intp_dict = {
    "nn":cv2.INTER_NEAREST,
    "cubic":cv2.INTER_CUBIC,
    "linear":cv2.INTER_LINEAR
}

##################################
#  color related functions
##################################
# basic color conversion
def get_color_rgb_uint8_from_hex(rgb_hex_str):
    from PIL import ImageColor
    rgb_uint8_list = ImageColor.getcolor(rgb_hex_str, "RGB")
    return rgb_uint8_list

def get_color_rgb_float_from_uint8(rgb_uint8_list):
    if(np.max(rgb_uint8_list)>255):
        raise NotImplementedError("RGB Uint8 value > 255")
    if(np.min(rgb_uint8_list)<0):
        raise NotImplementedError("RGB Uint8 value < 0")
    rgb_uint8_arr = np.array(rgb_uint8_list)
    rgb_float_arr = np.clip(rgb_uint8_arr/255, 0, 1).astype(np.float32)
    rgb_float_list = rgb_float_arr.tolist()
    return rgb_float_list

def get_color_rgb_float_from_hex(rgb_hex_str):
    rgb_uint8_list = get_color_rgb_uint8_from_hex(rgb_hex_str)
    rgb_float_list = get_color_rgb_float_from_uint8(rgb_uint8_list)
    return rgb_float_list

def get_color_rgb_hex_from_uint8(rgb_uint8_list):
    rgb_hex_str = '#' + ''.join(f'{i:02X}' for i in rgb_uint8_list)
    return rgb_hex_str

def get_color_rgb_uint8_from_float(rgb_float_list):
    if(np.max(rgb_float_list)>1.0):
        raise NotImplementedError("RGB float value > 1.0")
    if(np.min(rgb_float_list)<0.0):
        raise NotImplementedError("RGB float value < 0.0")
    rgb_float_arr = np.array(rgb_float_list)
    rgb_uint8_arr = np.clip(rgb_float_arr*255, 0, 255).astype(np.uint8)
    rgb_unit8_list = rgb_uint8_arr.tolist()
    return rgb_unit8_list

def get_color_rgb_hex_from_float(rgb_float_list):
    rgb_uint8_list = get_color_rgb_uint8_from_float(rgb_float_list)
    rgb_hex_str = get_color_rgb_hex_from_uint8(rgb_uint8_list)
    return rgb_hex_str

def get_color_bgr_uint8_from_hex(rgb_hex_str):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    return get_color_rgb_uint8_from_hex(rgb_hex_str)[::-1]

def get_color_bgr_float_from_hex(rgb_hex_str):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    return get_color_rgb_float_from_hex(rgb_hex_str)[::-1]

def get_color_bgr_float_from_uint8(bgr_uint8_list):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    return get_color_rgb_float_from_uint8(bgr_uint8_list)

def get_color_bgr_hex_from_uint8(bgr_uint8_list):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    rgb_uint8_list = bgr_uint8_list[::-1]
    return get_color_rgb_hex_from_uint8(rgb_uint8_list)

def get_color_bgr_hex_from_float(bgr_float_list):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    rgb_float_list = bgr_float_list[::-1]
    return get_color_rgb_hex_from_float(rgb_float_list)

def get_color_bgr_uint8_from_float(bgr_float_list):
    """
    notes: `hex` always follow `rgb` order, 
           here 'bgr' means the order of `uint8` and `float`
    """
    return get_color_rgb_uint8_from_float(bgr_float_list)

# color N-point array conversion for visualization
def get_color_Npoint_array_rgb_from_float(rgb_float_list, point_num):
    rgb_float_array = np.array(rgb_float_list)
    rgb_npoint_array = np.repeat(rgb_float_array[np.newaxis, :], repeats=point_num, axis=0)
    return rgb_npoint_array

def get_color_Npoint_array_bgr_from_float(bgr_float_list, point_num):
    bgr_float_array = np.array(bgr_float_list)
    bgr_npoint_array = np.repeat(bgr_float_array[np.newaxis, :], repeats=point_num, axis=0)
    return bgr_npoint_array

##################################
#  image io related functions
##################################

# read and write float gray images
def read_image_uint8_gray(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def write_image_uint8_gray(image, image_path, shape=None, interpolation="cubic"):
    if(shape is not None):
        shape=tuple(shape)
        image=cv2.resize(image,dsize=[shape[1], shape[0]], interpolation=intp_dict[interpolation])
    cv2.imwrite(image_path, image)

def read_image_float_gray(image_path, max_value=255.0):
    image = read_image_uint8_gray(image_path)
    image = cvt_image_uint8_to_float(image, max_value=max_value)
    return image

def write_image_float_gray(image, image_path, max_value=255.0, shape=None, interpolation="cubic"):
    make_parent_dir(image_path)
    image = cvt_image_float_to_uint8(image, max_value=max_value)
    return write_image_uint8_gray(image, image_path, shape, interpolation=intp_dict[interpolation])

def write_image_direct_gray(image, image_path):
    cv2.imwrite(image_path, image)

# float color images read/write functions
def read_image_uint8_color(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def write_image_uint8_color(image, image_path, shape=None, interpolation="cubic"):
    if(shape is not None):
        shape=tuple(shape)
        image=cv2.resize(image,dsize=[shape[1], shape[0]], interpolation=intp_dict[interpolation])
    cv2.imwrite(image_path, image)

def read_image_float_color(image_path, max_value=255.0):
    image = read_image_uint8_color(image_path)
    image = cvt_image_uint8_to_float(image, max_value=max_value)
    return image

def write_image_float_color(image, image_path, max_value=255.0, shape=None, interpolation="cubic"):
    make_parent_dir(image_path)
    image = cvt_image_float_to_uint8(image, max_value=max_value)
    return write_image_uint8_color(image, image_path, shape, interpolation=interpolation)

######################################
# image conversion related functions
######################################

# image dtype convert
def cvt_image_float_to_uint8(image, max_value=255):
    image = np.clip(image*max_value, 0, max_value).astype(np.uint8)
    return image

def cvt_image_uint8_to_float(image, max_value=255.0):
    image = np.clip(image.astype(np.float32)/max_value, 0, 1)
    return image

# image color/gray convert functions
def cvt_image_gray_to_color(image):
    return cv2.cvtColor(src=image, code=cv2.COLOR_GRAY2BGR)

def cvt_image_color_to_gray(image):
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

# image color convert functions
def cvt_image_bgr_to_rgb(image_array):
    # color array is of shape[3]
    cvt_image = image_array.copy()
    cvt_image[:, :, 0] = image_array[:, :, 2]
    cvt_image[:, :, 2] = image_array[:, :, 0]
    return cvt_image

def cvt_image_rgb_to_bgr(image_array):
    # color array is of shape[3]
    cvt_image = image_array.copy()
    cvt_image[:, :, 0] = image_array[:, :, 2]
    cvt_image[:, :, 2] = image_array[:, :, 0]
    return cvt_image

def cvt_color_bgr_to_rgb(color_array):
    # color array is of shape[3]
    cvt_color = color_array.copy()
    cvt_color[0] = color_array[2]
    cvt_color[2] = color_array[0]
    return cvt_color

def cvt_color_rgb_to_bgr(color_array):
    # color array is of shape[3]
    cvt_color = color_array.copy()
    cvt_color[0] = color_array[2]
    cvt_color[2] = color_array[0]
    return cvt_color

# image blend functions
def blend_image_with_color_mask(image,mask,blend_color,alpha=0.5):
    # mask [h,w] or [h,w,c] float [0,1]
    # image [h,w,c] float [0,1]
    if(len(mask.shape)!=len(image.shape)):
        c = image.shape[2]
        blend_mask = np.repeat(mask[:,:, np.newaxis], repeats=c, axis=2)
    else:
        blend_mask = mask
    blend_color = blend_mask*blend_color
    blend_image=image*(1-blend_mask)+image*blend_mask*(1-alpha)+blend_color*blend_mask*alpha
    return blend_image

def blend_image_with_image(image_a, image_b, color_a=None, color_b=None, alpha_color_a=0.5, alpha_color_b=0.5, alpha=0.5):
    # image_a, image_b [h,w,c] float [0,1]
    # color_a, color_b [3] float [0,1]
    if(color_a is None):
        colored_image_a = image_a
    else:
        mask_a = np.ones_like(image_a)
        colored_image_a = blend_image_with_color_mask(image=image_a, mask=mask_a, blend_color=color_a, alpha=alpha_color_a)
    
    if(color_b is None):
        colored_image_b = image_b
    else:
        mask_b = np.ones_like(image_b)
        colored_image_b = blend_image_with_color_mask(image=image_b, mask=mask_b, blend_color=color_b, alpha=alpha_color_b)
    
    blend_image = colored_image_a*(1-alpha) + colored_image_b*(alpha)
    return blend_image

def blend_image_float_gray_by_plt_cm(image, cm_type):
    from matplotlib import cm
    plt_cm = cm.get_cmap(cm_type)
    blend_image = plt_cm(image)
    c = blend_image.shape[2]
    blend_image = blend_image[:, :, :c-1] * blend_image[:, :, c-1:]
    blend_image = cvt_image_rgb_to_bgr(blend_image)
    return blend_image

# image contour functions
def get_cv2contours_binary_image_float(binary_image):
    """
    input: binary seg float image
    return: cv2_Contours
    """
    cv2_contours, _ = cv2.findContours(image=cvt_image_float_to_uint8(binary_image),mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return cv2_contours

def draw_cv2contours_image_float(image, contours, color, thickness=1):
    """
    input: float image, cv2_Contours, color [3] in [0,1]
    return: image with contour blended
    """
    if(color is not None):
        color = np.array(color)*255.0
    image = cvt_image_float_to_uint8(image)
    image = cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=color, thickness=thickness)
    return cvt_image_uint8_to_float(image)

def get_contours_mask_binary_image_float(binary_image, thickness=1):
    cur_binary_image = binary_image
    cv2_contours = get_cv2contours_binary_image_float(
        binary_image=cur_binary_image
    )
    shape_h = binary_image.shape[0]
    shape_w = binary_image.shape[1]
    contours_color_mask = np.zeros(shape=[shape_h, shape_w, 3])
    contours_color_mask = draw_cv2contours_image_float(
        image = contours_color_mask,
        contours = cv2_contours,
        color = [0, 1, 0],
        thickness = thickness
    )
    contours_binary_mask = np.max(contours_color_mask, axis=2)
    contours_binary_mask = np.where(contours_binary_mask>1e-1, 1, 0).astype(np.float32)
    return contours_binary_mask

######################################
# embeding functions
######################################
from sklearn import manifold, decomposition
def PCA_embed(x,embed_dim=2):
    print("PCA embedding")
    x_pca=decomposition.TruncatedSVD(n_components=embed_dim).fit_transform(x)
    return x_pca

def Isomap_embed(x,embed_dim=2,n_neighbors=30):
    print("Isomap embedding")
    x_iso=manifold.Isomap(n_components=embed_dim,n_neighbors=n_neighbors).fit_transform(x)
    return x_iso

def LLE_embed(x,embed_dim=2,n_neighbors=30):
    print("LLE embedding")
    x_lle=manifold.LocallyLinearEmbedding(n_components=embed_dim,n_neighbors=n_neighbors).fit_transform(x)
    return x_lle

def MDS_embed(x,embed_dim=2,n_init=1,max_iter=100):
    print("MDS embedding")
    x_mds=manifold.MDS(n_components=embed_dim,n_init=n_init,max_iter=max_iter).fit_transform(x)
    return x_mds

def Spec_embed(x,embed_dim=2,random_state=0,eigen_solver="arpack"):
    print("Spec embedding")
    x_spec=manifold.SpectralEmbedding(n_components=embed_dim,random_state=random_state,eigen_solver=eigen_solver).fit_transform(x)
    return x_spec

def Tsne_embed(x,embed_dim=2,init="pca",random_state=0):
    print("Tsne embedding")
    input_x = copy.copy(x)
    x_tsne=manifold.TSNE(n_components=embed_dim,init=init,random_state=random_state).fit_transform(input_x)
    return x_tsne

def plot_embedding_func(total_label,
                   embed_x,
                   label_name_dic,
                   label_color_dic,
                   save_path,
                   ignore_label_name_list=None,
                   title=""):
    # print("ploting embedding")
    fig,ax=plt.subplots()
    cord_min=np.min(embed_x,axis=0)
    cord_max=np.max(embed_x,axis=0)
    embed_x=(embed_x-cord_min)/(cord_max-cord_min)*1.4
    s_legend=[]
    name_legend=[]
    for label in label_name_dic.keys():
        class_idx = np.where(total_label==label)[0]
        class_x = embed_x[class_idx, 0]
        class_y = embed_x[class_idx, 1]
        class_color = label_color_dic[label]
        class_name = label_name_dic[label]
        if(ignore_label_name_list is not None):
            if(class_name in ignore_label_name_list):
                continue
        class_s=ax.scatter(x=class_x,y=class_y,c=class_color)
        s_legend.append(class_s)
        name_legend.append(class_name)
    plt.legend(s_legend,name_legend,loc="best")
    plt.title(title)
    make_parent_dir(save_path)
    plt.savefig(save_path)
    plt.close()

##################################
#  image shape convert functions
##################################
# image crop functions
def crop_image_array2D(image, center, size, hw_ratio=None):
    """
    Crop 2D image slice based on center and size
    image: 2D array, [h, w, c]
    center: [center_h, center_w]
    size: [size_h, size_w]
    """
    image_h, image_w = image.shape[0], image.shape[1]
    center_h, center_w = center
    size_h, size_w = size
    if(hw_ratio is not None):
        size_w = round_int(size_h/hw_ratio)
    start_h = round_int(max(center_h - size_h//2, 0))
    end_h = round_int(min(start_h + size_h, image_h-1))
    start_w = round_int(max(center_w - size_w//2, 0))
    end_w = round_int(min(start_w + size_w, image_w-1))
    crop_image = image[start_h:end_h, start_w:end_w]
    return crop_image

# image zoom function
def zoom_to_shape_2D(image, shape, interpolation_type="cubic"):
    zoom_image = cv2.resize(image,  dsize=[shape[1], shape[0]], interpolation=intp_dict[interpolation_type])
    return zoom_image


def check_gray_image2D(input_image):
    gray_image_flag = False
    image_dim = len(input_image.shape)
    if(image_dim==2):
        gray_image_flag = True
    elif(image_dim==3):
        gray_image_flag = False
    else:
        raise NotImplementedError(f"Check_gray_image2D: Error Image Dim:{image_dim}")
    return gray_image_flag

def get_unifrom_dim_image2D(input_image):
    """
    Recieve input_image of dim `[h, w, c]` or `[h, w]`,
            output uniform input image of dim `[h, w, c]`.
    Could be pairly used with `get_restore_dim_image2D` to restore the origin image dim
    """
    gray_image_flag = check_gray_image2D(input_image)
    if(gray_image_flag):
        uni_input_image = input_image[:, :, np.newaxis]
    else:
        uni_input_image = input_image
    return uni_input_image, gray_image_flag

def get_restore_dim_image2D(uni_input_image, gray_image_flag):
    """
    Recieve uniform input image of dim `[h, w, c]`.
            output input_image of dim `[h, w, c]` or `[h, w]` according to gray_image_flag
    Could be pairly used with `get_unifrom_dim_image2D`.
    """
    if(gray_image_flag):
        restore_input_image = uni_input_image[:, :, 0]
    else:
        restore_input_image = uni_input_image
    return restore_input_image

def get_pad_image2D(input_image, pad_h, pad_w, pad_value=0):
    image_padder = ImagePadder2D(pad_h=pad_h, pad_w=pad_w, pad_value=pad_value)
    pad_image = image_padder.pad_image(input_image=input_image)
    return pad_image

def get_crop_image2D_by_center(input_image,
                               center_h,
                               center_w,
                               range_h,
                               range_w,
                               crop_type="direct",
                               pad_value=0,
                               print_info_flag=False):
    image_cropper = ImageCropper2D.from_center_and_range(
        center_h=center_h, center_w=center_w, range_h=range_h, range_w=range_w,
        crop_type=crop_type, pad_value=pad_value
    )
    crop_image = image_cropper.crop_image2D(input_image=input_image, print_info_flag=print_info_flag)
    return crop_image

def get_crop_image2D_by_corner(input_image,
                               left_top_h,
                               left_top_w,
                               right_bottom_h,
                               right_bottom_w,
                               crop_type="direct",
                               pad_value=0,
                               print_info_flag=False):
    image_cropper = ImageCropper2D.from_left_top_and_right_bottom(
        left_top_h=left_top_h, left_top_w=left_top_w,
        right_bottom_h=right_bottom_h, right_bottom_w=right_bottom_w,
        crop_type=crop_type, pad_value=pad_value
    )
    crop_image = image_cropper.crop_image2D(input_image=input_image, print_info_flag=print_info_flag)
    return crop_image

class ImagePadder2D:
    def __init__(self,
                 pad_h,
                 pad_w,
                 pad_value=0):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.pad_value = pad_value
    
    def pad_image(self, input_image):
        handle_image, gray_image_flag = get_unifrom_dim_image2D(input_image=input_image)
        
        image_h, image_w, image_c = handle_image.shape
        pad_image_shape_h = round_int(image_h + 2*self.pad_h)
        pad_image_shape_w = round_int(image_w + 2*self.pad_w)
        pad_image = np.ones(
            shape=[pad_image_shape_h, pad_image_shape_w, image_c]
        )*self.pad_value
        
        
        pad_image[
            round_int(self.pad_h):round_int(self.pad_h+image_h), 
            round_int(self.pad_w):round_int(self.pad_w+image_w), 
            :] = handle_image
        
        pad_image = get_restore_dim_image2D(uni_input_image=pad_image, gray_image_flag=gray_image_flag)
        return pad_image

class ImageCropper2D:
    def __init__(self, min_h,
                       max_h,
                       min_w,
                       max_w,
                       crop_type="direct",
                       pad_value=0):
        """
        input_image:
            shape [h, w, c] or [h, w]
        crop_type:
            `direct`: direct crop the image, if the crop range extends cross the image border itself,
            just preserve the overlap region of the image and the crop range.
            `pad`: direct crop the image, if the crop range extends cross than the image border itself,
            pad the extra region of the crop range.
            `fit`: crop the image, if the crop range extends cross than the image border itself, 
            check the overall crop range must be smaller than image, then modify the crop range to make 
            the cross border just fit the image border.
        """
        self.min_h = min_h
        self.max_h = max_h
        self.min_w = min_w
        self.max_w = max_w
        self.crop_type = crop_type
        self.pad_value = pad_value
        assert (self.crop_type in ["direct", "pad", "fit"]), f"Error crop_type:{self.crop_type}"
        
    @classmethod
    def from_center_and_range(
        cls, 
        center_h,
        center_w,
        range_h, 
        range_w,
        crop_type="direct",
        pad_value=0
    ):
        min_h = center_h - range_h//2
        max_h = min_h + range_h
        min_w = center_w - range_w//2
        max_w = min_w + range_w
        image_cropper2d = ImageCropper2D(
            min_h=min_h,
            max_h=max_h,
            min_w=min_w,
            max_w=max_w,
            crop_type=crop_type,
            pad_value=pad_value
        )
        return image_cropper2d
    
    @classmethod
    def from_left_top_and_right_bottom(
        cls,
        left_top_h,
        left_top_w,
        right_bottom_h,
        right_bottom_w,
        crop_type="direct",
        pad_value=0
    ):
        min_h = left_top_h
        min_w = left_top_w
        max_h = right_bottom_h
        max_w = right_bottom_w
        image_cropper2d = ImageCropper2D(
            min_h=min_h,
            max_h=max_h,
            min_w=min_w,
            max_w=max_w,
            crop_type=crop_type,
            pad_value=pad_value
        )
        return image_cropper2d
    
    def crop_direct_image2D(self, input_image, print_info_flag=False):
        # get uniform dim
        handle_image, gray_image_flag = get_unifrom_dim_image2D(input_image=input_image)
        
        # gen crop range
        image_h, image_w, image_c = handle_image.shape
        crop_min_h = int(np.clip(self.min_h, 0, image_h))
        crop_max_h = int(np.clip(self.max_h, 0, image_h))
        crop_min_w = int(np.clip(self.min_w, 0, image_w))
        crop_max_w = int(np.clip(self.max_w, 0, image_w))
        # gen crop image
        crop_image = copy.deepcopy(handle_image)
        # print(f"test type: crop_min_h:{type(crop_min_h)} crop_max_h:{type(crop_max_h)}")
        crop_image = crop_image[crop_min_h:crop_max_h, crop_min_w:crop_max_w, :]
        # restore origin dim
        crop_image = get_restore_dim_image2D(uni_input_image=crop_image, gray_image_flag=gray_image_flag)
        
        # print info
        if(print_info_flag):
            info_msg = "Image Cropper2D-crop_direct: "
            info_msg += f"input_image: {input_image.shape}\t"
            info_msg += f"crop_image: {crop_image.shape}\t"
            info_msg += f"crop_range: [{crop_min_h}:{crop_max_h}, {crop_min_w}:{crop_max_w}]"
            print(info_msg)
                    
        return crop_image
    
    def crop_pad_image2D(self, input_image, print_info_flag=False):
        # get uniform dim
        handle_image, gray_image_flag = get_unifrom_dim_image2D(input_image=input_image)
        
        # gen pad border
        image_h, image_w, image_c = handle_image.shape
        pad_min_h = max(0, 0-self.min_h)
        pad_max_h = max(0, self.max_h - image_h)
        pad_min_w = max(0, 0-self.min_w)
        pad_max_w = max(0, self.max_w - image_w)
        pad_border = np.max([pad_min_h, pad_max_h, pad_min_w, pad_max_w])
        # gen pad image
        pad_image = get_pad_image2D(input_image=handle_image,
                                    pad_h=pad_border,
                                    pad_w=pad_border,
                                    pad_value=self.pad_value)
        # gen crop_pad_range
        crop_pad_min_h = int(self.min_h + pad_border)
        crop_pad_max_h = int(self.max_h + pad_border)
        crop_pad_min_w = int(self.min_w + pad_border)
        crop_pad_max_w = int(self.max_w + pad_border)
        # crop pad image
        crop_pad_image = pad_image[crop_pad_min_h:crop_pad_max_h, crop_pad_min_w:crop_pad_max_w, :]
        # restore origin dim
        crop_pad_image = get_restore_dim_image2D(uni_input_image=crop_pad_image, gray_image_flag=gray_image_flag)
        
        # print info
        if(print_info_flag):
            info_msg = "Image Cropper2D-crop_pad: "
            info_msg += f"input_image: {input_image.shape}\t"
            info_msg += f"crop_image: {crop_pad_image.shape}\t"
            info_msg += f"crop_range: [{round_int(self.min_h)}:{round_int(self.max_h)}, {round_int(self.min_w)}:{round_int(self.max_w)}]"
            print(info_msg)
        
        return crop_pad_image
    
    def crop_fit_image2D(self, input_image, print_info_flag=False):
        # get uniform dim
        handle_image, gray_image_flag = get_unifrom_dim_image2D(input_image)
        # check dim
        image_h, image_w, image_c = handle_image.shape
        crop_range_h = self.max_h - self.min_h
        crop_range_w = self.max_w - self.min_w
        assert (crop_range_h<=image_h), f"crop_range_h={crop_range_h}>image_h={image_h}"
        assert (crop_range_w<=image_w), f"crop_range_w={crop_range_w}>image_w={image_w}"
        
        # gen fit bias
        if(self.min_h<0):
            fit_bias_h = 0-self.min_h
        elif(self.max_h>image_h):
            fit_bias_h = -(self.max_h-image_h)
        else:
            fit_bias_h = 0
        
        if(self.min_w<0):
            fit_bias_w = 0-self.min_w
        elif(self.max_w>image_w):
            fit_bias_w = -(self.max_w-image_w)
        else:
            fit_bias_w = 0
        
        # gen crop    
        crop_min_h = int(self.min_h + fit_bias_h)
        crop_max_h = int(self.max_h + fit_bias_h)
        crop_min_w = int(self.min_w + fit_bias_w)
        crop_max_w = int(self.max_w + fit_bias_w)
        # gen crop image
        crop_image = copy.deepcopy(handle_image)
        crop_image = crop_image[crop_min_h:crop_max_h, crop_min_w:crop_max_w, :]
        # restore origin dim
        crop_image = get_restore_dim_image2D(uni_input_image=crop_image, gray_image_flag=gray_image_flag)
        
        # print info
        if(print_info_flag):
            info_msg = "Image Cropper2D-crop_fit: "
            info_msg += f"input_image: {input_image.shape}\t"
            info_msg += f"crop_image: {crop_image.shape}\t"
            info_msg += f"crop_range: [{crop_min_h}:{crop_max_h}, {crop_min_w}:{crop_max_w}]"
            print(info_msg)
        return crop_image
    
    def crop_image2D(self, input_image, print_info_flag):
        if(self.crop_type=="direct"):
            crop_image = self.crop_direct_image2D(input_image=input_image, print_info_flag=print_info_flag)
        elif(self.crop_type=="pad"):
            crop_image = self.crop_pad_image2D(input_image=input_image, print_info_flag=print_info_flag)
        elif(self.crop_type=="fit"):
            crop_image = self.crop_fit_image2D(input_image=input_image, print_info_flag=print_info_flag)
        else:
            raise NotImplementedError(f"Error crop_type:{self.crop_type}")
        return crop_image

# image grid functions
class ImageGrider:
    def __init__(self, n_row, n_col, row_size):
        self.n_row = n_row
        self.n_col = n_col
        self.row_size = row_size
        self.path_matrix = {}
        for col_idx in range(0, self.n_col):
            self.path_matrix[col_idx] = {}
        self.image_matrix = {}
        for col_idx in range(0, self.n_col):
            self.image_matrix[col_idx] = {}
        self.col_index_list = [0]
    
    def set_path(self, row_idx, col_idx, image_path):
        self.path_matrix[col_idx][row_idx] = image_path
    
    def exam_path_func(self):
        print(f"\nExam Path Matrix")
        print(f"num_row:{self.n_row} num_col:{self.n_col}")
        for row_idx in range(0, self.n_row):
            for col_idx in range(0, self.n_col):
                image_path = self.path_matrix[col_idx][row_idx]
                print(f"\tindex {row_idx}-{col_idx} image_path:{image_path}")
        print("\n")

    def draw_grid(self, grid_path, print_info=False):
        # calculate index
        for col_idx in range(0, self.n_col):
            max_col_size = 0
            for row_idx in range(0, self.n_row):
                image_path = self.path_matrix[col_idx][row_idx]
                image = read_image_float_color(image_path)
                image_h, image_w, _ = image.shape
                zoom_scale = self.row_size/image_h
                zoom_h = self.row_size
                zoom_w = int(round(zoom_scale*image_w))
                zoom_image = cv2.resize(src=image, dsize=[zoom_w, zoom_h], interpolation=intp_dict["cubic"])
                max_col_size = max(max_col_size, zoom_w)
                self.image_matrix[col_idx][row_idx] = zoom_image
            cur_col_index = self.col_index_list[-1] + max_col_size
            self.col_index_list.append(cur_col_index)
        # set image
        grid_image_h = self.n_row * self.row_size
        grid_image_w = self.col_index_list[-1]
        grid_image = np.zeros(shape=[grid_image_h, grid_image_w, 3])
        for col_idx in range(0, self.n_col):
            for row_idx in range(0, self.n_row):
                image = self.image_matrix[col_idx][row_idx]
                src_row_index = row_idx * self.row_size
                dst_row_index = (row_idx + 1)*self.row_size
                src_col_index = self.col_index_list[col_idx]
                dst_col_index = self.col_index_list[col_idx+1]
                dst_col_index = min(dst_col_index, image.shape[1] + src_col_index)
                grid_image[src_row_index:dst_row_index, src_col_index:dst_col_index, :] = image
        make_parent_dir(grid_path)
        write_image_float_color(grid_image, grid_path)

# image edge filter functions
def get_point_dist_matrix_arrayND(point_array):
    # point array shape: [N, M], N is the number of points, M is the dimension of point space
    dist_matrix = np.sqrt(np.sum(np.square(point_array[:, np.newaxis, :] - point_array[np.newaxis, :, :]),  axis=2))
    return dist_matrix

def filter_sobel_x_image2D(image_slice, scale=10, use_abs=True):
    sobel_x_slice = cv2.Sobel(copy.copy(image_slice), cv2.CV_32F, 1, 0, ksize=3)
    if(use_abs):
        sobel_x_slice = np.abs(sobel_x_slice)
    sobel_x_slice = sobel_x_slice/scale
    return sobel_x_slice

def filter_sobel_y_image2D(image_slice, scale=10, use_abs=True):
    sobel_y_slice = cv2.Sobel(copy.copy(image_slice), cv2.CV_32F, 0, 1, ksize=3)
    if(use_abs):
        sobel_y_slice = np.abs(sobel_y_slice)
    sobel_y_slice = sobel_y_slice/scale
    return sobel_y_slice

def filter_sobel_all_image2D(image_slice, scale=10, use_abs=True):
    # sobel x
    sobel_x_slice = filter_sobel_x_image2D(image_slice=image_slice, scale=scale, use_abs=use_abs)
    # sobel y
    sobel_y_slice = filter_sobel_y_image2D(image_slice=image_slice, scale=scale, use_abs=use_abs)
    # sobel all
    sobel_all_slice = np.sqrt(sobel_x_slice**2 + sobel_y_slice**2)
    return sobel_all_slice

def get_frontdot_from_filter_image2D(filter_image_slice, 
                                     frontdot_value_thresh=0.04, 
                                     frontdot_h_range_scale=1,
                                     frontdot_w_range_scale=0.5,
                                     frontdot_dist_thresh=4, 
                                     use_dist_filter=True):
    shape_h, shape_w = filter_image_slice.shape
    frontdot_list = []
    # scan h slice
    frontdot_h_range = round_int(shape_h * frontdot_h_range_scale)
    for slice_h_idx in range(0, frontdot_h_range):
        slice_h = filter_image_slice[slice_h_idx, :]
        valid_w_idx_list = np.where(slice_h>frontdot_value_thresh)[0]
        if(len(valid_w_idx_list)==0):
            continue
        frontdot_h = slice_h_idx
        frontdot_w = np.min(valid_w_idx_list)
        frontdot_list.append([round_int(frontdot_h), round_int(frontdot_w)])
    # scan w slice
    frontdot_w_range = round_int(shape_w * frontdot_w_range_scale)
    for slice_w_idx in range(0, frontdot_w_range):
        slice_w = filter_image_slice[:, slice_w_idx]
        valid_h_idx_list = np.where(slice_w>frontdot_value_thresh)[0]
        if(len(valid_h_idx_list)==0):
            continue
        frontdot_h = np.max(valid_h_idx_list)
        frontdot_w = slice_w_idx
        frontdot_list.append([round_int(frontdot_h), round_int(frontdot_w)])
    # filter frontdot with too long distance
    if(use_dist_filter):
        if(len(frontdot_list)!=0):
            frontdot_array = np.array(frontdot_list)
            frontdot_dist_matrix = get_point_dist_matrix_arrayND(frontdot_array)
            # ignore select dist from self
            frontdot_dist_matrix += np.eye(N=frontdot_dist_matrix.shape[0])*1000
            sort_frontdot_dist_matrix = np.sort(frontdot_dist_matrix, axis=1)
            sort_frontdot_dist_matrix = sort_frontdot_dist_matrix[:, :8]
            frontdot_min_dist = np.mean(sort_frontdot_dist_matrix, axis=1)
            preserve_index = np.where(frontdot_min_dist<frontdot_dist_thresh)[0]
            frontdot_list = frontdot_array[preserve_index].tolist()
    return frontdot_list