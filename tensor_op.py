from .dependencies import *

def get_full_kernel3():
    return torch.ones(size=[1,1,3,3,3])

def get_register_inv_tensor(x):
    batch_size=x.shape[0]
    trans_x=torch.zeros_like(x)
    for batch_idx in range(0,batch_size):
        ori_cube=x[batch_idx,0,:,:,:].detach().cpu().numpy()
        ori_image=itk.GetImageFromArray(ori_cube)
        inv_cube=ori_cube[:,:,::-1]
        inv_image=itk.GetImageFromArray(inv_cube)
        init_transform = itk.CenteredTransformInitializer(ori_image, inv_image, itk.Euler3DTransform(), itk.CenteredTransformInitializerFilter.GEOMETRY)
        regis=itk.ImageRegistrationMethod()
        regis.SetMetricAsMeanSquares()
        regis.SetInterpolator(itk.sitkLinear)
        regis.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-2,numberOfIterations=100)
        regis.SetOptimizerScalesFromPhysicalShift()
        regis.SetInitialTransform(init_transform)
        transform=regis.Execute(fixed=ori_image,moving=inv_image)
        transed_image=itk.Resample(inv_image,ori_image,transform,itk.sitkLinear,(-999.0+1000)/2300,inv_image.GetPixelID())
        trans_cube=itk.GetArrayFromImage(transed_image)
        trans_x[batch_idx,0,:,:,:]=torch.FloatTensor(trans_cube)
    trans_x.requires_grad=False
    return trans_x

def get_var(x):
    #conv x
    conv_x_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_x_weight[0,0,0,1,1]=-1
    conv_x_weight[0,0,2,1,1]=1
    grad_x=F.conv3d(input=x,weight=conv_x_weight,bias=None,stride=1,padding=1)
    #conv y
    conv_y_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_y_weight[0,0,1,0,1]=-1
    conv_y_weight[0,0,1,2,1]=1
    grad_y=F.conv3d(input=x,weight=conv_y_weight,bias=None,stride=1,padding=1)
    #conv z
    conv_z_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_z_weight[0,0,1,1,0]=-1
    conv_z_weight[0,0,1,1,2]=1
    grad_z=F.conv3d(input=x,weight=conv_z_weight,bias=None,stride=1,padding=1)
    #conv norm
    grad_norm=(grad_x**2+grad_y**2+grad_z**2)**0.5
    x_var=torch.stack([grad_x,grad_y,grad_z,grad_norm],dim=1)
    return x_var

#bouding box
def get_mask_bbx3D_tensor(label_cube,margin=4):
    #get bounding box of the binary mask
    #input [h,w,l] tensor
    mask=torch.zeros_like(label_cube)
    if(torch.sum(label_cube)!=0):
        img_h,img_w,img_l=label_cube.shape
        hs,ws,ls=torch.where(label_cube>0)
        mask_hmin=torch.clip(torch.min(hs)-margin,0,img_h)
        mask_hmax=torch.clip(torch.max(hs)+margin,0,img_h)
        mask_wmin=torch.clip(torch.min(ws)-margin,0,img_w)
        mask_wmax=torch.clip(torch.max(ws)+margin,0,img_w)
        mask_lmin=torch.clip(torch.min(ls)-margin,0,img_l)
        mask_lmax=torch.clip(torch.max(ls)+margin,0,img_l)
        mask[mask_hmin:mask_hmax,mask_wmin:mask_wmax,mask_lmin:mask_lmax]=1
    mask=mask.astype(torch.float32)
    return mask

def get_mask_bbx4D_tensor(bin_mask,pad_h=5,pad_w=5,pad_l=5):
    #get bounding box of the binary mask
    #input [c,h,w,l] tensor
    mask=torch.zeros_like(bin_mask)
    if(torch.sum(bin_mask>0)!=0):
        _,h,w,l=bin_mask.shape
        _,hs,ws,ls=torch.where(bin_mask>0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        min_h=torch.clip(min_h-pad_h,0,h)
        max_h=torch.clip(max_h+pad_h,0,h)
        min_w=torch.clip(min_w-pad_w,0,w)
        max_w=torch.clip(max_w+pad_w,0,w)
        min_l=torch.clip(min_l-pad_l,0,l)
        max_l=torch.clip(max_l+pad_l,0,l)
        mask[:,min_h:max_h,min_w:max_w,min_l:max_l]=1
    return mask

def get_boundry_bbx4D_tensor(bbx_mask):
    """
    get the boundry of the given bbx_mask

    Args:
        bbx_mask (tensor): shape [c,h,w,l], binary.
    
    Returns:
        min_h (tensor): min_h.
        max_h (tensor): max_h.
        min_w (tensor): min_w.
        max_w (tensor): max_w.
        min_l (tensor): min_l.
        max_l (tensor): max_l.
    """
    _,hs,ws,ls=torch.where(bbx_mask>0)
    min_h,max_h=torch.min(hs),torch.max(hs)
    min_w,max_w=torch.min(ws),torch.max(ws)
    min_l,max_l=torch.min(ls),torch.max(ls)
    return min_h,max_h,min_w,max_w,min_l,max_l

def get_local_patch(x,mask,patch_h=64,patch_w=64,patch_l=64):
    #get local patch of the binary mask
    #input [c,h,w,l] tensor
    _,h,w,l=mask.shape
    if(torch.sum(mask)!=0):
        _,hs,ws,ls=torch.where(mask!=0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        start_h=min_h
        start_w=min_w
        start_l=min_l
        if(start_h+patch_h>=h):
            start_h=max_h-patch_h
        if(start_w+patch_w>=w):
            start_w=max_w-patch_w
        if(start_l+patch_l>=l):
            start_l=max_l-patch_l
    else:
        start_h=h-patch_h//2
        start_w=w-patch_w//2
        start_l=l-patch_l//2
    patch_x=x[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    patch_mask=mask[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    return patch_x,patch_mask

def is_tensor(x):
    return type(x) == torch.FloatTensor or type(x) == torch.LongTensor or type(x) == torch.Tensor

def to_tensor_list(number_list):
    tensor_list = [torch.FloatTensor([number]) for number in number_list]
    return tensor_list

def get_grid_tensor(shape):
    '''
    Parameters:
        shape: shape [3], (h,w,l) of image thats need grid
    Returns:
        grid: shape [dim,h,w,l], dim=3
    '''
    vectors = [torch.arange(0, s) for s in shape]
    grid = torch.meshgrid(vectors)
    grid = torch.stack(grid)
    grid = grid.float()
    return grid

def get_rigid_matrix_from_param(trans_x,trans_y,trans_z,roll,pitch,yaw,center_x=0,center_y=0,center_z=0):
    '''
    Parameters:
        trans_x: shape [1]
        trans_y: shape [1]
        trans_z: shape [1]
        roll: shape [1]
        pitch: shape [1]
        yaw: shape [1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    # input trans_x,trans_y,trans_z,roll,pitch,yaw
    # output rigid matrix of shape [4,4]
    if(not is_tensor(trans_x)):
        trans_x,trans_y,trans_z,roll,pitch,yaw = to_tensor_list([trans_x,trans_y,trans_z,roll,pitch,yaw])
    from torch import cos,sin
    device = trans_x.device
    # center
    Rc = torch.zeros(size=[4,4]).to(device)
    Rc[0,0] = 1
    Rc[1,1] = 1
    Rc[2,2] = 1
    Rc[0,3] = -center_x
    Rc[1,3] = -center_y
    Rc[2,3] = -center_z
    Rc[3,3] = 1
    
    # rotation
    # Rx
    Rx=torch.zeros(size=[4,4]).to(device)
    Rx[0,0]=1
    Rx[1,1]=cos(roll)
    Rx[1,2]=sin(roll)
    Rx[2,1]=-sin(roll)
    Rx[2,2]=cos(roll)
    Rx[3,3]=1
    # Ry
    Ry=torch.zeros(size=[4,4]).to(device)
    Ry[0,0]=cos(pitch)
    Ry[0,2]=-sin(pitch)
    Ry[1,1]=1
    Ry[2,0]=sin(pitch)
    Ry[2,2]=cos(pitch)
    Ry[3,3]=1
    # Rz
    Rz=torch.zeros(size=[4,4]).to(device)
    Rz[0,0]=cos(yaw)
    Rz[0,1]=sin(yaw)
    Rz[1,0]=-sin(yaw)
    Rz[1,1]=cos(yaw)
    Rz[2,2]=1
    Rz[3,3]=1
    # combine
    Rr = Rx@Ry@Rz
    
    # transformation
    Rt = torch.zeros(size=[4,4]).to(device)
    Rt[0,0] = 1
    Rt[1,1] = 1
    Rt[2,2] = 1
    Rt[0,3] = trans_x
    Rt[1,3] = trans_y
    Rt[2,3] = trans_z
    Rt[3,3] = 1

    # de-center
    Rdc = torch.zeros(size=[4,4]).to(device)
    Rdc[0,0] = 1
    Rdc[1,1] = 1
    Rdc[2,2] = 1
    Rdc[0,3] = center_x
    Rdc[1,3] = center_y
    Rdc[2,3] = center_z
    Rdc[3,3] = 1
        
    # total
    R = Rt@Rdc@Rr@Rc
    return R

def get_rigid_matrix_from_param_batch(trans_x,trans_y,trans_z,roll,pitch,yaw):
    '''
    Parameters:
        trans_x: shape [n,1]
        trans_y: shape [n,1]
        trans_z: shape [n,1]
        roll: shape [n,1]
        pitch: shape [n,1]
        yaw: shape [n,1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    batch_size = trans_x.shape[0]
    rigid_matrix = torch.zeros(size=[batch_size,4,4]).to(trans_x.device)
    for batch_idx in range(0,batch_size):
        rigid_matrix[batch_idx,:,:]=get_rigid_matrix_from_param(trans_x[batch_idx], trans_y[batch_idx], trans_z[batch_idx],\
            roll[batch_idx], pitch[batch_idx], yaw[batch_idx])
    return rigid_matrix

def get_deform_from_rigid_matrix(rigid_matrix,shape):
    '''
    Parameters:
        rigid_matrix: shape [4,4], homo transform
        shape: shape [3], shape of image
    Returns:
        deform: shape [dim,h,w,l], output deformation field 
    '''
    h,w,l = shape
    dim = len(shape)
    device = rigid_matrix.device
    # create sampling grid of shape [h,w,l,dim]
    grid =  get_grid_tensor(shape=shape).to(device).permute(1,2,3,0)
    # create deform of shape [h,w,l,dim+1], transformation
    coor = torch.zeros(size=[h,w,l,dim+1]).to(device)
    coor[:,:,:,:dim] = grid
    coor[:,:,:,dim:] = 1
    trans_coor = torch.matmul(coor[:,:,:,np.newaxis,:],rigid_matrix.transpose(0,1))[:,:,:,0,:]
    # construct deformation field
    deform = trans_coor[:,:,:,:dim]
    deform = deform - grid
    deform = deform.permute(3,0,1,2)
    return deform

def get_deform_from_rigid_matrix_batch(rigid_matrix, shape):
    '''
    Parameters:
        rigid_matrix: shape [n,4,4], homo transform
        shape: shape [3], shape of image
    Returns:
        deform: shape [n,dim,h,w,l], output deformation field 
    '''
    h,w,l = shape
    dim = len(shape)
    device = rigid_matrix.device
    batch_size = rigid_matrix.shape[0]
    deform = torch.zeros(size=[batch_size,dim,h,w,l]).to(device)
    for batch_idx in range(0, batch_size):
        deform[batch_idx] = get_deform_from_rigid_matrix(rigid_matrix[batch_idx], shape)
    return deform

def spatial_transform(image, deform, mode="bilinear"):
    '''
    Parameters:
        image: shape [n,c,h,w,l], image to be transform
        deform: shape [n,dim,h,w,l], deformation field
    Returns:
        trans_image: shape [n,c,h,w,l], transformed image
    '''
    image = torch.unsqueeze(image, dim=0)
    deform = torch.unsqueeze(deform, dim=0)
    shape = image.shape[2:]
    return spatial_transform_batch(image, deform, mode=mode)[0]

def spatial_transform_batch(image, deform, mode="bilinear"):
    '''
    Parameters:
        image: shape [n,c,h,w,l], image to be transform
        deform: shape [n,dim,h,w,l], deformation field
    Returns:
        trans_image: shape [n,c,h,w,l], transformed image
    '''
    shape = image.shape[2:]
    grid = get_grid_tensor(shape).to(image.device)
    new_locs = grid + deform
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    
    # adapt to nn.functional.grid_sample
    # move channels dim to last position
    # also not sure why, but the channels need to be reversed
    new_locs = new_locs.permute(0, 2, 3, 4, 1)
    new_locs = new_locs[..., [2, 1, 0]]
    return nnf.grid_sample(image, new_locs, align_corners=True, mode=mode)
    