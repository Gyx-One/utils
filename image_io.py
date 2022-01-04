from .dependencies import *

#read and write images
def imread_float_gray(image_path,max_value=255.0):
    image=np.clip(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)/max_value,0.0,1.0)
    return image

def imwrite_float_gray(image_path,image,max_value=255.0):
    image=np.clip(image*max_value,0.0,max_value).astype(np.uint8)
    return cv2.imwrite(image_path,image)

#read and write arrays
def read_mha_array3D(image_path,norm=False):
    image_array=itk.GetArrayFromImage(itk.ReadImage(image_path))
    if(norm):
        image_array=image_array/2500
    return image_array.astype(np.float32)

def read_mha_array4D(image_path,norm=False):
    return read_mha_array3D(image_path=image_path,norm=norm)[np.newaxis,:,:,:]

def write_mha_array3D(array,image_path,norm=False):
    if(norm):
        array=array*2500
    itk.WriteImage(itk.GetImageFromArray(array),image_path)

def write_mha_array4D(array,image_path,norm=False):
    write_mha_array3D(array=array[0],image_path=image_path,norm=norm)

#read and write tensors
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

#read and write dicoms
def read_dicom_image(dicom_dir):
    series_ids = itk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    series_file_names = itk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir,series_ids[0])
    series_reader = itk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    cube=itk.GetArrayFromImage(image3D)
    return cube

# repeater
from itertools import repeat
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

# sample
def not_putback_sampling(sample_list,sample_num,ret_ids=False):
    choice_list=[]
    ids_list=[]
    for i in range(sample_num):
        choice_list.append(random.choice(sample_list))
        ids_list.append(sample_list.index(choice_list[-1]))
        sample_list.remove(choice_list[-1])
    if(ret_ids):
        return choice_list,ids_list
    else:
        return choice_list