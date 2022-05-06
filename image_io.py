from .dependencies import *

#read and write images
def imread_float_gray(image_path,max_value=255.0):
    image=np.clip(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)/max_value,0.0,1.0)
    return image

def imwrite_float_gray(image_path,image,max_value=255.0,shape=None):
    image=np.clip(image*max_value,0.0,max_value).astype(np.uint8)
    if(shape!=None):
        shape=tuple(shape)
        image=cv2.resize(image,dsize=shape)
    return cv2.imwrite(image_path,image)

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