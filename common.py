# common operation code
from .dependencies import *

def get_mark(mha_path):
    name=os.path.basename(mha_path)
    mark=name[:name.index(".")]
    return mark

def get_name(mha_path):
    file_name=os.path.basename(mha_path)
    name=file_name[:file_name.index(".")]
    return name

def totensor(array_list):
    return map(lambda x: torch.FloatTensor(x),array_list)

def tonumpy(tensor_list):
    return map(lambda x: x.cpu().numpy(),tensor_list)

def tocuda(tensor_list):
    return map(lambda x: x.cuda(),tensor_list)

def make_parent_dir(path,exist_ok=True):
    os.makedirs(os.path.dirname(path),exist_ok=exist_ok)

def init_log(log_path):
    logger=logging.getLogger("train")
    file_handler=logging.FileHandler(filename=log_path,mode="a")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def log(msg):
    logger=logging.getLogger("train")
    logger.info(msg)
    print(msg)
