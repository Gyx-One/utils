# common operation code
import math
import sys

from py import process

from utils.tensor_op import get_rigid_matrix_from_param
from .dependencies import *
from scipy.spatial.transform import Rotation


# path functions
def get_mark(mha_path):
    name=os.path.basename(mha_path)
    mark=name[:name.index(".")]
    return mark

def get_name(mha_path):
    file_name=os.path.basename(mha_path)
    name=file_name[:file_name.index(".")]
    return name

def get_suffix(mha_path):
    file_name=os.path.basename(mha_path)
    suffix=file_name[file_name.index(".")+1:]
    return suffix

def make_parent_dir(path,exist_ok=True):
    os.makedirs(os.path.dirname(path),exist_ok=exist_ok)

def make_parent_dir_list(path_list,exist_ok=True):
    for path in path_list:
        make_parent_dir(path)

# print functions
def print_dict(data_dict, name=""):
    msg = name
    for key in data_dict.keys():
        value = data_dict[key]
        msg += f" {key}:{value}"
    return msg

# log functions
def init_log(log_path):
    for logger_name in ["Model","Dataset","Train"]:
        logger=logging.getLogger(logger_name)
        file_handler=logging.FileHandler(filename=log_path,mode="a")
        stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

def log(msg):
    logger=logging.getLogger("Train")
    logger.info(msg)

# math functions
def rad(angle):
    return angle/180*np.pi

# common geometry ops
def get_euler_from_rotation_matrix(rot_matrix, use_rad=True):
    use_degree = not use_rad
    matrix = copy.deepcopy(rot_matrix)
    rotation = Rotation.from_matrix(matrix[:3,:3])
    euler = rotation.as_euler(seq="zyx", degrees=use_degree)
    roll = -euler[2]
    pitch = -euler[1]
    yaw = -euler[0]
    return roll, pitch, yaw

def get_param_from_rigid_matrix(rigid_matrix,
                                center_x=0,
                                center_y=0,
                                center_z=0,
                                use_rad=True,
                                debug=False):
    matrix = copy.deepcopy(rigid_matrix)
    center = np.array([center_x, center_y, center_z])
    # trans
    trans = matrix[:3, 3]
    center = np.array(center)
    trans = trans - center + matrix[:3,:3]@center
    trans_x = trans[0]
    trans_y = trans[1]
    trans_z = trans[2]
    # eulers
    rot_matrix = matrix[:3,:3]
    eulers = get_euler_from_rotation_matrix(rot_matrix, use_rad=use_rad)
    roll = eulers[0]
    pitch = eulers[1]
    yaw = eulers[2]
    if(debug):
        print("")
        print(f"test get_param_from_rigid_matrix")
        print(f"type:{type(center)} center:{center}")
        print(f"origin matrix:\n{rigid_matrix}")
        compose_matrix = get_rigid_matrix_from_param(
            trans_x = trans_x,
            trans_y = trans_y,
            trans_z = trans_z,
            roll = torch.FloatTensor([roll/180*np.pi]),
            pitch = torch.FloatTensor([pitch/180*np.pi]),
            yaw = torch.FloatTensor([yaw/180*np.pi]),
            center_x = float(center[0]),
            center_y = float(center[1]),
            center_z = float(center[2])
        )
        print(f"compose matrix:\n{compose_matrix}")
    # param dict
    param_dict = {
        "trans_x" : trans_x,
        "trans_y" : trans_y,
        "trans_z" : trans_z,
        "roll" : roll,
        "pitch" : pitch,
        "yaw" : yaw,
        "matrix" : matrix.tolist()
    }
    return param_dict


import signal
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

class MultiProcess_Processor:
    def __init__(self, process_num=10):
        self.process_num = process_num
    
    def test_func(self, process_func, data_list, check_data_list=False):
        print("[Test func]")
        if(check_data_list):
            print("[Check data_list]")
            print(data_list[0])
        return process_func(data_list[0])
    
    def exec_func(self, process_func, data_list, check_data_list=False):
        """

        Args:
            func : the function to be execute, notes that the args of the func should be a datalist
            data_list : datalist to be execute
        """
        

        print("[MultiProcess Processor]")
        data_num = len(data_list)
        if(check_data_list):
            print("[Check data_list]")
            for data_idx, data in enumerate(data_list):
                print(f"data idx:{data_idx} data:{data}")
            print("")
        print(f"process num:{self.process_num} data_num:{data_num}")
        process_pool = Pool(processes=self.process_num)
        # try 
        try:
            result_list = []
            for data_idx in range(0, len(data_list)):
                result_list.append(process_pool.apply_async(process_func, data_list[data_idx:data_idx+1]))
            print("[Waiting Subprocess]")       
            for res_idx in range(0, len(result_list)):
                print(f"Task {res_idx} return: ", end="")
                print(result_list[res_idx].get())
        # handle ctrl+c sigint
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            process_pool.terminate()
            sys.exit(1)
        else:        
            process_pool.close()
            process_pool.join()
        print("[Subprocess finished]")
        