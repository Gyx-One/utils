"""
本文件包含的是通用的函数, 例如输入输出, 日志, 文件路径, 并行计算等有关的函数 common functions
"""
# common operation code
from .dependencies import *
import sys
import json
import time
import shutil
from multiprocessing import Pool

_cur_file_path = os.path.abspath(__file__)
st_utils_dir = os.path.abspath(os.path.dirname(_cur_file_path))

#################################
# common print functions
#################################
def print_dict(data_dict, name=""):
    msg = name
    for key in data_dict.keys():
        value = data_dict[key]
        msg += f" {key}:{value}"
    return msg

def get_format_dict_str(input_dict, indent=4):
    import json
    format_dict_str = json.dumps(input_dict, indent=indent)
    return format_dict_str

def get_format_list_str(input_list, indent=4):
    import json
    format_list_str = json.dumps(input_list, indent=indent)
    return format_list_str

def get_number_format_str(float_number, digit=10):
    format_str = "{:+."+f"{digit}"+"}"
    return format_str.format(float(float_number))

def num_str(float_number, digit=2):
    return get_number_format_str(float_number=float_number, digit=digit)

def num_str_2(float_number, digit=2):
    return get_number_format_str(float_number=float_number, digit=digit)

def num_str_3(float_number, digit=3):
    return get_number_format_str(float_number=float_number, digit=digit)

#################################
#      common log functions     
#################################
# log functions
def init_log(log_path):
    make_parent_dir(log_path)
    for logger_name in ["Model","Dataset","Train"]:
        logger=logging.getLogger(logger_name)
        if(len(logger.handlers)==0):
            file_handler=logging.FileHandler(filename=log_path,mode="a")
            stream_handler = logging.StreamHandler()
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        
def log(msg, add_time=True):
    logger=logging.getLogger("Train")
    log_msg = msg
    if(add_time):
        log_msg = f"[{get_cur_time_str()}] {log_msg}"
    logger.info(log_msg)

##################################
# common set random seed functions
##################################
# set random seed
def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    return True

#################################
# common path functions
#################################

# path name functions
def get_mark(mha_path):
    name=os.path.basename(mha_path)
    mark=name[:name.index(".")]
    return mark

def get_name(mha_path):
    file_name=os.path.basename(mha_path)
    name=file_name[:file_name.index(".")]
    return name

def get_name_list(path_list):
    name_list = []
    for path in path_list:
        name = get_name(path)
        name_list.append(name)
    return name_list

def get_suffix(mha_path):
    file_name=os.path.basename(mha_path)
    suffix=file_name[file_name.index(".")+1:]
    return suffix

def get_origin_name_from_aug_name(aug_name, name_split_layer=1):
    origin_name = "_".join(aug_name.split("_")[:-name_split_layer])
    return origin_name

# abs path function
def get_abs_path(relative_path):
    return os.path.abspath(relative_path)

# link path source function
def get_real_path(link_path):
    return os.path.realpath(link_path)

def exam_path(path):
    """
    如果路径存在, 输出True, 不存在则输出False
    """
    abs_path = os.path.abspath(path)
    exist_flag = os.path.exists(abs_path)
    return exist_flag

def file_copy_func(src_path, dst_path, make_dirs=True):
    if(make_dirs):
        make_parent_dir(dst_path)
    shutil.copy(src_path, dst_path)


def remove_path_by_os_func(remove_path):
    abs_remove_path = os.path.abspath(remove_path)
    ret = os.system(f"rm -rf {abs_remove_path}")
    return ret 

def remove_path_list_by_os_func(remove_path_list):
    ret_list = []
    for remove_path in remove_path_list:
        ret = remove_path_by_os_func(remove_path)
        ret_list.append(ret)
    return ret_list


#################################
#      common dir functions
#################################
# directory functions
def make_dir(path, exists_ok=True):
    os.makedirs(path, exist_ok=exists_ok)
    
def make_dir_list(path_list, exist_ok=True):
    for path in path_list:
        make_dir(path=path, exist_ok=exist_ok)
        
def get_parent_dir(path):
    return os.path.dirname(path)

def make_parent_dir(path,exist_ok=True):
    os.makedirs(os.path.dirname(path),exist_ok=exist_ok)

def make_parent_dir_list(path_list,exist_ok=True):
    for path in path_list:
        make_parent_dir(path)
        
# get sub-directory function
def get_sub_dirs(current_dir):
    sub_dir_list=[]
    for sub_path in glob.glob(f"{current_dir}/*"):
        if(os.path.isdir(sub_path)):
            sub_dir_list.append(sub_path)
    return sub_dir_list

def dir_copy_func(src_dir, dst_dir, make_dirs=True, exists_ok=False):
    if(make_dirs):
        make_parent_dir(dst_dir)
    if(exists_ok):
        if(os.path.exists(dst_dir)):
            remove_dir_by_os_func(dst_dir)
    shutil.copytree(src_dir, dst_dir)

def remove_dir_by_os_func(remove_dir):
    abs_remove_dir = os.path.abspath(remove_dir)
    ret = os.system(f"rm -rf {abs_remove_dir}")
    return ret 

def os_soft_link_func(
    src_path,
    dst_path,
    use_abs_path_flag=True,
    make_parent_dir_flag=True
):
    abs_src_path = src_path
    abs_dst_path = dst_path
    if(use_abs_path_flag):
        abs_src_path = get_abs_path(src_path)
        abs_dst_path = get_abs_path(dst_path)
    if(make_parent_dir_flag):
        make_parent_dir(abs_dst_path)
    ret = os.system(f"ln -s {abs_src_path} {abs_dst_path}")
    return ret

#################################
# common  multiprocess functions
#################################
# multiprocessing acceleration
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
        ret_list = []
        try:
            result_list = []
            for data_idx in range(0, len(data_list)):
                result_list.append(process_pool.apply_async(process_func, data_list[data_idx:data_idx+1]))
            print("[Waiting Subprocess]")       
            for res_idx in tqdm(range(0, len(result_list))):
                ret_value = result_list[res_idx].get()
                print(f"Task {res_idx}/{data_num} return: ", end="")
                print(ret_value)
                ret_list.append(ret_value)
        # handle ctrl+c sigint
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            process_pool.terminate()
            sys.exit(1)
        else:        
            process_pool.close()
            process_pool.join()
        print("[Subprocess finished]")
        return ret_list

def single_process_exec(process_func, process_num, data_list, check_data_list=False):
    ret_list = []
    for data in tqdm(data_list):
        ret = process_func(data)
        ret_list.append(ret)
    return ret_list

def multi_process_exec(process_func, process_num, data_list, check_data_list=False):
    processor = MultiProcess_Processor(process_num=process_num)
    return processor.exec_func(process_func=process_func, data_list=data_list, check_data_list=check_data_list)


##################################
# common small data file functions
#################################
# json file write/load functions
def common_json_dump(json_object, file_path, make_parent_dir_flag=True):
    if(make_parent_dir_flag):
        make_parent_dir(file_path)
    json_file = open(file_path, mode="w")
    json.dump(json_object, json_file)
    json_file.close()

def common_json_load(file_path):
    json_file = open(file_path, mode="r")
    json_object = json.load(json_file)
    json_file.close()
    return json_object

# numpy array write/load functions 
def common_array_dump(np_ndarray, file_path):
    return np.savetxt(file_path, np_ndarray)

def common_array_load(file_path):
    return np.loadtxt(fname=file_path)




###################################
# common dataset process functions
###################################
# data sample function
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

# data split function
def dataset_split_func(data_paths,
                       split_subset_name_list=["train", "val", "test"],
                       split_subset_num_list=[15, 5, 10]
                       ):
    import copy
    left_data_paths = copy.copy(data_paths)
    assert len(split_subset_name_list)==len(split_subset_num_list)
    assert len(left_data_paths)>=sum(split_subset_num_list)
    split_subset_paths_list = []
    for subset_name, subset_num in zip(split_subset_name_list, split_subset_num_list):
        subset_paths = not_putback_sampling(sample_list=left_data_paths, sample_num=subset_num)
        split_subset_paths_list.append(subset_paths)
        print(f"splitting {subset_name}: {len(subset_paths)}")
    return split_subset_paths_list
    
# dataloader repeater functions
from itertools import repeat
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

# data filter function
def filter_func(path_list, filter_list):
    filter_path_list = []
    for path in path_list:
        select_flag = False
        for filter_name in filter_list:
            if(filter_name in path):
                select_flag = True
                break
        if(select_flag):
            filter_path_list.append(path)
    print(f"path num before filter:{len(path_list)} after_filter:{len(filter_path_list)}")
    return filter_path_list

# data exclude function
def exclude_list_func(input_name_list, exclude_name_list, print_detail_info_flag=False):
    filter_name_list = []
    for name in input_name_list:
        if(name in exclude_name_list):
            continue
        else:
            filter_name_list.append(name)
    if(print_detail_info_flag):
        print(f"Exclude_list_func:")
        print(f"input_name_list: {len(input_name_list)}\n\t{input_name_list}")
        print(f"exclude_name_list: {len(exclude_name_list)}\n\t{exclude_name_list}")
        print(f"filter_name_list: {len(filter_name_list)}\n\t{filter_name_list}")
        
    return filter_name_list


##################################
#      common time functions
##################################
def get_cur_time_str(format="%Y-%m-%d %H:%M:%S"):
    cur_time = time.localtime()
    cur_time_str = time.strftime(format, cur_time)
    return cur_time_str


##################################
#     common math functions
##################################
def round_int(x):
    return int(round(x))

def get_clip_value_list(value_list, clip_min, clip_max):
    value_array = np.array(value_list)
    clip_value_array = np.clip(value_array, a_min=clip_min, a_max=clip_max)
    clip_value_list = clip_value_array.tolist()
    return clip_value_list

def get_split_interval_list(min_x, max_x, split_num, margin=0):
    split_min_x = min_x-margin
    split_max_x = max_x+margin
    split_int = (split_max_x - split_min_x)/split_num
    split_anchor_list = np.arange(split_min_x, split_max_x, split_int).tolist() + [split_max_x]
    split_int_list = []
    for split_idx in range(0, len(split_anchor_list)-1):
        split_int_start = split_anchor_list[split_idx]
        split_int_end = split_anchor_list[split_idx+1]
        split_int_list.append([split_int_start, split_int_end])
    return split_int_list

##################################
# common data type cvt functions
##################################
def numpy_to_list(numpy_list):
    cvt_list = np.array(numpy_list).tolist()
    return cvt_list

def get_array_from_list(data_list):
    return np.array(data_list)

def get_list_from_array(data_array):
    return numpy_to_list(data_array)