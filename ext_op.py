"""
本文件包含的是调用外部程序封装成的API函数, external functions
"""

from .dependencies import *
from .common import *

def external_segment_func_pytorch_unet3d(
    model_name,
    abs_config_path,
    abs_checkpoint_dir,
    abs_input_image_dir,
    abs_output_label_dir,
    device=0
):
    """
    Using command line to construct trained pytorch-3dunet model for inference. 
    
        `name`: determine the `abs_checkpoint_dir` is abs_checkpoint_dir is not given.

        `abs_config_path`: the absolute path of pytorch-3dunet config yaml path.

        `abs_checkpoint_dir`: the absolute path of directory that contains `best_checkpoint.pytorch`
        
        `abs_input_dir`: the absolute path of directory that contains input images.
        
        `abs_output_dir`: the absolute path of directory that contains output predicted label.
    """
    abs_cur_code_dir = get_abs_path("./")
    abs_external_code_dir = get_abs_path(f"{st_utils_dir}/../../../../help/help_seg/code")
    os.chdir(abs_external_code_dir)
    sys_cmd = f"python predict_script.py --device={device}"
    sys_cmd += f" --clean --prepare --infer --convert_output --clean_output_dir"
    sys_cmd += f" --name={model_name}"
    sys_cmd += f" --config_path={abs_config_path}"
    sys_cmd += f" --checkpoint_dir={abs_checkpoint_dir}"
    sys_cmd += f" --input_dir={abs_input_image_dir}"
    sys_cmd += f" --final_output_dir={abs_output_label_dir}"
    ret_code = os.system(sys_cmd)
    os.chdir(abs_cur_code_dir)
    
    return ret_code