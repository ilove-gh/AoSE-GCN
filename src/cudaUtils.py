import torch
from .LoggerFactory import get_logger

logger = get_logger()


def all_cuda_infos():
    """
    View all current cuda device information
    :return:
    """
    if not cuda_is_available():
        logger.error(
            'The current device CUDA is not available,The return value of the function torch.cuda.is_available is false')
        return
    # Gets the number of Gpus available in the system
    gpu_num = torch.cuda.device_count()
    logger.info("The number of GPUs available to the device is {}.".format(gpu_num))

    # 遍历所有GPU
    for index in range(gpu_num):
        # Gets the name of the GPU
        gpu_name = torch.cuda.get_device_name(index)
        # Gets the available memory size (in bytes) of the GPU.
        gpu_total_memory = torch.cuda.get_device_properties(index).total_memory
        logger.info(
            "The number of GPUs on the device is {}, and the information in block {} is:device_name={},device_memory={} GB."
            .format(gpu_num, index, gpu_name, round(gpu_total_memory / 1024 ** 3)))


def current_cuda_info():
    """
    View the current cuda device information
    :return:
    """
    if not cuda_is_available():
        logger.error(
            'The current device CUDA is not available,The return value of the function torch.cuda.is_available is false')
        return
    # Gets the number of Gpus available in the system
    gpu_num = torch.cuda.device_count()
    # Get the GPU index currently in use
    current_cuda_index = torch.cuda.current_device()
    # Gets the name of the GPU currently in use
    current_cuda_name = torch.cuda.get_device_name(current_cuda_index)
    # Gets the available memory size (in bytes) of the GPU currently in use.
    current_cuda_memory = torch.cuda.get_device_properties(current_cuda_index).total_memory
    logger.info(
        "The number of GPUs on the device is {}, and the current block is device {}:device_name={},device_memory={} GB."
        .format(gpu_num, current_cuda_index, current_cuda_name, round(current_cuda_memory / 1024 ** 3)))


def cuda_is_available():
    """
    Determine if a cuda is currently available
    :return:
    """
    return torch.cuda.is_available()


if __name__ == '__main__':
    print(cuda_is_available())
    current_cuda_info()
    all_cuda_infos()
