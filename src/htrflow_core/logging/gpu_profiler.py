import functools
import time

import torch

from htrflow_core.logging.logger import CustomLogger


def profile_gpu_usage(verbose=0):
    logger = CustomLogger("GPUProfiler", verbose=verbose).logger

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_time = time.time()
                start_mem = torch.cuda.memory_allocated()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                end_time = time.time()
                end_mem = torch.cuda.memory_allocated()
                logger.info(f"Memory Usage for {func.__name__}: {end_mem - start_mem} bytes")
                logger.info(f"Execution Time for {func.__name__}: {end_time - start_time} seconds")
            else:
                logger.warning("CUDA is not available. Running the function without profiling GPU usage.")
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":

    @profile_gpu_usage(verbose=2)
    def my_gpu_intensive_function(tensor_size, n_operations):
        a = torch.rand(tensor_size, tensor_size, device="cuda")
        b = torch.rand(tensor_size, tensor_size, device="cuda")
        for _ in range(n_operations):
            a = a * b + torch.sin(a)
        return a

    if torch.cuda.is_available():
        result = my_gpu_intensive_function(10000, 1000)
