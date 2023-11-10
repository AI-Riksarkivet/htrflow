import functools
import time

import pynvml


def gpu_memory_usage_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        mem_before = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in range(num_gpus)
        ]
        result = func(*args, **kwargs)
        mem_after = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in range(num_gpus)
        ]
        pynvml.nvmlShutdown()
        for i, (before, after) in enumerate(zip(mem_before, mem_after)):
            print(f"GPU {i} Memory Used: {(after - before) / (1024 ** 2):.2f} MB")
        return result

    return wrapper


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.2f} seconds")
        return result

    return wrapper
