import functools

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
    return wrapper
