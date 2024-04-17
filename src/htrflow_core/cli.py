import logging
import multiprocessing
import queue
import subprocess

import torch


logging.basicConfig(level=logging.INFO)


def verify_gpus(gpu_devices):
    """Check if the specified GPU devices are available"""
    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    verified_devices = [gpu for gpu in gpu_devices if gpu in available_gpus]
    if len(verified_devices) != len(gpu_devices):
        missing = set(gpu_devices) - set(verified_devices)
        logging.warning(f"Missing GPUs: {missing}")
    return verified_devices


def command_runner(command_queue, gpu_device, env_gpu_name):
    """Continuously execute commands using a specific GPU device set via environment variable"""
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
        except queue.Empty:
            break

        env = {env_gpu_name: gpu_device}
        try:
            subprocess.run(command, shell=True, env=env, text=True, check=True)
            logging.info(f"Command '{command}' completed successfully on {gpu_device}.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command '{command}' failed with error on {gpu_device}: {e}")
            continue


def queue_runner(commands, gpu_devices, env_gpu_name="CUDA_VISIBLE_DEVICES"):
    verified_gpus = verify_gpus(gpu_devices)
    if not verified_gpus:
        logging.error("No valid GPUs found, exiting.")
        return

    command_queue = multiprocessing.Queue()
    for command in commands:
        command_queue.put(command)

    processes = []
    for gpu in verified_gpus:
        p = multiprocessing.Process(target=command_runner, args=(command_queue, gpu, env_gpu_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logging.info("All jobs have been processed.")


if __name__ == "__main__":
    gpu_devices = ["cuda:0", "cuda:1"]
    commands = ["python script1.py", "python script2.py", "python script3.py"]

    queue_runner(commands, gpu_devices)


# def command_runner(command: str, gpu_device: str, env_gpu_name: str) -> tuple:
#     """ Execute a command using a specific GPU device set via environment variable """
#     env = {env_gpu_name: gpu_device}
#     try:
#         subprocess.run(command, shell=True, env=env, text=True, check=True)
#         return (0, command)  #
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Command '{command}' failed with error: {e}")
#         return (e.returncode, command)

# def queue_runner(commands: list, gpu_devices: list, env_gpu_name: str = 'CUDA_VISIBLE_DEVICES') -> list:
#     """ Execute commands each on a specified GPU if available GPUs match or exceed commands """
#     verified_gpus = verify_gpus(gpu_devices)
#     if not verified_gpus:
#         logging.error("No valid GPUs found, exiting.")
#         return commands

#     if len(commands) > len(verified_gpus):
#         logging.error(f"Not enough GPUs available for the commands. Available GPUs: {len(verified_gpus)}, Commands: {len(commands)}")
#         return commands
#     failed_commands = []
#     with multiprocessing.Pool(processes=len(verified_gpus)) as pool:
#         results = pool.starmap(command_runner, [
#             (cmd, gpu, env_gpu_name) for cmd, gpu in zip(commands, verified_gpus)
#         ])
#         failed_commands = [cmd for result, cmd in results if result != 0]

#     if failed_commands:
#         logging.warning(f"Failed commands: {failed_commands}")
#     else:
#         logging.info("All commands completed successfully.")

#     return failed_commands

# # Usage example:
# gpu_devices = ['cuda:0', 'cuda:1']
# commands = [
#     'python script1.py',
#     'python script2.py'
# ]

# remaining_cmds = queue_runner(commands, gpu_devices)
