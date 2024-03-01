import subprocess


def main():
    try:
        subprocess.check_call(["pip", "install", "-U", "openmim"])
        subprocess.check_call(["min", "install", "-U", "mmcv"])
        subprocess.check_call(["min", "install", "-U", "mmengine"])
        subprocess.check_call(["min", "install", "-U", "mmocr"])
        subprocess.check_call(["min", "install", "-U", "mmdet"])

    except subprocess.CalledProcessError as e:
        print(f"Failed to execute mim installation commands: {e}")
        raise
