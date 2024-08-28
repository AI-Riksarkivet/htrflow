# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "platform",
#     "subprocess",
# ]
# ///
import platform
import subprocess


def install_packages():
    try:
        subprocess.check_call(["uv", "pip", "install", "-U", "torch==2.0.0"])
        os_name = platform.system().lower()
        if os_name == "windows":
            mmcv_url = "https://github.com/AI-Riksarkivet/openmim_install/raw/main/mmcv-2.0.0-cp310-cp310-win_amd64.whl"
        elif os_name == "linux":
            mmcv_url = "https://github.com/AI-Riksarkivet/openmim_install/raw/main/mmcv-2.0.0-cp310-cp310-manylinux1_x86_64.whl"
        elif os_name == "darwin":  # macOS is identified as 'Darwin'
            mmcv_url = "https://github.com/AI-Riksarkivet/openmim_install/raw/main/mmcv-2.0.0-cp310-cp310-manylinux1_x86_64.whl"
        else:
            raise ValueError(f"Unsupported operating system: {os_name}")

        subprocess.check_call(["uv", "pip", "install", "-U", mmcv_url])
        subprocess.check_call(["uv", "pip", "install", "-U", "mmdet==3.1.0"])
        subprocess.check_call(["uv", "pip", "install", "-U", "mmengine==0.7.2"])
        subprocess.check_call(["uv", "pip", "install", "-U", "mmocr==1.0.1"])
        subprocess.check_call(["uv", "pip", "install", "-U", "yapf==0.40.1"])

    except subprocess.CalledProcessError as e:
        print(f"Failed to execute installation commands: {e}")
        raise


def main():
    install_packages()


if __name__ == "__main__":
    main()
