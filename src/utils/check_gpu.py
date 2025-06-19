import sys
import subprocess


from common_utils import get_logger

logger = get_logger(__name__)


def check_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info("nvidia-smi output:")
        logger.info(output.decode("utf-8"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "nvidia-smi failed or not found - but we'll continue if PyTorch detects GPU"
        )
        return False


def check_cuda_libraries():
    try:
        output = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        logger.info("Checking for CUDA libraries:")

        cuda_libs = ["libcuda.so", "libcudart.so", "libnvidia-ml.so"]
        for lib in cuda_libs:
            if lib in output:
                logger.info(f"✅ {lib} found")
            else:
                logger.warning(f"❌ {lib} NOT found")

        # We'll just report, not fail if libs are missing
        return True
    except subprocess.CalledProcessError:
        logger.warning(
            "Failed to check CUDA libraries - but we'll continue if PyTorch detects GPU"
        )
        return True


def check_torch_cuda():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {cuda_available}")

        if cuda_available:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(
                    f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                )
        return cuda_available
    except Exception as e:
        logger.error(f"Error checking PyTorch CUDA: {e}")
        return False


def do_all_gpu_checks():
    nvidia_smi_ok = check_nvidia_smi()
    cuda_libs_ok = check_cuda_libraries()
    torch_cuda_ok = check_torch_cuda()
    logger.info("-----------[start gpu checks]------------------")
    logger.info(f"nvidia_smi_ok: {nvidia_smi_ok}")
    logger.info(f"cuda_libs_ok: {cuda_libs_ok}")
    logger.info(f"torch_cuda_ok: {torch_cuda_ok}")
    logger.info("-------------[end gpu checks]------------------")
    return nvidia_smi_ok and cuda_libs_ok and torch_cuda_ok


if __name__ == "__main__":
    logger.info("Running GPU checks...")

    nvidia_smi_ok = check_nvidia_smi()
    cuda_libs_ok = check_cuda_libraries()
    torch_cuda_ok = check_torch_cuda()

    # Only fail if PyTorch can't see CUDA
    if torch_cuda_ok:
        logger.info("✅ PyTorch can access GPU - you are good to go!")
        sys.exit(0)
    else:
        logger.error("❌ PyTorch cannot access GPU - you are doomed!")
        sys.exit(1)
