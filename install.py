import subprocess
import sys

is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules


def install_requirements(is_qa: bool = False):
    """Installs the required packages for the project."""
    print("Installing base requirements ...")
    if is_qa:
        requirements = "requirements_qa.txt -f https://download.pytorch.org/whl/torch_stable.html".split()
    else:
        requirements = "requirements.txt"
    process_install = subprocess.run(
        ["python", "-m", "pip", "install", "-r", requirements], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process_install.returncode != 0:
        raise Exception("Failed to install requirements")
    else:
        print("Base requirements installed!")
    if is_colab or is_kaggle:
        import torch

        torch_version = torch.__version__.split("+")[0]
        print("Installing torch-scatter ...")
        process_scatter = subprocess.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "torch-scatter",
                "-f",
                f"https://data.pyg.org/whl/torch-{torch_version}+cu113.html",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode == -1:
            raise Exception("Failed to install torch-scatter")
        print("Installing Git LFS and soundfile ...")
        process_lfs = subprocess.run(
            ["apt", "install", "git-lfs", "libsndfile1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process_lfs.returncode == -1:
            raise Exception("Failed to install Git LFS")
        print("Install complete!")
