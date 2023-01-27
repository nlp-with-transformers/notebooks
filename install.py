import subprocess
import sys

is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules
# torch-scatter binaries depend on the torch and CUDA version, so we define the
# mappings here for Colab & Kaggle
torch_to_cuda = {"1.10.0": "cu113", "1.9.0": "cu111", "1.9.1": "cu111"}


def install_requirements(
    is_chapter2: bool = False, 
    is_chapter6: bool = False,
    is_chapter7: bool = False,
    is_chapter7_v2: bool = False,
    is_chapter10: bool = False,
    is_chapter11: bool = False
    ):
    """Installs the required packages for the project."""

    print("⏳ Installing base requirements ...")
    cmd = ["python", "-m", "pip", "install", "-r"]
    if is_chapter7:
        cmd += "requirements-chapter7.txt -f https://download.pytorch.org/whl/torch_stable.html".split()
    elif is_chapter7_v2:
        cmd.append("requirements-chapter7-v2.txt")
    else:
        cmd.append("requirements.txt")
    process_install = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_install.returncode != 0:
        raise Exception("😭 Failed to install base requirements")
    else:
        print("✅ Base requirements installed!")
    print("⏳ Installing Git LFS ...")
    process_lfs = subprocess.run(["apt", "install", "git-lfs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_lfs.returncode == -1:
        raise Exception("😭 Failed to install Git LFS and soundfile")
    else:
        print("✅ Git LFS installed!")

    if is_chapter2:
        transformers_cmd = "python -m pip install transformers==4.13.0 datasets==2.8.0".split()
        process_scatter = subprocess.run(
            transformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    if is_chapter6:
        transformers_cmd = "python -m pip install datasets==2.0.0".split()
        process_scatter = subprocess.run(
            transformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    if is_chapter10:
        wandb_cmd = "python -m pip install wandb".split()
        process_scatter = subprocess.run(
            wandb_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if is_chapter11:
        import torch

        torch_version = torch.__version__.split("+")[0]
        print(f"⏳ Installing torch-scatter for torch v{torch_version} ...")
        if is_colab:
            torch_scatter_cmd = f"python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch_version}+{torch_to_cuda[torch_version]}.html".split()
        else:
            # Kaggle uses CUDA 11.0 by default, so we need to build from source
            torch_scatter_cmd = "python -m pip install torch-scatter".split()
        process_scatter = subprocess.run(
            torch_scatter_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode == -1:
            raise Exception("😭 Failed to install torch-scatter")
        else:
            print("torch-scatter installed!")
        print("⏳ Installing soundfile ...")
        process_audio = subprocess.run(
            ["apt", "install", "libsndfile1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process_audio.returncode == -1:
            raise Exception("😭 Failed to install soundfile")
        else:
            print("✅ soundfile installed!")
        print("🥳 Chapter installation complete!")
