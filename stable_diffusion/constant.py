import platform

PRETRAINED_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-2-1"
TORCH_DEVICE = "mps" if (platform.system() == "Darwin") else "cuda"
HUGGINGFACE_TOKEN = ''
MAXSTAPEDENOISE = 1
