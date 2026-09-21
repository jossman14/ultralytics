import sys
import torch
import ultralytics
print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("Ultralytics version:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
