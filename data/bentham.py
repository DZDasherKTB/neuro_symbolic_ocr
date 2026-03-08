# clear_vram.py

import torch
import gc

def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("VRAM cleared.")

if __name__ == "__main__":
    clear_vram()