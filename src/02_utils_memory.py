import os
import gc
import psutil
import torch

def get_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

def log_memory(tag=""):
    print(f"[MEM {tag}] {get_memory_gb():.2f} GB")

def force_cleanup(*args):
    for obj in args:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
