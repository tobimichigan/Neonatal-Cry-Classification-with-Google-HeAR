import os
import torch

HF_TOKEN_PATH = "hf.token.txt"
HEAR_REPO_ID  = "google/hear"
LOCAL_MODEL_DIR = "./hear-model"

TARGET_SR     = 16000
CLIP_DURATION = 2
CLIP_LENGTH   = TARGET_SR * CLIP_DURATION
CLIP_OVERLAP  = 0.10

BATCH_SIZE    = 32
PT_BATCH      = 16
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-3
NUM_CLASSES   = 3

PLOTS_DIR     = "./plots"
CKPT_PATH     = "./best_model.pth"
ENSEMBLE_DIR  = "./ensemble_models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES   = ["Pain", "Hunger", "Neurological"]

AUGMENT_PROB = 0.5
AUGMENT_FACTOR = 3
