import os
import tensorflow as tf
from huggingface_hub import login, snapshot_download
from 01_config import HF_TOKEN_PATH, HEAR_REPO_ID, LOCAL_MODEL_DIR

def load_hear_model():
    with open(HF_TOKEN_PATH, "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)

    if not os.path.exists(LOCAL_MODEL_DIR):
        snapshot_download(repo_id=HEAR_REPO_ID,
                          local_dir=LOCAL_MODEL_DIR,
                          local_dir_use_symlinks=False)

    model = tf.saved_model.load(LOCAL_MODEL_DIR)
    infer = model.signatures["serving_default"]
    return infer
