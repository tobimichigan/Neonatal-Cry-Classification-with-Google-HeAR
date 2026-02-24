from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def collect_audio_files(root_paths):
    audio_files = []
    for root in root_paths:
        for path in tqdm(list(Path(root).rglob("*"))):
            if path.suffix.lower() in AUDIO_EXTS:
                audio_files.append(str(path))
    return audio_files

def split_data(files, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        files, labels, test_size=0.60, stratify=labels, random_state=42)
    X_val, X_hold_test, y_val, y_hold_test = train_test_split(
        X_temp, y_temp, test_size=0.75, stratify=y_temp, random_state=42)
    X_test, X_hold, y_test, y_hold = train_test_split(
        X_hold_test, y_hold_test, test_size=0.67, stratify=y_hold_test, random_state=42)
    return X_train, X_val, X_test, X_hold, y_train, y_val, y_test, y_hold
