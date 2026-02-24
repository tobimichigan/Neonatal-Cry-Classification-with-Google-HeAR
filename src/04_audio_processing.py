import numpy as np
import librosa
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d
import audiomentations as A
from 01_config import TARGET_SR, CLIP_LENGTH, CLIP_OVERLAP

augmenter = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    A.Shift(min_shift=-0.5, max_shift=0.5, p=0.3),
])

def preprocess_audio(file_path, augment=False):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio = wiener(audio).astype(np.float32)
    audio = gaussian_filter1d(audio, sigma=0.5)
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    audio = audio / rms
    if augment:
        audio = augmenter(samples=audio, sample_rate=TARGET_SR)
    return audio

def segment_audio(audio):
    step = int(CLIP_LENGTH * (1 - CLIP_OVERLAP))
    clips = []
    for start in range(0, max(1, len(audio) - CLIP_LENGTH + 1), step):
        clip = audio[start:start + CLIP_LENGTH]
        if len(clip) < CLIP_LENGTH:
            clip = np.pad(clip, (0, CLIP_LENGTH - len(clip)))
        clips.append(clip)
    return clips
