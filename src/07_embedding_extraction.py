import numpy as np
import tensorflow as tf
from 04_audio_processing import preprocess_audio, segment_audio

def extract_embeddings_batch(hear_infer, clips_batch):
    tf_input = tf.constant(clips_batch.astype(np.float32))
    out = hear_infer(x=tf_input)
    return out["output_0"].numpy()

def extract_file_embedding(hear_infer, file_path):
    audio = preprocess_audio(file_path)
    clips = segment_audio(audio)
    batch = np.stack(clips)
    emb = extract_embeddings_batch(hear_infer, batch)
    return emb.mean(axis=0)
