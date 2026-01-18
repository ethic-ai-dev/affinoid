from safetensors.torch import load_file, save_file
from tqdm import tqdm

import hashlib
import random
import torch
import os

MODEL_DIR = "deepseek-model"
NOISE_SCALE = 1e-5

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()

def is_mlp_tensor(name):
    keywords = ["mlp", "gate_proj", "up_proj", "down_proj"]
    return any(k in name for k in keywords)

safetensor_files = sorted(
    f for f in os.listdir(MODEL_DIR)
    if f.endswith(".safetensors")
)[:5]

hashes = []

for fname in tqdm(safetensor_files, desc = "Processing shards"):
    shard_path = os.path.join(MODEL_DIR, fname)
    old_hash = sha256(shard_path)
    
    tensors = load_file(shard_path)

    mlp_keys = [k for k in tensors.keys() if is_mlp_tensor(k)]
    if not mlp_keys: continue

    chosen_key = random.choice(mlp_keys)
    weight = tensors[chosen_key]

    weight_fp32 = weight.float()
    std = torch.std(weight_fp32).item()

    noise_fp32 = torch.randn_like(weight_fp32) * (std * NOISE_SCALE)
    tensors[chosen_key] = (weight_fp32 + noise_fp32).to(weight.dtype)
    
    save_file(tensors, shard_path)
    new_hash = sha256(shard_path)

    hashes.append((old_hash, new_hash))

print("* Done.")
print()

for oh, nh in hashes:
    print(f"{oh} --> {nh}")
