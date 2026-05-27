"""
Estrazione feature DINOv3 da video raw EGTEA Gaze+.

Uso:
    python -m src.utils.extract_dinov3_features \
        --videos_dir D:/egtea/videos \
        --output_dir D:/egtea/dinov3_features \
        --model facebook/dinov3-vitb16-pretrain-lvd1689m \
        --batch_size 32

Output: un file .npy per video, shape (T, feat_dim), dtype float32.
Il nome del file corrisponde al video_session usato nelle annotazioni EGTEA.
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, DINOv3ViTModel


@torch.inference_mode()
def extract_features_streaming(video_path: Path, model, processor,
                               device: str, batch_size: int) -> np.ndarray:
    """Legge e processa i frame in batch senza caricare l'intero video in RAM."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    all_feats = []
    batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if len(batch) == batch_size:
            inputs = processor(images=batch, return_tensors="pt").to(device)
            feats = model(**inputs).pooler_output.float().cpu().numpy()
            all_feats.append(feats)
            batch = []

    # Ultimo batch parziale
    if batch:
        inputs = processor(images=batch, return_tensors="pt").to(device)
        feats = model(**inputs).pooler_output.float().cpu().numpy()
        all_feats.append(feats)

    cap.release()
    return np.concatenate(all_feats, axis=0) if all_feats else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir",  default="D:/egtea/videos")
    parser.add_argument("--output_dir",  default="D:/egtea/dinov3_features")
    parser.add_argument("--model",       default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Caricamento modello {args.model} ...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = DINOv3ViTModel.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    feat_dim = model.config.hidden_size
    print(f"feat_dim: {feat_dim}")

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"Video trovati: {len(video_files)}")

    for video_path in tqdm(video_files, desc="Estrazione"):
        video_session = video_path.stem
        out_path = output_dir / f"{video_session}.npy"

        if out_path.exists():
            tqdm.write(f"  SKIP {video_session}")
            continue

        try:
            features = extract_features_streaming(
                video_path, model, processor, device, args.batch_size
            )
            if features is None:
                tqdm.write(f"  WARN {video_session}: nessun frame")
                continue

            np.save(str(out_path), features)
            tqdm.write(f"  OK   {video_session}: {features.shape}")

        except Exception as e:
            tqdm.write(f"  ERR  {video_session}: {e}")


if __name__ == "__main__":
    main()
