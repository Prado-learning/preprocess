"""
process_pope.py

Load lmms-lab/POPE (test split), shuffle, convert to ShareGPT format,
save locally, and push to HuggingFace.

Requirements:
  - sharegpt.py in the same directory
  - pip install datasets pillow huggingface_hub
"""

import io
import os

from PIL import Image as PILImage
from datasets import load_dataset
from huggingface_hub import login

from sharegpt import mm_features

SAVE_PATH = "./pope_sharegpt"
SEED      = 42


def pil_to_bytes(img: PILImage.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def map_pope(e: dict, idx: int) -> dict:
    problem = f"<image>\n{e['question']}"
    answer  = e["answer"].strip().lower()
    return {
        "problem": problem,
        "answer": answer,
        "conversations": [
            {"from": "human", "value": problem},
            {"from": "gpt",   "value": answer},
        ],
        "images": [{"bytes": pil_to_bytes(e["image"])}],
        "_qid": str(e["question_id"]),
    }


if __name__ == "__main__":
    print("Loading lmms-lab/POPE (test split) ...")
    ds = load_dataset("lmms-lab/POPE", split="test")
    print(f"  Total: {len(ds):,}")

    print(f"\nShuffling (seed={SEED}) ...")
    ds = ds.shuffle(seed=SEED)

    print("\nConverting to ShareGPT format ...")
    ds = ds.map(
        map_pope,
        with_indices=True,
        remove_columns=ds.column_names,
        features=mm_features,
        num_proc=4,
    )
    print(f"  Done: {len(ds):,}")

    print(f"\nSaving to {SAVE_PATH} ...")
    ds.save_to_disk(SAVE_PATH)
    print("Saved.\n")

    print("=== Verifying first sample ===")
    first = ds[0]
    print(f"_qid   : {first['_qid']}")
    print(f"problem: {first['problem']}")
    print(f"answer : {first['answer']}")
    img: PILImage.Image = first["images"][0]
    print(f"image  : {img.size}, mode={img.mode}")
    print("\n[conversations]")
    for turn in first["conversations"]:
        print(f"  [{turn['from']:5s}] {turn['value'][:150].replace(chr(10), chr(92)+'n')}")

    print("\nPushing to HuggingFace ...")
    login(token=os.environ["HF_TOKEN"])
    ds.push_to_hub("Prado2026/pope")
    print("Done. https://huggingface.co/datasets/Prado2026/pope")