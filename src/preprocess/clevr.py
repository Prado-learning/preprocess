"""
process_superclevr.py

Convert superCLEVR questions JSON to ShareGPT format and split into train/test sets.

Requirements:
  - sharegpt.py in the same directory
  - Images extracted to /root/sCLEVR/images/
"""

import io
import json
import os
import random

from PIL import Image as PILImage
from datasets import Dataset, DatasetDict

from sharegpt import mm_features

JSON_PATH  = "/root/sCLEVR/superCLEVR_questions_30k.json"
IMAGE_ROOT = "/root/sCLEVR/images/images"
SAVE_PATH  = "./superclevr_sharegpt"
SEED       = 42
TEST_RATIO = 0.3


def pil_to_bytes(img: PILImage.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def convert_answer(answer) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    return str(answer).strip()


def convert_one(item: dict) -> dict | None:
    image_path = f"{IMAGE_ROOT}/{item['image_filename']}"
    if not os.path.exists(image_path):
        return None

    image_bytes = pil_to_bytes(PILImage.open(image_path))
    answer = convert_answer(item["answer"])
    problem = f"<image>\n{item['question']}"

    return {
        "problem": problem,
        "answer": answer,
        "conversations": [
            {"from": "human", "value": problem},
            {"from": "gpt",   "value": answer},
        ],
        "images": [{"bytes": image_bytes}],
        "_qid": str(item["question_index"]),
    }


if __name__ == "__main__":
    print("Loading superCLEVR_questions_30k.json ...")
    with open(JSON_PATH) as f:
        questions = json.load(f)["questions"]
    print(f"Total: {len(questions):,} questions\n")

    print("Converting to ShareGPT format ...")
    converted, skipped = [], 0
    for i, item in enumerate(questions):
        result = convert_one(item)
        if result is None:
            skipped += 1
        else:
            converted.append(result)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} / {len(questions):,}")
    print(f"Done: {len(converted):,} converted, {skipped} skipped (missing images)\n")

    print("Splitting train / test (70% / 30%) ...")
    random.seed(SEED)
    random.shuffle(converted)
    split_idx  = int(len(converted) * (1 - TEST_RATIO))
    train_data = converted[:split_idx]
    test_data  = converted[split_idx:]
    print(f"  train: {len(train_data):,}")
    print(f"  test:  {len(test_data):,}\n")

    print(f"Saving to {SAVE_PATH} ...")
    ds = DatasetDict({
        "train": Dataset.from_list(train_data, features=mm_features),
        "test":  Dataset.from_list(test_data,  features=mm_features),
    })
    ds.save_to_disk(SAVE_PATH)
    print("Saved.\n")

    print("=== Verifying first train sample ===")
    first = ds["train"][0]
    print(f"_qid   : {first['_qid']}")
    print(f"problem: {first['problem']}")
    print(f"answer : {first['answer']}")
    img: PILImage.Image = first["images"][0]
    print(f"image  : {img.size}, mode={img.mode}")
    print("\n[conversations]")
    for turn in first["conversations"]:
        print(f"  [{turn['from']:5s}] {turn['value'][:150].replace(chr(10), chr(92)+'n')}")