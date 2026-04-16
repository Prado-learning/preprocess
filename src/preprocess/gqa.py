"""
process_gqa.py

Convert lmms-lab/GQA (balanced splits) to ShareGPT format.

Splits:
  train : train_balanced_instructions[:30000] + train_balanced_images
  val   : val_balanced_instructions[:30000]   + val_balanced_images
  test  : testdev_balanced_instructions (100%) + testdev_balanced_images

GT answer : fullAnswer column
Images    : raw JPEG bytes (no PIL conversion)

Requirements:
  - sharegpt.py in the same directory
  - pip install datasets huggingface_hub
"""

import os
import random

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Image
from huggingface_hub import login

from sharegpt import mm_features

HF_DATASET = "lmms-lab/GQA"
SAVE_PATH  = "./gqa_sharegpt"
SEED       = 42

SPLITS = [
    # (image_subset, image_split, instruct_subset, instruct_split, limit, out_name)
    ("train_balanced_images",   "train",   "train_balanced_instructions",   "train",   30000, "train"),
    ("val_balanced_images",     "val",     "val_balanced_instructions",     "val",     30000, "val"),
    ("testdev_balanced_images", "testdev", "testdev_balanced_instructions", "testdev", None,  "test"),
]


def build_image_lookup(subset: str, hf_split: str) -> dict[str, bytes]:
    """Load image subset, return {image_id: raw_jpeg_bytes}."""
    print(f"  Loading images [{subset} / {hf_split}] ...")
    ds = load_dataset(HF_DATASET, subset, split=hf_split)
    ds = ds.cast_column("image", Image(decode=False))
    lookup = {row["id"]: row["image"]["bytes"] for row in ds}
    print(f"    -> {len(lookup):,} images loaded")
    return lookup


def convert_instructions(
    instruct_subset: str,
    hf_split: str,
    image_lookup: dict[str, bytes],
    limit: int | None,
) -> list[dict]:
    """Load instruction subset, join with image lookup, return ShareGPT records."""
    print(f"  Loading instructions [{instruct_subset} / {hf_split}] ...")
    ds = load_dataset(HF_DATASET, instruct_subset, split=hf_split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"    -> limited to {limit:,}")

    converted, skipped = [], 0
    for i, row in enumerate(ds):
        if row["imageId"] not in image_lookup:
            skipped += 1
            continue
        problem = f"<image>\n{row['question']}"
        answer  = str(row["fullAnswer"]).strip()
        converted.append({
            "problem": problem,
            "answer":  answer,
            "conversations": [
                {"from": "human", "value": problem},
                {"from": "gpt",   "value": answer},
            ],
            "images": [{"bytes": image_lookup[row["imageId"]]}],
            "_qid":   row["id"],
        })
        if (i + 1) % 10_000 == 0:
            print(f"    {i+1:,} processed ...")

    print(f"    -> {len(converted):,} converted, {skipped} skipped (missing image)\n")
    return converted


if __name__ == "__main__":
    random.seed(SEED)
    all_splits: dict[str, Dataset] = {}

    for img_subset, img_split, inst_subset, inst_split, limit, out_name in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing split: {out_name}")
        print(f"{'='*60}")

        image_lookup = build_image_lookup(img_subset, img_split)
        data = convert_instructions(inst_subset, inst_split, image_lookup, limit)
        del image_lookup

        if out_name == "train":
            random.shuffle(data)

        all_splits[out_name] = Dataset.from_list(data, features=mm_features)

    print(f"\nSaving to {SAVE_PATH} ...")
    DatasetDict(all_splits).save_to_disk(SAVE_PATH)
    print("Saved.\n")

    print("=== Verifying first sample of each split ===")
    ds = load_from_disk(SAVE_PATH)
    for split_name, split_ds in ds.items():
        first = split_ds[0]
        print(f"\n[{split_name}]")
        print(f"  _qid   : {first['_qid']}")
        print(f"  problem: {first['problem'][:80]}")
        print(f"  answer : {first['answer']}")
        img = first["images"][0]
        if isinstance(img, bytes):
            print(f"  image  : {len(img):,} bytes")
        else:
            print(f"  image  : {img.size}, mode={img.mode}")

    print("\nPushing to HuggingFace ...")
    login(token=os.environ["HF_TOKEN"])
    DatasetDict(all_splits).push_to_hub("Prado2026/gqa")
    print("Done. https://huggingface.co/datasets/Prado2026/gqa")