"""
process_gqa_answer.py

Identical to process_gqa.py, except GT answer uses the `answer` field
(short answer, e.g. "yes" / "boy") instead of `fullAnswer`
(full sentence, e.g. "Yes, the shorts are dark.").

Local save : ./gqa_answer_sharegpt
HuggingFace: Prado2026/gqa_answer

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
SAVE_PATH  = "./gqa_answer_sharegpt"
SEED       = 42

SPLITS = [
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
        answer  = str(row["answer"]).strip()
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
    ds = DatasetDict(all_splits)
    ds.save_to_disk(SAVE_PATH)
    print("Saved.\n")

    print("Pushing to HuggingFace ...")
    login(token=os.environ["HF_TOKEN"])
    ds.push_to_hub("Prado2026/gqa_answer")
    print("Done. https://huggingface.co/datasets/Prado2026/gqa_answer")