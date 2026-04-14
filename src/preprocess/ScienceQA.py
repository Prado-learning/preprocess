"""
process_scienceqa.py

Load derek-thomas/ScienceQA, filter image-only entries,
convert to ShareGPT format, save locally, and push to HuggingFace.

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

SAVE_PATH = "./scienceqa_sharegpt"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")


def pil_to_bytes(img: PILImage.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGHIJ"
    return "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def map_scienceqa(e: dict, idx: int) -> dict:
    answer = "ABCDEFGHIJ"[e["answer"]]
    problem = f"<image>\n{e['question']}\n{format_choices(e['choices'])}"
    return {
        "problem": problem,
        "answer": answer,
        "conversations": [
            {"from": "human", "value": problem},
            {"from": "gpt",   "value": answer},
        ],
        "images": [{"bytes": pil_to_bytes(e["image"])}],
        "_qid": idx,
    }


if __name__ == "__main__":
    print("Loading derek-thomas/ScienceQA ...")
    ds = load_dataset("derek-thomas/ScienceQA")
    for split, subset in ds.items():
        print(f"  {split}: {len(subset):,}")

    print("\nFiltering entries without images ...")
    ds = ds.filter(lambda e: e["image"] is not None, num_proc=4)
    for split, subset in ds.items():
        print(f"  {split}: {len(subset):,} (with image)")

    print("\nConverting to ShareGPT format ...")
    ds = ds.map(
        map_scienceqa,
        remove_columns=ds["train"].column_names,
        features=mm_features,
        with_indices=True,
        num_proc=4,
    )
    for split, subset in ds.items():
        print(f"  {split}: {len(subset):,}")

    print(f"\nSaving to {SAVE_PATH} ...")
    ds.save_to_disk(SAVE_PATH)
    print("Saved.\n")

    print("=== Verifying first train sample ===")
    first = ds["train"][0]
    print(f"_qid   : {first['_qid']}")
    print(f"problem:\n{first['problem']}")
    print(f"answer : {first['answer']}")
    img: PILImage.Image = first["images"][0]
    print(f"image  : {img.size}, mode={img.mode}")
    print("\n[conversations]")
    for turn in first["conversations"]:
        print(f"  [{turn['from']:5s}] {turn['value'][:200].replace(chr(10), chr(92)+'n')}")

    print("\nPushing to HuggingFace ...")
    login(token=HF_TOKEN)
    ds.push_to_hub("Prado2026/scienceqa")
    print("Done.")