"""
process_textvqa.py

Load lmms-lab/textvqa, select majority-vote answer from 10 candidates,
convert to ShareGPT format, save locally, and push to HuggingFace.

Requirements:
  - sharegpt.py in the same directory
  - pip install datasets pillow
"""

import io
from collections import Counter

from PIL import Image as PILImage
from datasets import load_dataset

from sharegpt import mm_features

SAVE_PATH = "./textvqa_sharegpt"


def get_majority_answer(answers: list[str]) -> str:
    counter = Counter(a.strip().lower() for a in answers)
    return counter.most_common(1)[0][0]


def pil_to_bytes(img: PILImage.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def map_textvqa(e: dict) -> dict:
    answer  = get_majority_answer(e["answers"])
    problem = f"<image>\n{e['question']}"
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
    print("Loading lmms-lab/textvqa ...")
    ds = load_dataset("lmms-lab/textvqa")
    for split, subset in ds.items():
        print(f"  {split}: {len(subset):,}")

    print("\nConverting to ShareGPT format ...")
    ds = ds.map(
        map_textvqa,
        remove_columns=ds["train"].column_names,
        features=mm_features,
        num_proc=4,
    )

    print(f"\nSaving to {SAVE_PATH} ...")
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
        print(f"  [{turn['from']:5s}] {turn['value'][:120].replace(chr(10), chr(92)+'n')}")

    print("\nPushing to HuggingFace ...")
    ds.push_to_hub("Prado2026/textvqa")
    print("Done.")