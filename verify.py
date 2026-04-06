from datasets import load_dataset

ds = load_dataset("HBaoAL/LRS2", streaming=True, split="train")

import libreface
for i, sample in enumerate(ds):
    print(i, sample.keys())

