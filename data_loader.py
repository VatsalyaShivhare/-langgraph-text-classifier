import os
from tqdm import tqdm

def load_imdb_dataset(data_dir="aclImdb", split="train", limit=1000):
    texts = []
    labels = []

    for label in ["pos", "neg"]:
        path = os.path.join(data_dir, split, label)
        files = os.listdir(path)[:limit // 2]

        for file in tqdm(files, desc=f"Loading {label}"):
            with open(os.path.join(path, file), encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(1 if label == "pos" else 0)

    return texts, labels
