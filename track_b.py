"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np


def evaluate(labeled_data_path, embedding_lookup):
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Look up cosine similarities
    df["sim_a"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]), axis=1
    )

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


# Select baseline method
baseline = "sbert"  # or "random"
data = pd.read_json("data/dev_track_b.jsonl", lines=True)

if baseline == "sbert":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data["text"], show_progress_bar=True)
elif baseline == "random":
    embeddings = torch.rand((len(data), 512))
else:
    sys.exit("Invalid baseline")

embedding_lookup = dict(zip(data["text"], embeddings))
accuracy = evaluate("data/dev_track_a.jsonl", embedding_lookup)
print(f"Accuracy: {accuracy:.3f}")

np.save("output/track_b.npy", embeddings)
