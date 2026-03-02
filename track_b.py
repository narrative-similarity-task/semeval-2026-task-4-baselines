"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
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
data = pd.read_json("narrative-similarity-dataset/test/test_track_b.jsonl", lines=True)

if baseline == "sbert":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data["text"], show_progress_bar=True)
elif baseline == "e5" or baseline == "story-emb":
    device = "cuda:0"
    if baseline == "e5":
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    else:
        word_embedding_model = models.Transformer("uhhlt/story-emb")
        # Specify the pooling mode
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,      # Use [CLS] token
            pooling_mode_mean_tokens=False,   # Don't use mean pooling
            pooling_mode_max_tokens=False     # Don't use max pooling
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Define batch size
    batch_size = 32

    embeddings = model.encode(
        ["Retrieve stories with a similar narrative to the given story: " + x for x in data["text"]],
        convert_to_tensor=True,
        device=device,
        batch_size=32,
        show_progress_bar=True
    )

elif baseline == "random":
    embeddings = torch.rand((len(data), 512))
else:
    sys.exit("Invalid baseline")

embedding_lookup = dict(zip(data["text"], embeddings))
accuracy = evaluate("narrative-similarity-dataset/test/labels/test_track_b_labels.jsonl", embedding_lookup)
print(f"Accuracy: {accuracy:.3f}")

np.save("output/track_b.npy", embeddings)
