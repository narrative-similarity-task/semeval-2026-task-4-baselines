"""
Track A baseline system.

We use a naive prompt for chatGPT, a random baseline, or a Jaccard similarity baseline.
"""

import random
from enum import Enum

from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel

tqdm.pandas()

class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum


def jaccard_similarity(text1, text2):
    """Calculates the Jaccard similarity between two strings."""
    s1 = set(text1.lower().split())
    s2 = set(text2.lower().split())
    if not s1 and not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


def predict_openai(row, client):
    """
    Uses the OpenAI API to determine which of two stories (A or B) is more narratively similar to an anchor story.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert on stories and narratives. Tell us which of two stories is narratively similar to the anchor story.",
            },
            {
                "role": "user",
                "content": f"Anchor story: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}",
            },
        ],
        response_format=SimilarityPrediction,
    )
    return completion.choices[0].message.parsed.closer == ResponseEnum.A


# Configuration: "openai", "random", or "jaccard"
baseline = "jaccard"

df = pd.read_json("narrative-similarity-dataset/test/test_track_a.jsonl", lines=True)
df_labels = pd.read_json("narrative-similarity-dataset/test/labels/test_track_a_labels.jsonl", lines=True)

if baseline == "openai":
    client = OpenAI()
    df["predicted_text_a_is_closer"] = df.progress_apply(lambda row: predict_openai(row, client), axis=1)

elif baseline == "jaccard":
    def predict_jaccard(row):
        sim_a = jaccard_similarity(row["anchor_text"], row["text_a"])
        sim_b = jaccard_similarity(row["anchor_text"], row["text_b"])
        # If similarities are equal, we default to False or could use a random tie-break
        return sim_a > sim_b

    df["predicted_text_a_is_closer"] = df.apply(predict_jaccard, axis=1)

elif baseline == "random":
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )

accuracy = (df["predicted_text_a_is_closer"] == df_labels["text_a_is_closer"]).mean()
print(f"Baseline: {baseline}")
print(f"Accuracy: {accuracy * 100:.2f}")
