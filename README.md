# Baselines SemEval 2026 Task 4: Shared Task on Narrative Story Similarity

This repository holds baseline systems for the shared task **SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**.
For details on the task please see [our website](https://narrative-similarity-task.github.io/).


## Setup

Install the dependencies (you probably want to do this in an isolated environment of some sort: virtualenv, conda, etc.):
```
pip install -r requirements.txt
```

Second, to download the data you will need to initialize the submodle with the dataset: `git submodule update --init --recursive`

Run `python track_a.py` or `python track_b.py`
