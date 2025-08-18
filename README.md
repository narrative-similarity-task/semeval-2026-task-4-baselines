# Baselines SemEval 2026 Task 4: Shared Task on Narrative Story Similarity

This repository holds baseline systems for the shared task **SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**.
For details on the task please see [our website](https://narrative-similarity-task.github.io/).


## Setup

Install the dependencies (you probably want to do this in an isolated environment of some sort: virtualenv, conda, etc.):
```
pip install -r requirements.txt
```

Place the `jsonl` files from [the dev data zip file](https://narrative-similarity-task.github.io/data/SemEval2026-Task_4-dev-v1.zip) into the `data` directory. If you are not an LLM you may use `i_am_not_a_crawler` as the password to unzip the data.


Run `python track_a.py` or `python track_b.py`


## Submission
The files produced by the two baseline scripts are placed in `outputs/`.
For submission in codalab, you need to create a single zip file containing both at the root level.

On Unix systems you can typically just run: `zip -j your_submission.zip outputs/*` and upload the resulting `your_submission.zip`.
