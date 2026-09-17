# Cyberbullying Detection (ML)

A small machine-learning project that classifies a social-media comment as cyberbullying or safe, using TF-IDF and logistic regression with a Gradio web interface.

## Overview

The script loads a labelled CSV of comments, cleans the text, vectorises it with TF-IDF and trains a logistic regression classifier. Accuracy on a held-out split is printed to the console at startup. A Gradio interface then lets you type a comment and see the verdict, and every prediction is written to a local SQLite log.

This is a learning-scale project: the point is the end-to-end pipeline (data → features → model → interface → logging), not benchmark performance.

## Features

- Text cleaning: lowercasing, URL and @mention removal, non-letter stripping
- TF-IDF vectorisation
- Logistic regression classifier (`max_iter=1000`)
- Train/test split with accuracy reported at startup
- Gradio web interface for entering a comment and getting a verdict
- SQLite logging of every prediction (comment + result)

## Screenshots

**Gradio interface**

![Application interface](screenshots/01-application-interface.png)

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Language | Python 3.10+ |
| Machine learning | scikit-learn (TF-IDF, LogisticRegression), pandas |
| Interface | Gradio |
| Storage | SQLite (prediction log) |

## Project Structure

```
cyberbullying-detection-ml/
├── main.py               # Cleaning, training, evaluation, SQLite logging, Gradio UI
├── cyberbullying.csv     # Labelled dataset — text,label (389 rows)
├── requirements.txt
├── screenshots/
├── test.py               # Earlier experiment script, kept for reference
├── test1.py              # Earlier experiment script, kept for reference
└── .gitignore
```

## Dataset

`cyberbullying.csv` is a small hand-built dataset of 389 labelled comments with the columns `text` and `label` (`1` = cyberbullying, `0` = safe). The classes are close to balanced (195 / 194). It is large enough to demonstrate the pipeline but far too small to support a real-world accuracy claim.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

The console prints the model accuracy for the current train/test split, then Gradio starts and prints a local URL (typically `http://127.0.0.1:7860`) to open in a browser.

## Notes / Limitations

- The dataset is small (389 comments), so the reported accuracy is illustrative rather than meaningful. Expect it to vary between runs because the train/test split is random.
- Binary classification only, English text only, and no handling of sarcasm, context or obfuscated spelling.
- Predictions are written to a local `cyberbullying_logs.db` file, which is gitignored.
- No model versioning, no evaluation harness and no saved model artefact — the model is retrained on every run.
- `test.py` and `test1.py` are earlier experiment scripts kept for reference; `main.py` is the current version.
- Development dates are approximate. The project was published to GitHub as a single initial commit rather than being developed in public.

## Future Improvements

- A larger, properly sourced labelled dataset.
- Evaluation with precision/recall and a confusion matrix instead of accuracy alone.
- Saving the trained model so it does not need retraining on each run.
