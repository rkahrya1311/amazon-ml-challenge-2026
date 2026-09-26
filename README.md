# Amazon ML Challenge 2026

## Problem
Business Entity Resolution across three independent business data sources.

## Team
- Ritees Kumar Ahrya — Leader
- Kishanth G
- Arun Kumar M
- Kishor PM

## Pipeline
Data Loading → Normalization → Blocking → Similarity Features → ML Matching → F₀.₅ Evaluation → Submission

## Repository Structure
- src/ — reusable Python code
- notebooks/ — experiments
- experiments/ — experiment records
- output/ — final generated outputs

## Note
The Amazon challenge dataset is not stored in this repository.

## Blocking reproduction

The reusable implementation is in `src/blocking.py`. It implements the existing B1, B2, and B3 rules from `kishanth_blocking.ipynb` and exports one row per unique S1-to-S2/S3 candidate pair. If multiple rules produce a pair, `blocking_rule` contains labels such as `B1|B3`.

The rules and cutoff are:

- B1: country plus shared normalized business-name token; tokens shorter than 3 characters and configured name stop words are excluded.
- B2: first 4 characters of the normalized business name with spaces removed; country is not part of the key.
- B3: country plus shared normalized business-address token; tokens shorter than 3 characters and configured address stop words are excluded.
- A blocking key is retained only when it matches at most 100 target records across S2 and S3. There are no transliteration or numeric rescue rules.

### Install and run

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python src/blocking.py --data-dir ./train --output ./experiments/candidate_pairs.csv
```

For training data, the folder must contain the three `train_source*.tsv` files and `train_ground_truth.tsv`. For Google Colab, install the requirements and run:

```bash
!pip -q install -r requirements.txt
!python src/blocking.py --data-dir /content/cleaned_training_data --output /content/drive/MyDrive/AmazonML/experiments/candidate_pairs.csv
```

For test data, provide `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` in the data folder. Test data has no ground truth, so candidate recall is reported as unavailable. Run:

```bash
!python src/blocking.py --split test --data-dir /content/test_data --output /content/drive/MyDrive/AmazonML/experiments/test_candidate_pairs.csv
```

Add `--overwrite` to replace an existing output. Use `--maximum-records-per-key N` to explicitly change the default cutoff of 100. The command prints source and country counts, per-rule counts, combined counts, deduplication, candidate distribution, recall, file size, sample rows, and validation results. The candidate CSV is ignored by Git and should be stored in Google Drive, not committed.
