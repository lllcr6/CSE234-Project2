# Schema Linking with LoRA-Tuned Qwen2.5

This repository contains a schema-linking pipeline for CSE/DSC 234 Project 2.
Given a natural-language question and a target database schema, the system
predicts the tables and columns needed to answer the question.

This project uses a LoRA-tuned `Qwen/Qwen2.5-1.5B-Instruct` model with
schema-aware JSON post-processing. On the validation set, the selected
configuration achieves an overall schema-linking score of **0.4993**, with
table score **0.5566** and column score **0.4421**.

The overall score is `0.5 * table_score + 0.5 * column_score`. Each component
score is the mean of precision, recall, and F1 at that granularity.

## Task

Input examples contain a question, a `db_id`, and a schema stored in Spider JSON
format under `schemas/`. The model outputs a JSON object mapping referenced
tables to referenced columns:

```json
{
  "question_id": 1,
  "schema_links": {
    "INJURY": ["AIS", "REGION"]
  }
}
```

Tables referenced without specific columns are included with an empty list. For
example, `select count(*) from AIRBAG` corresponds to:

```json
{"AIRBAG": []}
```

## Repository Layout

```text
.
├── main.py                         # Inference entry point
├── eval.py                         # Validation grader
├── train.json                      # Training split with gold SQL and links
├── validation.json                 # Validation split with gold SQL and links
├── validation_input.json           # Input-only validation questions
├── validation_gold_schema_links.json
├── schemas/                        # Spider-format database schemas
├── adapter/                        # Selected LoRA adapter
├── all_adapters/                   # Saved adapters from ablations
├── predictions/                    # Validation predictions from experiments
├── eval_results.md                 # Full validation result tables
├── scheme_linking_SFT.ipynb        # RapidFire AI SFT experiments
└── scheme_linking_SFT_Qwen_lr.ipynb # Qwen learning-rate sweep
```

## Setup

Create a Python environment and install the inference dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch transformers peft accelerate tqdm
```

The training notebooks also require RapidFire AI and the usual SFT stack
(`trl`, `datasets`, etc.).

## Run Inference

Run the selected model on the validation input:

```bash
python main.py \
  --input validation_input.json \
  --output preds.json \
  --schemas_dir schemas \
  --adapter adapter
```

The default adapter path is `./adapter`, so this shorter command is equivalent:

```bash
python main.py \
  --input validation_input.json \
  --output preds.json \
  --schemas_dir schemas
```

To evaluate a different saved adapter:

```bash
python main.py \
  --input validation_input.json \
  --output preds_qwen_lr2e4.json \
  --schemas_dir schemas \
  --adapter all_adapters/qwen2.5-1.5b-lora-rank8-qv-3epoch-lr2e-4
```

## Evaluate Predictions

```bash
python eval.py \
  --predictions preds.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas \
  --questions_input validation_input.json \
  --per_question_out per_question_metrics.csv
```

The evaluator reports table precision/recall/F1, column precision/recall/F1,
table score, column score, and the combined overall schema-linking score.

## Pipeline

The pipeline has four stages:

1. Load the target schema from `schemas/<db_id>.json`.
2. Serialize the schema compactly as one table per line, e.g.
   `TABLE_NAME(col1, col2, col3)`.
3. Prompt `Qwen/Qwen2.5-1.5B-Instruct` with the database id, schema, question,
   and JSON-only output rules.
4. Extract the first valid JSON object from generation and canonicalize it
   against the schema.

Selected model and adapter settings:

| Component | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 3 |
| Learning rate | `2e-4` |
| Decoding | greedy, `max_new_tokens=256` |

Post-processing removes hallucinated tables and columns, matches identifiers
case-insensitively, deduplicates columns, and emits schema-canonical casing.
This is important because invalid identifiers count as false positives during
grading.

## Training Methodology

Training examples come from `train.json`. Each example provides:

- `question_id`
- `db_id`
- natural-language `question`
- `gold_sql`
- gold `schema_links`

The model is trained to produce `schema_links`, not SQL. For each example, the
Spider-format schema is converted into compact text:

```text
TABLE_1(col_a, col_b, col_c)
TABLE_2(col_d, col_e)
```

The prompt uses a fixed chat format with a system instruction to return only
valid JSON. The completion is the gold schema-link object serialized with sorted
keys. We did not add data augmentation.

Experiments were run with RapidFire AI multi-config sweeps over base model,
epoch count, learning rate, LoRA rank, and LoRA target modules. Training-time
loss and token accuracy were used for diagnostics, while model selection was
based on `eval.py` schema-linking metrics.

## Results

### Model Selection

The first comparison used rank-8 LoRA on `q_proj + v_proj` with learning rate
`1e-4`.

| Model | Epochs | Table Score | Column Score | Overall Score |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1 | 0.4619 | 0.2882 | 0.3750 |
| Qwen2.5-1.5B | 2 | 0.4283 | 0.2928 | 0.3606 |
| Qwen2.5-1.5B | 3 | 0.4994 | 0.3724 | 0.4359 |
| Llama3.2-1B | 1 | 0.1169 | 0.0909 | 0.1039 |
| Llama3.2-1B | 2 | 0.2150 | 0.1408 | 0.1779 |
| SmolLM2-1.7B | 1 | 0.2589 | 0.1508 | 0.2049 |
| SmolLM2-1.7B | 2 | 0.2903 | 0.1720 | 0.2312 |

Qwen2.5-1.5B was clearly strongest, especially on table detection, so later
sweeps focused on Qwen.

### Learning Rate and Epoch Sweep

All runs below use Qwen2.5-1.5B with rank-8 LoRA on `q_proj + v_proj`.

| Epochs | LR | Table Score | Column Score | Overall Score |
|---:|---:|---:|---:|---:|
| 1 | `5e-4` | 0.4669 | 0.3977 | 0.4323 |
| 1 | `2e-4` | 0.4167 | 0.2748 | 0.3458 |
| 1 | `1e-4` | 0.4619 | 0.2882 | 0.3750 |
| 1 | `5e-5` | 0.4487 | 0.2763 | 0.3625 |
| 1 | `1e-5` | 0.0490 | 0.0667 | 0.0578 |
| 2 | `5e-4` | 0.5134 | 0.4110 | 0.4622 |
| 2 | `2e-4` | 0.5114 | 0.4104 | 0.4609 |
| 2 | `1e-4` | 0.4283 | 0.2928 | 0.3606 |
| 2 | `5e-5` | 0.4609 | 0.3094 | 0.3852 |
| 2 | `1e-5` | 0.2891 | 0.1977 | 0.2434 |
| 3 | `5e-4` | 0.5280 | 0.4133 | 0.4707 |
| 3 | `2e-4` | **0.5566** | **0.4421** | **0.4993** |
| 3 | `1e-4` | 0.4994 | 0.3724 | 0.4359 |

The best validation result came from 3 epochs and learning rate `2e-4`. Very
low learning rates underfit badly, while `5e-4` was strong but slightly worse
than `2e-4` after longer training.

### LoRA Ablation

These runs use Qwen2.5-1.5B, 3 epochs, and learning rate `2e-4`.

| Config | Rank | Target Modules | Table Score | Column Score | Overall Score |
|---|---:|---|---:|---:|---:|
| qv | 8 | `q_proj + v_proj` | **0.5566** | **0.4421** | **0.4993** |
| qv | 16 | `q_proj + v_proj` | 0.5200 | 0.4047 | 0.4623 |
| qvko | 8 | `q_proj + v_proj + k_proj + o_proj` | 0.5252 | 0.3870 | 0.4561 |
| qvko | 16 | `q_proj + v_proj + k_proj + o_proj` | 0.5553 | 0.4352 | 0.4953 |

The simpler rank-8 `q_proj + v_proj` adapter performed best. Increasing adapter
capacity did not consistently improve generalization on this small dataset.

## Analysis

Base model choice was the largest factor. Qwen2.5-1.5B substantially outscored
Llama3.2-1B and SmolLM2-1.7B on both table and column metrics, likely because it
followed the JSON-only instruction more reliably and handled schema grounding
better.

Learning rate and epoch count were the next most important knobs. Column-level
metrics improved most from longer training at a moderate learning rate: the
column score rose from 0.2748 at 1 epoch with `2e-4` to 0.4421 at 3 epochs with
the same learning rate. Table-level predictions were easier; exact column
selection remained the main source of errors.

The LoRA ablation showed that bigger adapters are not automatically better.
Rank 16 with more target modules nearly matched the best score, but the smaller
rank-8 query/value adapter was still best. This suggests the dataset is small
enough that extra adapter capacity can overfit training patterns without
improving exact schema-link generalization.

## Dataset Provenance

The natural-language questions, SQL queries, and database schemas are drawn from
SNAILS (Luoma & Kumar, SIGMOD 2025), an artifact suite developed at UCSD
ADALab. Gold schema links are extracted automatically from SQL using
`sqlglot` parsing plus schema-aware column qualification.
