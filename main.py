"""
main.py -- Inference entry point for Project 2 schema linking.

This script mirrors the CLI contract from sample_main.py, but adds adapter
selection so you can evaluate different LoRA / QLoRA configs stored under
./adapter/.

Examples
--------
Use a specific adapter subdirectory:
    python main.py \
        --input validation_input.json \
        --output preds.json \
        --adapter adapter/qwen2.5-1.5b-lora-rank8-1epoch-lr1e-4

Use a root adapter directory containing exactly one adapter:
    python main.py \
        --input validation_input.json \
        --output preds.json

The output format is a JSON list of:
    {"question_id": <int>, "schema_links": {"Table": ["Col1", ...]}}
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a schema linking assistant. "
    "Given a database schema and a natural language question, "
    "identify the database tables and columns needed to answer the question. "
    "Return only valid JSON. Do not include explanations."
)


def db_id_to_schema_path(db_id: str, schemas_dir: Path) -> Path:
    filename = db_id.replace(" ", "_").replace("/", "_") + ".json"
    return schemas_dir / filename


@lru_cache(maxsize=None)
def load_schema_as_dict(db_id: str, schemas_dir: str) -> dict[str, list[str]]:
    schema_path = db_id_to_schema_path(db_id, Path(schemas_dir))
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_json = json.load(f)

    table_names = schema_json["table_names_original"]
    schema = {table_name: [] for table_name in table_names}

    for table_idx, column_name in schema_json["column_names_original"]:
        if table_idx == -1:
            continue
        schema[table_names[table_idx]].append(column_name)

    return schema


def compact_schema_text(schema: dict[str, list[str]]) -> str:
    """Format schema as:
        TABLE(col1, col2)
        TABLE2(col1, ...)
    """
    lines = []
    for table_name, columns in schema.items():
        columns_text = ", ".join(columns)
        lines.append(f"{table_name}({columns_text})")
    return "\n".join(lines)


def build_user_prompt(db_id: str, schema_text: str, question: str) -> str:
    return f"""Database:
{db_id}

Schema:
{schema_text}

Question:
{question}

Output format:
{{"TABLE_NAME": ["COLUMN_NAME_1", "COLUMN_NAME_2"], "TABLE_ONLY": []}}

Rules:
- Use only table and column names that appear in the schema.
- Include a table with an empty list if the table is referenced but no specific column is needed.
- Do not include SQL.
- Return only the JSON object."""


def format_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Some tokenizers expose slightly different signatures.
            return tokenizer.apply_chat_template(messages, tokenize=False)
    return (
        f"{system_prompt}\n\n"
        f"{user_prompt}\n\n"
        f"Assistant:\n"
    )


def resolve_adapter_path(adapter_root: str) -> Path:
    root = Path(adapter_root)
    if (root / "adapter_config.json").exists():
        return root

    if not root.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {root}. "
            "Point --adapter to an adapter directory or an adapter root."
        )

    child_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "adapter_config.json").exists()
    ]
    if len(child_dirs) == 1:
        return child_dirs[0]
    if not child_dirs:
        raise FileNotFoundError(
            f"No adapter_config.json found under {root}. "
            "Point --adapter to a specific adapter directory."
        )
    raise ValueError(
        f"Multiple adapters found under {root}; point --adapter to a specific subdirectory. "
        f"Available configs: {[p.name for p in child_dirs]}"
    )


def available_adapter_names(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "adapter_config.json").exists()
    )


def load_model_and_tokenizer(adapter_path: Path):
    with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto",
        "trust_remote_code": True,
    }

    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    except TypeError:
        # Older transformers versions may not support `dtype` yet.
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = "auto"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    model.generation_config = generation_config
    model.eval()
    return model, tokenizer


def extract_json_object(text: str):
    """Extract the first JSON object embedded in model output."""
    text = text.strip()
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def canonicalize_schema_links(raw_links, schema: dict[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(raw_links, dict):
        return {}

    lc_tables = {table.lower(): table for table in schema}
    lc_cols = {table: {col.lower(): col for col in cols} for table, cols in schema.items()}

    out = {}
    for table_name, cols in raw_links.items():
        table_key = str(table_name).strip().lower()
        if table_key not in lc_tables:
            continue

        canonical_table = lc_tables[table_key]
        canonical_cols = []
        if isinstance(cols, list):
            seen = set()
            for col_name in cols:
                col_key = str(col_name).strip().lower()
                if col_key in lc_cols[canonical_table] and col_key not in seen:
                    canonical_cols.append(lc_cols[canonical_table][col_key])
                    seen.add(col_key)

        out[canonical_table] = canonical_cols

    return {table: out[table] for table in sorted(out)}


def predict_schema_links(question: str, db_id: str, schemas_dir: str, model, tokenizer):
    schema = load_schema_as_dict(db_id, schemas_dir)
    schema_text = compact_schema_text(schema)
    user_prompt = build_user_prompt(db_id, schema_text, question)
    prompt = format_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device) if "attention_mask" in inputs else None

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    parsed = extract_json_object(generated_text)
    if parsed is None:
        return {}

    return canonicalize_schema_links(parsed, schema)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument(
        "--adapter",
        default="./adapter",
        help="Adapter directory or adapter root containing one or more config subfolders.",
    )
    args = ap.parse_args()

    adapter_path = resolve_adapter_path(args.adapter)
    model, tokenizer = load_model_and_tokenizer(adapter_path)

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    preds = []
    for it in tqdm(items, desc="Predicting", unit="question"):
        links = predict_schema_links(
            question=it["question"],
            db_id=it["db_id"],
            schemas_dir=args.schemas_dir,
            model=model,
            tokenizer=tokenizer,
        )
        preds.append({
            "question_id": it["question_id"],
            "schema_links": links,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(preds)} predictions to {args.output}")
    print(f"Used adapter: {adapter_path}")


if __name__ == "__main__":
    main()
