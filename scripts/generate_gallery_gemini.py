#!/usr/bin/env python3
"""
Generate `gallery_openai.csv` by enriching image metadata with an OpenAI vision model.

The script reads source rows from the input CSV, asks the model to provide
structured descriptions for selected fields, and writes the combined output to
the target CSV file. Whenever information cannot be determined from the image,
the value `nan` is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
    from openai import OpenAIError
except ImportError as exc:
    raise SystemExit(
        "The `openai` package is required. Install with `pip install openai`."
    ) from exc

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError as exc:
    raise SystemExit(
        "The `pydantic` package is required. Install with `pip install pydantic`."
    ) from exc


# OPTIMIZATION: Updated prompts for refined headstone-focused schema.
DEFAULT_SYSTEM_PROMPT = """You are an expert content analyst specializing in stonecraft photography for a Polish manufacturer.
Output MUST be a single valid JSON object.
Language: Polish.
Unknown/uncertain → exact 'nan'.
Respect incoming "category" and DO NOT change it.
If category == "nagrobek", include an object "headstone" with extended attributes described in the user prompt. For other categories set "headstone" to 'nan'.
Use controlled vocabularies where provided; if outside, choose the closest match or 'nan'.
Optimize "caption" and "tags" for RAG semantic search (natural, bez emoji/hasztagów)."""

DEFAULT_USER_PROMPT_TEMPLATE = """User prompt
Analyze the provided stonecraft product photo and metadata.

Input data (JSON):
{record_summary}

Return a single JSON object with EXACTLY these keys:
- "color": Dominujący kolor/kolory kamienia (krótko).
- "stone": Rodzaj kamienia (kontrolowane lub 'nan').
- "finish": Typ wykończenia (kontrolowane lub 'nan').
- "surrounding": Jedno zdanie o kontekście ujęcia.
- "caption": Jedno zdanie marketingowe (≤ 20 słów).
- "tags": Lista 3–6 słów/krótkich fraz wspierających wyszukiwanie semantyczne (małe litery).
- "headstone": 
    Jeśli "category" w danych wejściowych == "nagrobek", zwróć OBIEKT z polami:
      - "style": ["nowoczesny","klasyczny","minimalistyczny","rustykalny","tradycyjny"] lub 'nan'
      - "letter_material": ["mosiądz","stal nierdzewna","brąz","aluminium","złoto","farba złota","farba biała","farba czarna"] lub 'nan'
      - "letter_technique": ["litery nakładane 3d","piaskowane","grawerowane","kute"] lub 'nan'
      - "has_photo_on_headboard": "tak" | "nie" | "nan"
      - "photo_type": ["zdjęcie porcelanowe","grawer laserowy","druk uv","nan"]
      - "grave_type": ["pojedynczy","podwójny","rodzinny","urnowy","dziecięcy"] lub 'nan'
      - "headboard_shape": ["prosty","łuk","fala","nieregularny"] lub 'nan'
      - "cover_type": ["pełna płyta","ramy z płytą środkową","kostka","żwir"] lub 'nan'
      - "accessories": lista 0–5 elementów z ["wazon","lampion","ławka","krzyż","krzyż nierdzewny","krzyż mosiężny"] (gdy brak → [])
    W przeciwnym razie ustaw "headstone": "nan".

Constraints:
- JSON bez dodatkowych pól i komentarzy, bez trailing commas.
- Jeżeli coś jest niewidoczne/niepewne → 'nan'.
- "tags" nie duplikują danych słownikowych 1:1; mają wspierać zapytania typu „ze zdjęciem na tablicy”, „litery mosiężne”, „nowoczesny”."""

OUTPUT_COLUMNS = [
    "id",
    "image_url",
    "thumb_url",
    "category",
    "color",
    "stone",
    "finish",
    "surrounding",
    "caption",
    "tags",
    "headstone",
]
MODEL_COLUMNS = [
    "color",
    "stone",
    "finish",
    "surrounding",
    "caption",
    "tags",
    "headstone",
]
MANDATORY_INPUT_COLUMNS = ["id", "image_url", "category"]
DEFAULT_FILL_VALUE = "nan"

PROJECT_ROOT = Path(__file__).resolve().parent


class RunConfig(BaseModel):
    """Configuration model for the script, validated by Pydantic."""

    input_path: str = Field(default="data/gallery_test.csv")
    output_path: str = Field(default="data/gallery_openai.csv")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    delay: float = Field(default=1.0, ge=0.0)
    max_retries: int = Field(default=3, ge=0)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    user_prompt_template: str = Field(default=DEFAULT_USER_PROMPT_TEMPLATE)
    only_missing: bool = Field(default=False)
    throttle: float = Field(default=0.5, ge=0.0)

    model_config = ConfigDict(extra="forbid")


# --- Helper Functions ---


RETRY_HINT_RE = re.compile(
    r"try again in\s+(?P<value>[\d\.]+)\s*(?P<unit>ms|milliseconds|s|seconds)",
    re.IGNORECASE,
)


def _extract_retry_after_seconds(message: str) -> Optional[float]:
    match = RETRY_HINT_RE.search(message)
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit.startswith("ms"):
        return value / 1000.0
    return value


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "http_status", None)
    if status == 429:
        return True
    text = str(error).lower()
    return "rate limit" in text or "429" in text


def _compute_backoff_seconds(base_delay: float, attempt: int, error: Exception) -> float:
    text = str(error)
    retry_after = _extract_retry_after_seconds(text)
    sleep_seconds = max(base_delay * max(attempt, 1), 0.0)
    if retry_after is not None:
        sleep_seconds = max(sleep_seconds, retry_after)
    elif _is_rate_limit_error(error):
        sleep_seconds = max(sleep_seconds, 0.8 * attempt)
    if sleep_seconds <= 0:
        return 0.0
    jitter = random.uniform(0.05, 0.25)
    return min(sleep_seconds + jitter, 30.0)


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolves a string path relative to the project root."""
    if not path_str:
        return None
    candidate = Path(path_str)
    return (
        candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    )


def load_prompt(path: Optional[str], default_value: str) -> str:
    """Loads a prompt from a file or returns the default value."""
    resolved = resolve_path(path)
    if resolved and resolved.exists():
        return resolved.read_text(encoding="utf-8").strip()
    return default_value


def load_existing_output(path: Optional[Path]) -> dict[str, dict[str, str]]:
    """Loads existing rows from the output file into a lookup map."""
    if not path or not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return {
            key: dict(row)
            for row in reader
            if (key := row.get("id") or row.get("image_url"))
        }


def normalize_result(result: dict[str, object]) -> dict[str, str]:
    """Ensures model output values are strings and formats tags correctly."""
    normalized: dict[str, str] = {}
    for column in MODEL_COLUMNS:
        value = result.get(column)
        if column == "tags" and isinstance(value, list):
            normalized[column] = ", ".join(str(item).strip() for item in value if item)
        elif column == "headstone":
            if isinstance(value, dict):
                normalized[column] = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                )
            else:
                normalized[column] = str(value) if value is not None else ""
        else:
            normalized[column] = str(value) if value is not None else ""
    return normalized


def fill_defaults(row: dict[str, str]) -> dict[str, str]:
    """Fills missing values in a row with the default placeholder."""
    filled = {}
    for column in OUTPUT_COLUMNS:
        value = str(row.get(column, "")).strip()
        filled[column] = value if value else DEFAULT_FILL_VALUE
    return filled


def merge_rows(
    base: dict[str, str], *others: Optional[dict[str, str]]
) -> dict[str, str]:
    """Merges multiple rows, prioritizing values from later rows, except for category."""
    merged = dict(base)
    for other in others:
        if not other:
            continue
        for key, value in other.items():
            text = str(value).strip()
            if text and text.lower() != DEFAULT_FILL_VALUE:
                merged[key] = text
    merged["category"] = base.get("category", DEFAULT_FILL_VALUE)
    return merged


def has_all_model_values(row: dict[str, str]) -> bool:
    """Checks if a row has all its model-generated fields filled."""
    category = (row.get("category") or "").strip().lower()
    for column in MODEL_COLUMNS:
        value = (row.get(column) or "").strip()
        if not value:
            return False
        if value.lower() == DEFAULT_FILL_VALUE:
            if column == "headstone" and category != "nagrobek":
                continue
            return False
    return True


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Writes the final list of rows to a CSV file."""
    if not rows:
        logging.warning("No data to write to output file.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows([fill_defaults(row) for row in rows])


# --- Core API Logic ---


def create_client() -> OpenAI:
    """Creates and returns an OpenAI client instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY environment variable. Please set it in a .env file."
        )
    return OpenAI(api_key=api_key)


def request_analysis(
    client: OpenAI, config: RunConfig, row: dict[str, str]
) -> dict[str, str]:
    """
    Sends a request to the OpenAI API to analyze an image and returns the structured data.

    OPTIMIZATION: This function now uses the modern `client.chat.completions.create` method
    and enables JSON Mode for reliable, structured output, removing the need for manual parsing.
    """
    record_summary = json.dumps(
        {
            "id": row.get("id"),
            "image_url": row.get("image_url"),
            "category": row.get("category"),
        },
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = config.user_prompt_template.format(record_summary=record_summary)

    messages = [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": row["image_url"]}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        response_format={"type": "json_object"},  # Enable JSON mode
        messages=messages,
    )

    response_content = response.choices[0].message.content
    if not response_content:
        raise ValueError("Received an empty response from the model.")

    parsed = json.loads(response_content)
    return normalize_result(parsed)


# --- Main Application Class ---


class GalleryPipeline:
    def __init__(self, config: RunConfig, client: OpenAI):
        self.config = config
        self.client = client

    def run(self) -> int:
        input_path = resolve_path(self.config.input_path)
        output_path = resolve_path(self.config.output_path)

        if not input_path or not input_path.exists():
            logging.error(
                "Input path '%s' is invalid or does not exist.", self.config.input_path
            )
            return 1

        source_rows = self._load_source_rows(input_path)
        existing_map = load_existing_output(output_path)

        final_rows: List[Dict[str, str]] = []

        for row in source_rows:
            key = row.get("id") or row.get("image_url")
            if not key:
                logging.warning("Skipping row without id or image_url: %s", row)
                continue

            base_row = self._create_baseline_row(row)
            existing_row = existing_map.pop(key, None)
            merged = merge_rows(base_row, existing_row)

            if self.config.only_missing and has_all_model_values(merged):
                final_rows.append(merged)
                continue

            updates = self._analyze_with_retry(merged)
            final_rows.append(merge_rows(base_row, existing_row, updates))
            if self.config.throttle > 0:
                time.sleep(self.config.throttle)

        # Add rows that were in the old output but not in the new input
        final_rows.extend(existing_map.values())

        write_csv(output_path, final_rows)
        logging.info("Wrote %d rows to %s", len(final_rows), output_path)
        return 0

    def _load_source_rows(self, path: Path) -> List[Dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as infile:
            return list(csv.DictReader(infile))

    def _create_baseline_row(self, row: dict[str, str]) -> dict[str, str]:
        baseline = {}
        for col in MANDATORY_INPUT_COLUMNS:
            if not row.get(col):
                raise SystemExit(f"Missing required column '{col}' in row: {row}")
            baseline[col] = row[col]
        baseline["thumb_url"] = row.get("thumb_url", DEFAULT_FILL_VALUE)
        return baseline

    def _analyze_with_retry(self, row: dict[str, str]) -> dict[str, str]:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logging.info(
                    "Analyzing %s (attempt %d/%d)...",
                    row.get("id", "N/A"),
                    attempt,
                    self.config.max_retries,
                )
                if self.config.delay > 0 and attempt > 1:
                    time.sleep(self.config.delay)
                return request_analysis(self.client, self.config, row)
            except (OpenAIError, ValueError, json.JSONDecodeError) as error:
                logging.warning(
                    "Model call for ID %s failed on attempt %d: %s",
                    row.get("id", "N/A"),
                    attempt,
                    error,
                )
                if attempt < self.config.max_retries:
                    sleep_for = _compute_backoff_seconds(
                        self.config.delay, attempt, error
                    )
                    if sleep_for > 0:
                        logging.info(
                            "Waiting %.2f s before retrying %s due to rate limit/error.",
                            sleep_for,
                            row.get("id", "N/A"),
                        )
                        time.sleep(sleep_for)
        logging.error(
            "Exceeded retry limit for ID %s. Continuing with placeholders.",
            row.get("id", "N/A"),
        )
        return {}


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Analyze gallery images with an OpenAI vision model."
    )
    parser.add_argument(
        "--input", default="data/gallery_test.csv", help="Path to source CSV."
    )
    parser.add_argument(
        "--output",
        default="data/gallery_openai.csv",
        help="Path for enriched CSV output.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Vision model name.")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Model sampling temperature."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds to wait between retries."
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per row on failure."
    )
    parser.add_argument(
        "--system-prompt-file", help="File with a custom system prompt."
    )
    parser.add_argument(
        "--user-prompt-file", help="File with a custom user prompt template."
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip rows that are already fully populated.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.5,
        help="Seconds to wait after each successful model call (rate-limit friendly).",
    )

    args = parser.parse_args()

    return RunConfig(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        temperature=args.temperature,
        delay=args.delay,
        max_retries=args.max_retries,
        system_prompt=load_prompt(args.system_prompt_file, DEFAULT_SYSTEM_PROMPT),
        user_prompt_template=load_prompt(
            args.user_prompt_file, DEFAULT_USER_PROMPT_TEMPLATE
        ),
        only_missing=args.only_missing,
        throttle=args.throttle,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()  # Load environment variables from a .env file

    try:
        config = parse_args()
        client = create_client()
        pipeline = GalleryPipeline(config, client)
        return pipeline.run()
    except (SystemExit, Exception) as e:
        logging.error(f"A critical error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
