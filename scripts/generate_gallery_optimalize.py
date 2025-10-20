#!/usr/bin/env python3
"""
Produkcyjna, uproszczona wersja pipeline'u analizy zdjęć.
Zachowuje dotychczasową logikę: czyta data/gallery_category.csv -> zapisuje data/gallery_openai.csv.
Dodatkowo:
- chroni przed ponowną analizą zdjęć już obecnych w gallery_openai.csv (domyślnie włączone),
- dodaje elastyczność wyboru providera (obecnie wspierany: openai),
- uproszcza klienta jako warstwę pośrednią (extendable dla innych dostawców).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - helps diagnose missing dependency
    raise SystemExit(
        "The `openai` package is required. Install with `pip install openai`."
    ) from exc

try:
    from openai import OpenAIError  # type: ignore
except ImportError:
    OpenAIError = Exception  # type: ignore[assignment]

try:
    from pydantic import BaseModel, Field, validator

    try:
        from pydantic import ConfigDict  # type: ignore
    except Exception:
        ConfigDict = None
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The `pydantic` package is required. Install with `pip install pydantic`."
    ) from exc

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior content specialist at a Polish stonecraft company. "
    "Your task is to analyse a product photo together with minimal metadata and "
    "deliver concise, production-ready descriptions. "
    "Always reply with strict JSON that matches the requested schema and write "
    "'nan' when the image does not provide enough information."
)

DEFAULT_USER_PROMPT_TEMPLATE = """\
Przeanalizuj poniższe dane produktu kamieniarskiego oraz zdjęcie, jeśli jest dostępne.
Informacje wejściowe (JSON):
{record_summary}

Na wyjściu zwróć obiekt JSON z następującymi kluczami:
- color: dominujący kolor lub zestaw kolorów kamienia (po polsku).
- stone: prawdopodobny rodzaj kamienia (po polsku); jeśli niepewny, zapisz 'nan'.
- finish: rodzaj wykończenia powierzchni (np. poler, mat, szczotkowany) w języku polskim.
- surrounding: krótkie zdanie opisujące kontekst lub otoczenie produktu (po polsku).
- caption: zwięzły, marketingowy opis jednego zdania (po polsku).
- tags: lista 3-6 krótkich słów kluczowych po polsku.

Nie zmieniaj kategorii z danych wejściowych. Gdy nie da się ustalić odpowiedzi, zwróć 'nan'.
"""

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
]

MODEL_COLUMNS = ["color", "stone", "finish", "surrounding", "caption", "tags"]

MANDATORY_INPUT_COLUMNS = ["id", "image_url", "category"]
DEFAULT_FILL_VALUE = "nan"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


class RunConfig(BaseModel):
    input_path: str = Field(default="data/gallery_category.csv")
    output_path: str = Field(default="data/gallery_openai.csv")
    model: str = Field(default="gpt-4o-mini")
    model_provider: str = Field(default="openai")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    delay: float = Field(default=0.5, ge=0.0)
    max_retries: int = Field(default=3, ge=0)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    user_prompt_template: str = Field(default=DEFAULT_USER_PROMPT_TEMPLATE)
    only_missing: bool = Field(default=False)
    # New: do not re-request analysis for rows already present in output CSV
    never_rerequest_existing: bool = Field(default=True)

    if "ConfigDict" in globals() and ConfigDict is not None:  # pragma: no cover
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"

    @validator("model_provider")
    def provider_must_be_known(cls, v: str) -> str:
        providers = {"openai", "google", "anthropic"}
        if v not in providers:
            raise ValueError(
                f"Unsupported provider '{v}'. Supported: {', '.join(sorted(providers))}"
            )
        return v


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_prompt(path: Optional[str], default_value: str) -> str:
    if not path:
        return default_value
    resolved = resolve_path(path)
    if not resolved or not resolved.exists():
        return default_value
    with open(resolved, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    return content or default_value


def load_existing_output(
    path: Optional[Path],
) -> tuple[List[Dict[str, str]], List[str]]:
    if not path or not path.exists() or not path.is_file():
        return [], []
    with open(path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def select_row_key(row: Dict[str, str]) -> Optional[str]:
    return row.get("id") or row.get("image_url")


def build_user_message(template: str, row: Dict[str, str]) -> str:
    record = {
        "id": row.get("id", DEFAULT_FILL_VALUE),
        "image_url": row.get("image_url", DEFAULT_FILL_VALUE),
        "category": row.get("category", DEFAULT_FILL_VALUE),
    }
    summary = json.dumps(record, ensure_ascii=False, indent=2)
    return template.format(record_summary=summary)


def extract_response_text(response: object) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    output = getattr(response, "output", None)
    if output:
        for item in output:
            contents = getattr(item, "content", None)
            if not contents:
                continue
            for content in contents:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", None)
                    if text:
                        return text
    raise ValueError("Response did not contain textual output.")


def normalize_result(result: Dict[str, object]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for column in MODEL_COLUMNS:
        value = result.get(column)
        if column == "tags":
            if isinstance(value, list):
                normalized[column] = ", ".join(
                    str(item).strip() for item in value if item
                )
            else:
                normalized[column] = str(value) if value is not None else ""
        else:
            normalized[column] = str(value) if value is not None else ""
    return normalized


def fill_defaults(row: Dict[str, str], columns: Iterable[str]) -> Dict[str, str]:
    filled: Dict[str, str] = {}
    for column in columns:
        value = row.get(column)
        if value is None:
            filled[column] = DEFAULT_FILL_VALUE
            continue
        if isinstance(value, (int, float)):
            filled[column] = str(value)
            continue
        text = str(value).strip()
        filled[column] = text if text else DEFAULT_FILL_VALUE
    return filled


def merge_rows(
    base: Dict[str, str], *others: Optional[Dict[str, str]]
) -> Dict[str, str]:
    merged = dict(base)
    for other in others:
        if not other:
            continue
        for key, value in other.items():
            if key == "category":
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() == DEFAULT_FILL_VALUE:
                continue
            merged[key] = text
    merged["category"] = base.get("category", DEFAULT_FILL_VALUE)
    return merged


def has_all_model_values(row: Dict[str, str]) -> bool:
    for column in MODEL_COLUMNS:
        value = (row.get(column) or "").strip()
        if not value or value.lower() == DEFAULT_FILL_VALUE:
            return False
    return True


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    sanitized_rows = [fill_defaults(row, fieldnames) for row in rows]
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanitized_rows)


# Lightweight provider abstraction layer
class VisionClient:
    def analyze(self, row: Dict[str, str], config: RunConfig) -> Dict[str, str]:
        raise NotImplementedError


class OpenAIVisionClient(VisionClient):
    def __init__(self, api_key: str, openai_model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model = openai_model_name

    def analyze(self, row: Dict[str, str], config: RunConfig) -> Dict[str, str]:
        user_message = build_user_message(config.user_prompt_template, row)
        user_content: List[Dict[str, str]] = [
            {"type": "input_text", "text": user_message}
        ]
        image_url = row.get("image_url")
        if image_url:
            user_content.append({"type": "input_image", "image_url": image_url})

        response = self.client.responses.create(
            model=self.model,
            temperature=config.temperature,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": config.system_prompt}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        raw_text = extract_response_text(response)
        cleaned = raw_text.strip()
        # unwrap fenced codeblocks and optional leading 'json'
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("` \n")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
        logging.debug("CLEANED: %r", cleaned)
        parsed = json.loads(cleaned)
        return normalize_result(parsed)


def create_vision_client(provider: str, model: str) -> VisionClient:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "Missing OPENAI_API_KEY environment variable for OpenAI provider."
            )
        return OpenAIVisionClient(api_key=api_key, openai_model_name=model)
    # placeholders for future providers
    raise SystemExit(f"Provider '{provider}' is not implemented yet.")


class GalleryPipeline:
    def __init__(self, config: RunConfig, client: VisionClient):
        self.config = config
        self.client = client

    def run(self) -> int:
        input_path = resolve_path(self.config.input_path)
        output_path = resolve_path(self.config.output_path)
        if not input_path or not input_path.exists():
            raise SystemExit("Input path is invalid or does not exist.")
        if not output_path:
            raise SystemExit("Output path is invalid.")

        source_rows = self._load_input_rows(input_path)
        existing_rows, _ = load_existing_output(output_path)
        # map existing by key -> skip re-analysis if requested
        existing_map = {
            key: row
            for row in existing_rows
            if (key := select_row_key(row)) is not None
        }

        processed: List[Dict[str, str]] = []
        for raw_row in source_rows:
            key = select_row_key(raw_row)
            base_row = self._baseline_row(raw_row)
            existing_row = existing_map.pop(key, None) if key else None

            # If policy says never re-request existing, just retain the existing row as-is
            if existing_row and self.config.never_rerequest_existing:
                merged = merge_rows(base_row, existing_row)
                processed.append(fill_defaults(merged, OUTPUT_COLUMNS))
                logging.debug("Skipping analysis for %s (present in output)", key)
                continue

            merged = merge_rows(base_row, existing_row)

            # If only_missing is set and row already has all model values -> skip analysing
            if self.config.only_missing and has_all_model_values(merged):
                processed.append(fill_defaults(merged, OUTPUT_COLUMNS))
                continue

            updates = self._analyze_with_retry(merged)
            merged = merge_rows(base_row, existing_row, updates)
            processed.append(fill_defaults(merged, OUTPUT_COLUMNS))

        # Append residual existing rows that were not in source (preserve)
        for residual in existing_map.values():
            baseline = self._baseline_row(residual)
            processed.append(fill_defaults(baseline, OUTPUT_COLUMNS))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(str(output_path), OUTPUT_COLUMNS, processed)
        logging.info("Wrote %d rows to %s", len(processed), output_path)
        return 0

    def _load_input_rows(self, input_path: Path) -> List[Dict[str, str]]:
        with open(input_path, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise SystemExit("Input CSV does not contain headers.")
            rows = [dict(row) for row in reader]
        return rows

    def _baseline_row(self, row: Dict[str, str]) -> Dict[str, str]:
        baseline: Dict[str, str] = {}
        for column in MANDATORY_INPUT_COLUMNS:
            value = (row.get(column) or "").strip()
            if not value:
                raise SystemExit(f"Missing required column '{column}' in row: {row}")
            baseline[column] = value

        thumb = (row.get("thumb_url") or "").strip()
        baseline["thumb_url"] = thumb if thumb else DEFAULT_FILL_VALUE

        for column in MODEL_COLUMNS:
            value = (row.get(column) or "").strip()
            if value:
                baseline[column] = value
        return baseline

    def _analyze_with_retry(self, row: Dict[str, str]) -> Dict[str, str]:
        attempt = 0
        while attempt < self.config.max_retries:
            attempt += 1
            try:
                logging.info(
                    "Analyzing %s (attempt %d/%d)...",
                    row.get("id", "<unknown>"),
                    attempt,
                    self.config.max_retries,
                )
                result = self.client.analyze(row, self.config)
                return result
            except (OpenAIError, ValueError, json.JSONDecodeError) as error:
                logging.warning(
                    "Model call for %s failed (%s).",
                    row.get("id", "<unknown>"),
                    error,
                )
                if attempt >= self.config.max_retries:
                    logging.error(
                        "Exceeded retry limit for %s, continuing with placeholders.",
                        row.get("id", "<unknown>"),
                    )
                    break
                time.sleep(self.config.delay)
        return {}


def parse_args(argv: Optional[Iterable[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze gallery images with a vision model and create gallery_openai.csv "
            "from gallery_category.csv."
        )
    )
    parser.add_argument("--input", default="data/gallery_category.csv")
    parser.add_argument("--output", default="data/gallery_openai.csv")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--provider", default="openai", help="Model provider: openai, google, anthropic"
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--system-prompt-file", help="Optional file with system prompt")
    parser.add_argument(
        "--user-prompt-file", help="Optional file with user prompt template"
    )
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument(
        "--never-rerequest-existing",
        action="store_true",
        help="Do not re-analyze rows already present in output CSV",
    )

    args = parser.parse_args(argv)

    system_prompt = load_prompt(args.system_prompt_file, DEFAULT_SYSTEM_PROMPT)
    user_prompt_template = load_prompt(
        args.user_prompt_file, DEFAULT_USER_PROMPT_TEMPLATE
    )

    return RunConfig(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        model_provider=args.provider,
        temperature=args.temperature,
        delay=args.delay,
        max_retries=args.max_retries,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        only_missing=args.only_missing,
        never_rerequest_existing=args.never_rerequest_existing,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()
    config = parse_args(argv)
    client = create_vision_client(config.model_provider, config.model)
    pipeline = GalleryPipeline(config, client)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
