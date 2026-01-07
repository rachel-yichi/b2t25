"""
Optional stage 3: use an external chat-completions LLM to lightly clean/adjust
phoneme-to-text outputs so they read more naturally. This script takes the CSV
from stage2_phoneme2text_t5_infer.py, calls a proxy-compatible OpenAI-style
endpoint, and writes a new CSV with the edited text.

Example:
python stage2_llm_cleanup.py \
  --input_csv /path/to/p2t_t5_test_predicted_sentences.csv \
  --output_csv /path/to/p2t_t5_test_predicted_sentences_llm.csv \
  --model_id your-model \
  --api_base https://$BASE_URL/v1/ \
  --api_key $API_KEY
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package not installed. pip install openai>=1.0.0")


SYS_PROMPT = (
    "You are a conservative editor for ASR hypotheses. Your job is to make the "
    "smallest possible change: only fix obvious misspellings or duplicated words. "
    "Do NOT add or remove words, do NOT change word order, do NOT paraphrase. "
    "If you are unsure, return the input as-is. Return only the final sentence. "
    "If the input is empty, return an empty string."
)


def normalize_text(txt: str) -> str:
    """Simple normalization: decode, strip, lower, remove punctuation and extra spaces."""
    if txt is None:
        return ""
    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode()
    txt = str(txt)
    txt = txt.replace("\n", " ").replace("\t", " ")
    txt = txt.lower()
    # keep letters, numbers, apostrophes, hyphens, spaces
    txt = "".join(ch if (ch.isalnum() or ch in [" ", "'", "-"]) else " " for ch in txt)
    txt = " ".join(txt.strip().split())
    return txt


def call_llm(
    client: OpenAI, model_id: str, text: str, temperature: float, max_tokens: int
) -> str:
    """Call LLM with a guarded prompt; return cleaned text."""
    if not text.strip():
        return ""

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Hypothesis: {text}"},
    ]

    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM cleanup of P2T outputs")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with id,text")
    parser.add_argument("--output_csv", type=str, required=True, help="CSV to write cleaned text")
    parser.add_argument("--model_id", type=str, required=True, help="Model to send to proxy")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL like https://host/v1/")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Single API key (or set env B2TXT_API_KEY); ignored if --api_keys is provided",
    )
    parser.add_argument(
        "--api_keys",
        type=str,
        default=None,
        help="Comma-separated API keys to enable parallel calls (or env B2TXT_API_KEYS)",
    )
    parser.add_argument(
        "--api_keys_file",
        type=str,
        default=None,
        help="Path to a txt file containing one API key per line (blank lines ignored)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--max_tokens", type=int, default=64, help="Max tokens returned by the LLM"
    )
    parser.add_argument(
        "--sleep_on_error",
        type=float,
        default=1.0,
        help="Seconds to sleep after a recoverable error before retrying",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries per request on transient errors",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Thread pool size; defaults to number of API keys provided",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # collect API keys
    key_str = args.api_keys or os.getenv("B2TXT_API_KEYS")
    api_keys: List[str] = []
    if args.api_keys_file and os.path.exists(args.api_keys_file):
        with open(args.api_keys_file, "r", encoding="utf-8") as f:
            api_keys = [line.strip() for line in f if line.strip()]
    elif key_str:
        api_keys = [k.strip() for k in key_str.split(",") if k.strip()]
    else:
        single_key = args.api_key or os.getenv("B2TXT_API_KEY")
        if single_key:
            api_keys = [single_key]

    if not api_keys:
        raise ValueError("API key(s) missing. Provide --api_keys / --api_key or env B2TXT_API_KEYS / B2TXT_API_KEY.")

    api_base = args.api_base or os.getenv("B2TXT_API_BASE")
    def build_client(key: str) -> OpenAI:
        client_kwargs = {"api_key": key}
        if api_base:
            client_kwargs["base_url"] = api_base
        return OpenAI(**client_kwargs)

    clients = [build_client(k) for k in api_keys]

    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    cleaned = [""] * len(df)
    num_workers = args.num_workers or len(clients)

    def process(idx: int, text: str):
        client = clients[idx % len(clients)]  # round-robin across provided keys
        hyp = str(text)
        retries = 0
        while True:
            try:
                return call_llm(
                    client=client,
                    model_id=args.model_id,
                    text=hyp,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                retries += 1
                if retries > args.max_retries:
                    print(f"[WARN] row {idx} failed after retries: {e}", file=sys.stderr)
                    return hyp
                time.sleep(args.sleep_on_error)

    progress = tqdm(total=len(df), desc="LLM cleanup", leave=False) if tqdm else None
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process, i, df.iloc[i]["text"]): i for i in range(len(df))}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                cleaned[i] = normalize_text(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] row {i} failed unexpectedly: {e}", file=sys.stderr)
                cleaned[i] = normalize_text(df.iloc[i]["text"])
            if progress:
                progress.update(1)
    if progress:
        progress.close()

    df_out = pd.DataFrame({"id": df["id"], "text": cleaned})
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved cleaned outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
