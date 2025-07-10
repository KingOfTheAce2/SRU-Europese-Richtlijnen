import os
import json
import time
import logging
from typing import List, Set, Optional
import requests
from lxml import etree
from bs4 import BeautifulSoup
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login

# --- Configuration ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_CONNECTION = "eur"
SRU_VERSION = "2.0"
HTTP_ACCEPT = "application/xml"

# CELEX checkpoint file
CHECKPOINT_FILE = "processed_celex.json"
# Batch size for fetching SRU records and for HF pushes
PAGE_SIZE = 100
# Delay seconds between SRU/EUR-Lex requests
REQUEST_DELAY = 1.0
# Retry settings for EUR-Lex fetch
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5
TIMEOUT = 30

# Hugging Face dataset repo
HF_REPO_ID = "vGassen/Dutch-European-Directives"
HF_SPLIT = "train"

# EUR-Lex content URL template
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{celex}"

# Fixed source field for HF dataset
DATA_SOURCE = "EU richtlijnen"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# XML namespaces for SRU + DC terms
NS = {
    'sru': 'http://docs.oasis-open.org/ns/search-ws/sruResponse',
    'dcterms': 'http://purl.org/dc/terms/'
}

# --- State Management ---
def load_processed() -> Set[str]:
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_processed(processed: Set[str]):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted(processed), f, indent=2)

# --- SRU crawler ---
def fetch_celex_sru(start: int = 1) -> List[str]:
    celexs: Set[str] = set()
    cursor = start
    total = None
    while True:
        params = {
            'x-connection': SRU_CONNECTION,
            'operation': 'searchRetrieve',
            'version': SRU_VERSION,
            'query': 'cql.allRecords=1',
            'startRecord': cursor,
            'maximumRecords': PAGE_SIZE,
            'httpAccept': HTTP_ACCEPT
        }
        logger.info(f"Fetching SRU records: start={cursor}")
        resp = requests.get(SRU_ENDPOINT, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        root = etree.fromstring(resp.content)
        if total is None:
            num = root.find('.//sru:numberOfRecords', namespaces=NS)
            total = int(num.text) if num is not None else None
            logger.info(f"Total records: {total}")
        records = root.findall('.//sru:record', namespaces=NS)
        if not records:
            break
        for rec in records:
            el = rec.find('.//dcterms:identifier', namespaces=NS)
            if el is not None and el.text:
                celexs.add(el.text.strip())
        count = len(records)
        cursor += count
        if total and cursor > total:
            break
        time.sleep(REQUEST_DELAY)
    logger.info(f"Discovered {len(celexs)} CELEX IDs")
    return sorted(celexs)

# --- EUR-Lex fetch ---
def get_with_retries(url: str) -> Optional[requests.Response]:
    for i in range(RETRY_ATTEMPTS):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Retry {i+1}/{RETRY_ATTEMPTS} failed: {e}")
            time.sleep(RETRY_BACKOFF)
    return None


def strip_html(html: str) -> str:
    return BeautifulSoup(html, 'lxml').get_text(separator='\n', strip=True)

# --- Hugging Face integration ---
def init_hf_dataset():
    token = os.getenv('HF_TOKEN')
    if not token:
        logger.error("HF_TOKEN not set")
        raise RuntimeError("Missing HF_TOKEN")
    login(token=token)
    api = HfApi()
    try:
        api.dataset_info(HF_REPO_ID)
        ds = load_dataset(HF_REPO_ID, split=HF_SPLIT)
        logger.info(f"Loaded existing dataset with {len(ds)} records")
    except Exception:
        ds = None
    return api, ds


def push_to_hf(new_data: List[dict], existing_ds, api: HfApi):
    ds_new = Dataset.from_list(new_data)
    combined = concatenate_datasets([existing_ds, ds_new]) if existing_ds else ds_new
    combined.push_to_hub(HF_REPO_ID, private=False)
    return combined

# --- Main Pipeline ---
def main():
    logger.info("--- Starting SRU→EUR-Lex→HF pipeline ---")
    processed = load_processed()
    all_celex = fetch_celex_sru()
    to_process = [c for c in all_celex if c not in processed]
    logger.info(f"{len(to_process)} new CELEX IDs to process")

    api, existing_ds = init_hf_dataset()
    batch_data: List[dict] = []

    for idx, celex in enumerate(to_process, 1):
        url = EURLEX_URL.format(celex=celex)
        resp = get_with_retries(url)
        processed.add(celex)
        if not resp:
            continue
        content = strip_html(resp.text)
        if len(content) < 50:
            continue
        batch_data.append({
            'url': url,
            'content': content,
            'source': DATA_SOURCE
        })

        # push every PAGE_SIZE records or at end
        if len(batch_data) >= PAGE_SIZE or idx == len(to_process):
            logger.info(f"Uploading {len(batch_data)} records to Hugging Face...")
            existing_ds = push_to_hf(batch_data, existing_ds, api)
            batch_data.clear()
            save_processed(processed)

        time.sleep(REQUEST_DELAY)

    logger.info("Pipeline complete.")

if __name__ == '__main__':
    main()
