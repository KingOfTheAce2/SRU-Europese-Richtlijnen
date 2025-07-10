import os
import json
import time
import logging
from typing import List, Dict, Set
import requests
from lxml import etree
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login

# --- Configuration ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_CONNECTION = "eur"
SRU_VERSION = "2.0"
HTTP_ACCEPT = "application/xml"

# CELEX checkpoint file
CHECKPOINT_FILE = "processed_celex.json"
# Page size for SRU requests and batch size for HF pushes
PAGE_SIZE = 100
# Delay seconds between SRU requests
REQUEST_DELAY = 1.0
TIMEOUT = 30

# Hugging Face dataset repo and split
HF_REPO_ID = "vGassen/Dutch-European-Directives"
HF_SPLIT = "train"

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
def fetch_sru_records(start: int = 1) -> List[Dict]:
    records_list: List[Dict] = []
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
        rec_elems = root.findall('.//sru:record', namespaces=NS)
        if not rec_elems:
            break
        for rec in rec_elems:
            # extract CELEX identifier
            celex_el = rec.find('.//dcterms:identifier', namespaces=NS)
            celex = celex_el.text.strip() if celex_el is not None and celex_el.text else None
            # raw XML of the recordData element
            data_el = rec.find('.//sru:recordData', namespaces=NS)
            content = etree.tostring(data_el, encoding='utf-8').decode('utf-8') if data_el is not None else None
            if celex and content:
                records_list.append({'url': celex, 'content': content, 'source': DATA_SOURCE})
        count = len(rec_elems)
        cursor += count
        if total and cursor > total:
            break
        time.sleep(REQUEST_DELAY)
    logger.info(f"Fetched {len(records_list)} SRU records")
    return records_list

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


def push_to_hf(batch: List[Dict], existing_ds, api: HfApi):
    ds_batch = Dataset.from_list(batch)
    if existing_ds:
        combined = concatenate_datasets([existing_ds, ds_batch])
    else:
        combined = ds_batch
    combined.push_to_hub(HF_REPO_ID, private=False)
    return combined

# --- Main Pipeline ---
def main():
    logger.info("--- Starting SRU→HF pipeline ---")
    processed = load_processed()
    all_records = fetch_sru_records()
    new_records = [r for r in all_records if r['url'] not in processed]
    logger.info(f"{len(new_records)} new records to upload")

    api, existing_ds = init_hf_dataset()
    batch: List[Dict] = []
    for idx, rec in enumerate(new_records, 1):
        batch.append(rec)
        processed.add(rec['url'])
        # push every PAGE_SIZE or at end
        if len(batch) >= PAGE_SIZE or idx == len(new_records):
            logger.info(f"Uploading {len(batch)} records to Hugging Face...")
            existing_ds = push_to_hf(batch, existing_ds, api)
            save_processed(processed)
            batch.clear()
    logger.info("Pipeline complete.")

if __name__ == '__main__':
    main()
