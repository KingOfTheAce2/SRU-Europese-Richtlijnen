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

# Namespaces
NS = {
    'sru':      'http://docs.oasis-open.org/ns/search-ws/sruResponse',
    'ohrl':     'http://standaarden.overheid.nl/rl/terms/',
    'dcterms':  'http://purl.org/dc/terms/'
}

# CELEX checkpoint file (store processed CELEX IDs)
CHECKPOINT_FILE = "processed_celex.json"
# SRU pagination size
PAGE_SIZE = 100
# Politeness delays and timeouts
REQUEST_DELAY = 1.0
TIMEOUT = 30

# Hugging Face dataset config
HF_REPO_ID = "vGassen/Dutch-European-Directives"
HF_SPLIT   = "train"
DATA_SOURCE= "EU richtlijnen"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- State Management ---
def load_processed() -> Set[str]:
    if os.path.isfile(CHECKPOINT_FILE):
        try:
            return set(json.load(open(CHECKPOINT_FILE, 'r', encoding='utf-8')))
        except Exception:
            pass
    return set()


def save_processed(processed: Set[str]):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted(processed), f, indent=2)

# --- SRU Crawler: extract XML URLs for directives ---
def fetch_sru_locaties(start: int = 1) -> List[str]:
    xml_urls: List[str] = []
    cursor = start
    total: Optional[int] = None
    while True:
        params = {
            'x-connection': SRU_CONNECTION,
            'operation':    'searchRetrieve',
            'version':      SRU_VERSION,
            'query':        'cql.allRecords=1',
            'startRecord':  cursor,
            'maximumRecords': PAGE_SIZE,
            'httpAccept':   HTTP_ACCEPT
        }
        logger.info(f"Fetching SRU records: start={cursor}")
        resp = requests.get(SRU_ENDPOINT, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        root = etree.fromstring(resp.content)
        if total is None:
            num = root.find('.//sru:numberOfRecords', namespaces=NS)
            total = int(num.text) if num is not None else None
            logger.info(f"Total SRU records available: {total}")
        recs = root.findall('.//sru:record', namespaces=NS)
        if not recs:
            break
        for rec in recs:
            loc_el = rec.find('.//ohrl:locatie', namespaces=NS)
            if loc_el is not None and loc_el.text:
                xml_urls.append(loc_el.text.strip())
        count = len(recs)
        cursor += count
        if total and cursor > total:
            break
        time.sleep(REQUEST_DELAY)
    logger.info(f"Collected {len(xml_urls)} directive XML URLs")
    return xml_urls

# --- XML to CELEX number ---
def extract_celex_from_xml(xml_url: str) -> Optional[str]:
    try:
        r = requests.get(xml_url, timeout=TIMEOUT)
        r.raise_for_status()
        root = etree.fromstring(r.content)
        part = root.find('.//dcterms:isPartOf', namespaces=NS)
        if part is not None and part.text:
            return part.text.strip()
    except Exception as e:
        logger.warning(f"Failed to extract CELEX from {xml_url}: {e}")
    return None

# --- EUR-Lex HTML fetch & strip ---
def fetch_eurlex_text(celex: str) -> Optional[str]:
    url = f"https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{celex}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.content, 'lxml').get_text(separator='\n', strip=True)
    except Exception as e:
        logger.warning(f"Failed to fetch EUR-Lex for CELEX {celex}: {e}")
    return None

# --- Hugging Face integration ---
def init_hf_dataset():
    token = os.getenv('HF_TOKEN')
    if not token:
        raise RuntimeError("HF_TOKEN env var is required")
    login(token=token)
    api = HfApi()
    try:
        api.dataset_info(HF_REPO_ID)
        ds = load_dataset(HF_REPO_ID, split=HF_SPLIT)
        logger.info(f"Loaded existing dataset with {len(ds)} entries")
    except Exception:
        ds = None
    return api, ds


def push_batch(api: HfApi, existing_ds, batch: List[dict]):
    ds = Dataset.from_list(batch)
    if existing_ds is not None:
        combined = concatenate_datasets([existing_ds, ds])
    else:
        combined = ds
    combined.push_to_hub(HF_REPO_ID, private=False)
    return combined

# --- Main Pipeline ---
def main():
    logger.info("Starting full SRU→EUR-Lex pipeline")
    processed = load_processed()

    # Step 1: get all directive XML URLs
    xml_urls = fetch_sru_locaties()

    # Step 2: for each XML, get CELEX and then EUR-Lex text
    entries = []
    for xml_url in xml_urls:
        celex = extract_celex_from_xml(xml_url)
        if not celex or celex in processed:
            continue
        text = fetch_eurlex_text(celex)
        if not text or len(text) < 50:
            processed.add(celex)
            continue
        entries.append({'url': f"https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{celex}",
                         'content': text,
                         'source': DATA_SOURCE})
        processed.add(celex)
        time.sleep(REQUEST_DELAY)

    if not entries:
        logger.info("No new entries to upload")
        return

    # Step 3: upload in batches
    api, existing_ds = init_hf_dataset()
    batch: List[dict] = []
    for entry in entries:
        batch.append(entry)
        if len(batch) >= PAGE_SIZE:
            existing_ds = push_batch(api, existing_ds, batch)
            save_processed(processed)
            batch.clear()
    if batch:
        existing_ds = push_batch(api, existing_ds, batch)
        save_processed(processed)

    logger.info("Pipeline complete")

if __name__ == '__main__':
    main()
