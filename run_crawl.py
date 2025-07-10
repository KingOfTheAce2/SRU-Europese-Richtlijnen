import os
import json
import time
import logging
from typing import List, Set, Optional
import requests
from lxml import etree
from bs4 import BeautifulSoup

# --- Configuration ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_CONNECTION = "eur"
SRU_VERSION = "2.0"
HF_CHECKPOINT = "processed_celex.json"
BATCH_SIZE = 100
REQUEST_DELAY = 1.0
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5  # seconds
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{celex}"
TIMEOUT = 30

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- State Management ---
def load_processed() -> Set[str]:
    if os.path.isfile(HF_CHECKPOINT):
        try:
            with open(HF_CHECKPOINT, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return set()


def save_processed(processed: Set[str]):
    try:
        with open(HF_CHECKPOINT, 'w', encoding='utf-8') as f:
            json.dump(sorted(processed), f, indent=2)
    except Exception as e:
        logger.error(f"Could not save checkpoint: {e}")

# --- SRU Fetch ---
NS = {
    'sru': 'http://docs.oasis-open.org/ns/search-ws/sruResponse',
    'dcterms': 'http://purl.org/dc/terms/'
}

def fetch_celex_sru(start: int = 1, max_records: int = 1000) -> List[str]:
    celex_set: Set[str] = set()
    fetched = 0
    cursor = start
    while fetched < max_records:
        batch = min(100, max_records - fetched)
        params = {
            'x-connection': SRU_CONNECTION,
            'operation': 'searchRetrieve',
            'version': SRU_VERSION,
            'query': 'cql.allRecords=1',
            'startRecord': cursor,
            'maximumRecords': batch,
            'httpAccept': 'application/xml'
        }
        logger.info(f"SRU fetch: start={cursor} batch={batch}")
        try:
            resp = requests.get(SRU_ENDPOINT, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            root = etree.fromstring(resp.content)
            records = root.findall('.//sru:record', namespaces=NS)
            if not records:
                break
            for rec in records:
                part = rec.find('.//dcterms:isPartOf', namespaces=NS)
                if part is not None and part.text:
                    celex_set.add(part.text.strip())
            fetched += len(records)
            cursor += len(records)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"Error fetching SRU records: {e}")
            break
    logger.info(f"Discovered {len(celex_set)} unique CELEX numbers")
    return sorted(celex_set)

# --- EUR-Lex Fetch ---
def get_with_retries(url: str, attempts: int = RETRY_ATTEMPTS) -> Optional[requests.Response]:
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logger.warning(f"Attempt {i+1}/{attempts} failed for {url}: {e}")
            time.sleep(RETRY_BACKOFF)
    logger.error(f"All retries failed for {url}")
    return None


def strip_html(html: str) -> str:
    return BeautifulSoup(html, 'lxml').get_text(separator='\n', strip=True)

# --- Main ---
def main():
    logger.info("Starting CELEX crawler...")
    processed = load_processed()
    all_celex = fetch_celex_sru(max_records=1000)
    to_process = [c for c in all_celex if c not in processed]
    logger.info(f"{len(to_process)} new CELEX numbers to process")

    for celex in to_process:
        url = EURLEX_URL.format(celex=celex)
        resp = get_with_retries(url)
        if not resp:
            continue
        content = strip_html(resp.text)
        if len(content) < 100:
            logger.info(f"Skipped {celex}: content too short")
            processed.add(celex)
            continue
        # Here you can process or store the content, e.g., write to file or push to dataset
        fname = f"data/{celex}.txt"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved content for CELEX {celex}")
        processed.add(celex)
        save_processed(processed)
        time.sleep(REQUEST_DELAY)

    logger.info("Crawling complete.")

if __name__ == '__main__':
    main()
