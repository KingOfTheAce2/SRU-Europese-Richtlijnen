import os
import json
import time
import requests
from typing import List, Dict, Set
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login

# --- Configuration ---

HF_DATASET_REPO = "vGassen/Dutch-European-Directives"
CHECKPOINT_FILE = "processed_celex_numbers.json"
BATCH_SIZE = 200
GLOBAL_RUN_LIMIT = 2000  # Max records per script run
REQUEST_DELAY = 1.5  # Seconds between requests to EUR-Lex
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5
SRU_SOURCE = "Europese Richtlijnen"
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri={celex}"

# --- State Management Functions ---

def load_processed_celex() -> Set[str]:
    """Loads the set of already processed CELEX numbers from the checkpoint file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(json.load(f))
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Starting fresh.")
        return set()

def save_processed_celex(processed: Set[str]):
    """Saves the updated set of processed CELEX numbers to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(sorted(list(processed)), f, indent=2)
    except Exception as e:
        print(f"FATAL: Could not save checkpoint! Error: {e}")

# --- SRU Harvesting Functions ---

SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_NAMESPACES = {
    'sru': 'http://docs.oasis-open.org/ns/search-ws/sruResponse',
    'gzd': 'http://standaarden.overheid.nl/sru',
    'dcterms': 'http://purl.org/dc/terms/'
}

def fetch_sru_records(start: int = 1, limit: int = 2000) -> List[str]:
    """Fetches SRU records and extracts unique CELEX numbers."""
    celex_found = set()
    total_processed = 0
    current = start
    while total_processed < limit:
        print(f"Fetching SRU records from {current}...")
        params = {
            "x-connection": "eur",
            "operation": "searchRetrieve",
            "version": "2.0",
            "query": "cql.allRecords=1",
            "startRecord": current,
            "maximumRecords": min(100, limit - total_processed),
            "httpAccept": "application/xml",
        }
        try:
            r = requests.get(SRU_ENDPOINT, params=params, timeout=30)
            r.raise_for_status()
            root = etree.fromstring(r.content)
            records = root.findall('.//sru:record', namespaces=SRU_NAMESPACES)
            if not records:
                print("No more SRU records found.")
                break
            for record in records:
                celex_elem = record.find('.//dcterms:isPartOf', namespaces=SRU_NAMESPACES)
                if celex_elem is not None and celex_elem.text:
                    celex_found.add(celex_elem.text)
            total_processed += len(records)
            next_start = current + len(records)
            num_records_elem = root.find('.//sru:numberOfRecords', namespaces=SRU_NAMESPACES)
            if num_records_elem is None or next_start > int(num_records_elem.text):
                break
            current = next_start
            time.sleep(1)
        except Exception as e:
            print(f"SRU fetch error: {e}. Stopping early.")
            break
    print(f"Discovered {len(celex_found)} unique CELEX numbers from SRU.")
    return sorted(list(celex_found))

# --- EUR-Lex Content Fetching Functions ---

def get_with_retries(url: str) -> requests.Response | None:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url} (Attempt {attempt + 1}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    print(f"Giving up on {url}")
    return None

def strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup, keep text structure."""
    return BeautifulSoup(text, "lxml").get_text(separator="\n", strip=True)

def fetch_eurlex_content(celex: str) -> str | None:
    url = EURLEX_URL.format(celex=celex)
    resp = get_with_retries(url)
    if not resp:
        return None
    return strip_html(resp.text)

# --- Hugging Face Upload Logic ---

def push_batch_to_hf(batch_data: List[Dict[str, str]], existing_dataset, repo_id: str):
    """Pushes a batch to Hugging Face and combines with previous data."""
    ds = Dataset.from_list(batch_data)
    if existing_dataset:
        combined = concatenate_datasets([existing_dataset, ds])
    else:
        combined = ds
    combined.push_to_hub(repo_id, private=False)
    return combined

# --- Main Execution ---

def main():
    print("--- Starting SRU Europese Richtlijnen Scraper ---")
    # 1. Authenticate with Hugging Face
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Please set HF_TOKEN env variable for Hugging Face access.")
        return
    try:
        login(token=token)
        print("Authenticated with Hugging Face.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    api = HfApi()
    # Try to load existing dataset
    existing_dataset = None
    try:
        api.dataset_info(HF_DATASET_REPO)
        existing_dataset = load_dataset(HF_DATASET_REPO, split="train")
        print(f"Loaded {len(existing_dataset)} existing records.")
    except Exception:
        print("No existing dataset found, or cannot load it. Proceeding fresh.")

    # 2. Load state
    processed_celex = load_processed_celex()
    print(f"Found {len(processed_celex)} processed CELEX numbers in checkpoint.")

    # 3. Harvest SRU records (CELEX numbers)
    all_celex = fetch_sru_records(start=1, limit=GLOBAL_RUN_LIMIT)
    celex_to_process = [c for c in all_celex if c not in processed_celex]
    if not celex_to_process:
        print("No new CELEX numbers to process. Exiting.")
        return
    if len(celex_to_process) > GLOBAL_RUN_LIMIT:
        celex_to_process = celex_to_process[:GLOBAL_RUN_LIMIT]

    print(f"{len(celex_to_process)} new CELEX numbers will be processed.")

    # 4. Fetch and push in batches
    for i in range(0, len(celex_to_process), BATCH_SIZE):
        batch = celex_to_process[i:i + BATCH_SIZE]
        print(f"\n--- Processing batch {i//BATCH_SIZE+1}/{(len(celex_to_process)-1)//BATCH_SIZE+1} ---")
        batch_data = []
        for celex in batch:
            url = EURLEX_URL.format(celex=celex)
            content = fetch_eurlex_content(celex)
            if content and len(content) > 50:
                batch_data.append({
                    "URL": url,
                    "Content": content,
                    "Source": SRU_SOURCE
                })
            else:
                print(f"  Skipping CELEX {celex}: content too short or missing.")
            time.sleep(REQUEST_DELAY)
        if not batch_data:
            print("Batch resulted in no data. Skipping push.")
            continue
        try:
            print(f"Pushing batch of {len(batch_data)} records to Hugging Face...")
            existing_dataset = push_batch_to_hf(batch_data, existing_dataset, HF_DATASET_REPO)
            processed_celex.update(batch)
            save_processed_celex(processed_celex)
            print("Checkpoint updated.")
        except Exception as e:
            print(f"FATAL: Failed to upload batch. Error: {e}")
            print("Stopping to avoid data loss.")
            break
    print("All batches processed. Script finished.")

if __name__ == "__main__":
    main()
