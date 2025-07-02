import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfFolder

CHECKPOINT_FILE = "checkpoint.txt"
MAX_RECORDS_PER_RUN = 250

def load_checkpoint(file_path: str = CHECKPOINT_FILE) -> int:
    """Load the starting record from a checkpoint file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                value = int(f.read().strip())
                return value
        except Exception:
            pass
    return 1

def save_checkpoint(next_record: int, file_path: str = CHECKPOINT_FILE) -> None:
    """Save the next starting record to a checkpoint file."""
    with open(file_path, "w") as f:
        f.write(str(next_record))

def strip_tags(text: str) -> str:
    """
    Removes all XML/HTML tags from a string, leaving plain text.
    Uses BeautifulSoup for robust parsing.
    """
    # Use 'lxml' for speed, fall back to 'html.parser' if needed
    return BeautifulSoup(text, "lxml").get_text(separator="\n", strip=True)

def fetch_records(start_record: int = 1, limit: int = MAX_RECORDS_PER_RUN):
    """Crawl the SRU endpoint and return up to ``limit`` records starting from
    ``start_record``. The function stops early if fewer records are available."""
    base_url = "https://zoekservice.overheid.nl/sru/Search"
    params = {
        "x-connection": "eur",
        "operation": "searchRetrieve",
        "version": "2.0",
        "query": "cql.allRecords=1",
        "startRecord": start_record,
        "maximumRecords": min(100, limit),
        "httpAccept": "application/xml",
    }
    
    # Define namespaces to correctly parse the XML with lxml
    namespaces = {
        'sru': 'http://docs.oasis-open.org/ns/search-ws/sruResponse',
        'gzd': 'http://standaarden.overheid.nl/sru',
        'dcterms': 'http://purl.org/dc/terms/'
    }

    rows = []
    total_records_processed = 0

    print("Starting crawl of 'Europese Richtlijnen'...")

    while total_records_processed < limit:
        print(f"Fetching records starting from position {params['startRecord']}...")
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()  # Raise an exception for bad status codes

            xml_content = resp.content
            if not xml_content:
                print("Received empty response from server. Stopping.")
                break
                
            root = etree.fromstring(xml_content)

            # Find all records in the current response
            records = root.findall('.//sru:record', namespaces=namespaces)
            if not records:
                print("No more records found. Crawl finished.")
                break

            for record in records:
                # Extract the CELEX number
                celex_elem = record.find('.//dcterms:isPartOf', namespaces=namespaces)
                celex_number = celex_elem.text if celex_elem is not None else None

                # Construct the official EUR-Lex URL based on the CELEX number
                celex_url = (
                    f"https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri={celex_number}"
                    if celex_number
                    else ''
                )

                # Download the full text from the EUR-Lex page
                content = ''
                if celex_url:
                    try:
                        page = requests.get(celex_url, timeout=30)
                        page.raise_for_status()
                        content = strip_tags(page.text)
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to retrieve CELEX page {celex_url}: {e}")

                rows.append({
                    "URL": celex_url,
                    "Content": content,
                    "Source": "Europese Richtlijnen"
                })

                # Be polite to the EUR-Lex server
                time.sleep(0.5)
            
            total_records_processed += len(records)
            print(f"Successfully processed {len(records)} records. Total processed: {total_records_processed}")

            # Determine if there are more records to fetch
            num_records_elem = root.find('.//sru:numberOfRecords', namespaces=namespaces)
            if num_records_elem is None:
                print("Warning: Could not find 'numberOfRecords' in the response. Stopping.")
                break
            
            total_records = int(num_records_elem.text)

            next_start = params["startRecord"] + len(records)

            if total_records_processed >= limit:
                break

            if next_start > total_records:
                print("All records have been retrieved.")
                break

            params["startRecord"] = next_start
            params["maximumRecords"] = min(100, limit - total_records_processed)

            # Be polite to the server by waiting between requests
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
            break
        except etree.XMLSyntaxError as e:
            print(f"Error parsing XML: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    return rows, params["startRecord"]

def main():
    """
    Main function to run the crawl, process the data, and upload to Hugging Face.
    """
    # --- Step 1: Authentication ---
    # The Hugging Face token is retrieved from an environment variable.
    # In GitHub Actions, this is set from a repository secret.
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    # Log in to Hugging Face
    HfFolder.save_token(hf_token)
    print("Successfully authenticated with Hugging Face Hub.")

    # --- Step 2: Crawl and Process Data ---
    start_record = load_checkpoint()
    records_data, next_start = fetch_records(start_record=start_record, limit=MAX_RECORDS_PER_RUN)

    if not records_data:
        print("No data was collected. Exiting.")
        return

    # --- Step 3: Create DataFrame and Dataset ---
    df = pd.DataFrame(records_data)
    print(f"\nCreated DataFrame with {len(df)} rows and {len(df.columns)} columns.")
    print("DataFrame Info:")
    df.info()
    print("\nSample of 5 records:")
    print(df.head())

    # Convert the pandas DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # --- Step 4: Upload to Hugging Face Hub ---
    repo_id = "vGassen/Dutch-European-Directives"
    print(f"\nUploading dataset to Hugging Face Hub at {repo_id}...")

    try:
        dataset.push_to_hub(repo_id, private=False)  # Set private=True if you want
        print("Dataset successfully uploaded!")
        save_checkpoint(next_start)
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()
