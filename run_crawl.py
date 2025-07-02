import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder

def strip_tags(text: str) -> str:
    """
    Removes all XML/HTML tags from a string, leaving plain text.
    Uses BeautifulSoup for robust parsing.
    """
    # Use 'lxml' for speed, fall back to 'html.parser' if needed
    return BeautifulSoup(text, "lxml").get_text(separator="\n", strip=True)

def fetch_all_records():
    """
    Crawls the SRU endpoint for 'Europese Richtlijnen', handling pagination.
    """
    base_url = "https://zoekservice.overheid.nl/sru/Search"
    params = {
        "x-connection": "eur",
        "operation": "searchRetrieve",
        "version": "2.0",
        "query": "cql.allRecords=1",
        "startRecord": 1,
        "maximumRecords": 100,  # Fetch 100 records per request
        "httpAccept": "application/xml"
    }
    
    # Define namespaces to correctly parse the XML with lxml
    namespaces = {
        'sru': 'http://docs.oasis-open.org/ns/search-ws/sruResponse',
        'gzd': 'http://standaarden.overheid.nl/sru'
    }

    rows = []
    total_records_processed = 0
    
    print("Starting crawl of 'Europese Richtlijnen'...")

    while True:
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
                # Extract the preferred URL
                url_elem = record.find('.//gzd:preferredUrl', namespaces=namespaces)
                url = url_elem.text if url_elem is not None else ''

                # Extract the full record data for cleaning
                recorddata_elem = record.find('.//sru:recordData', namespaces=namespaces)
                if recorddata_elem is not None:
                    # Convert the recordData element back to a string for cleaning
                    raw_xml = etree.tostring(recorddata_elem, encoding='unicode')
                    content = strip_tags(raw_xml)
                else:
                    content = ''

                rows.append({
                    "URL": url,
                    "Content": content,
                    "Source": "Europese Richtlijnen"
                })
            
            total_records_processed += len(records)
            print(f"Successfully processed {len(records)} records. Total processed: {total_records_processed}")

            # Determine if there are more records to fetch
            num_records_elem = root.find('.//sru:numberOfRecords', namespaces=namespaces)
            if num_records_elem is None:
                print("Warning: Could not find 'numberOfRecords' in the response. Stopping.")
                break
            
            total_records = int(num_records_elem.text)
            
            # Move to the next page
            params["startRecord"] += params["maximumRecords"]

            if params["startRecord"] > total_records:
                print("All records have been retrieved.")
                break
            
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
            
    return rows

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
    records_data = fetch_all_records()

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
        dataset.push_to_hub(repo_id, private=False) # Set private=True if you want
        print("Dataset successfully uploaded!")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()
