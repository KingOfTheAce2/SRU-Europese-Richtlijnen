import time
import requests
from lxml import etree

# — Configuration —
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
CONNECTION   = "eur"          # the EUR collection
VERSION      = "2.0"
HTTP_ACCEPT  = "application/xml"
PAGE_SIZE    = 100            # max 1000 allowed by SRU, but 100 is polite
PAUSE_SEC    = 1.0            # polite delay between calls

# SRU XML namespaces
NS = {
    'sru':     'http://docs.oasis-open.org/ns/search-ws/sruResponse',
    'dcterms': 'http://purl.org/dc/terms/'
}

def crawl_eur_collection():
    """Yield all CELEX identifiers from the EUR SRU collection."""
    start_record = 1
    total_records = None

    while True:
        params = {
            "x-connection":   CONNECTION,
            "operation":      "searchRetrieve",
            "version":        VERSION,
            "query":          "cql.allRecords=1",  # match everything in this collection
            "startRecord":    start_record,
            "maximumRecords": PAGE_SIZE,
            "httpAccept":     HTTP_ACCEPT
        }
        resp = requests.get(SRU_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()

        root = etree.fromstring(resp.content)
        # on first page, grab total number of hits
        if total_records is None:
            total_elem = root.find(".//sru:numberOfRecords", namespaces=NS)
            total_records = int(total_elem.text) if total_elem is not None else 0
            print(f"Total records in EUR collection: {total_records}")

        records = root.findall(".//sru:record", namespaces=NS)
        if not records:
            break

        for rec in records:
            # dcterms:identifier contains the CELEX number
            id_el = rec.find(".//dcterms:identifier", namespaces=NS)
            if id_el is not None and id_el.text:
                yield id_el.text.strip()

        start_record += len(records)
        if start_record > total_records:
            break

        time.sleep(PAUSE_SEC)

if __name__ == "__main__":
    seen = set()
    for celex in crawl_eur_collection():
        if celex not in seen:
            seen.add(celex)
            print(celex)
