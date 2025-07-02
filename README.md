# SRU European Directives

This repository contains a Python crawler that collects "Europese Richtlijnen" (European directives) from the Dutch governmental SRU service. For each directive the crawler follows the CELEX reference to the official EUR-Lex page and extracts the plain text. The resulting data is uploaded as a dataset to the [Hugging Face Hub](https://huggingface.co/).

## Features

- Fetches records from the SRU search endpoint in batches.
- Extracts the CELEX number for each record and downloads the corresponding text from EUR-Lex.
- Uses a checkpoint file to remember the last fetched record so subsequent runs only process new data.
- Limits each run to a maximum of 250 records before uploading to Hugging Face.
- Scheduled weekly via GitHub Actions or can be run locally.

## Usage

1. Install dependencies:
   ```bash
   pip install requests pandas beautifulsoup4 lxml datasets huggingface_hub
   ```
2. Set the environment variable `HF_TOKEN` with a valid Hugging Face access token.
3. Run the crawler:
   ```bash
   python run_crawl.py
   ```

The script will store the next starting record in `checkpoint.txt` after a successful upload.

## Workflow

The GitHub Actions workflow (`.github/workflows/main.yml`) installs the dependencies and runs `run_crawl.py` once a week. The token used for authentication is provided through the `HF_TOKEN` secret.

## License

This project is licensed under the [MIT License](LICENSE).
