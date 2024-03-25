# Tool Collection

A collection of Python scripts and shell scripts that utilize the GPT-4 language model to process and analyze text files, particularly scientific papers. These tools are designed to automate various tasks such as summarization, reformatting, and extraction of specific information from the text.

## Scripts

### 1. `read_paper.py`

This Python script performs agglomerative information extraction and summarization on a directory of text files. It processes the files, splits them into paragraphs or sentences, classifies the sentences into predefined topics using a Language Model (LLM), and generates summaries for each topic and the overall document.

### 2. `embed_dist.py`

This Python script calculates the Euclidean distance and cosine similarity between two string embeddings. It utilizes an API to generate the embeddings for the provided strings.

### 3. `summarize_paper.py`

This Python script summarizes the content of a file or a directory of files using an API. It breaks down the content into chunks, sends each chunk to the API for summarization, and then concatenates the summaries to generate a final summary.

### 4. `rewrite.py`

This Python script processes a file or a directory of files, applying a specific instruction to the content using an API. It breaks down the content into chunks, sends each chunk to the API for processing, and concatenates the processed chunks to generate the final output.

### 5. `go_linux.sh`

This Bash script takes a natural language prompt as input and sends it to an API endpoint using curl. The API response, which is expected to be a precise Linux command based on the input prompt, is then displayed in the terminal.

### 6. `rewrite-line.py`

This Python script processes a file line by line, applying a specific instruction to each line using an API. It sends each line to the API for processing and stores the original line and the corresponding output in a JSON format.

## Usage

Detailed usage instructions for each script can be found in their respective sections in this README.

## Dependencies

- Python 3.x
- argparse
- os
- json
- requests
- numpy
- scikit-learn
- curl
- jq

## Example Workflow

1. Convert PDF files to text format using `pdf_to_text`.
2. Summarize the extracted text files using `summarize_paper.py` with specific instructions.
3. Rewrite the generated summaries using `rewrite.py` to group relevant applications by type and cancer type.
4. Extract verbatim prompt examples from the summaries using `rewrite.py` and create a table.
5. Extract high-level results worth highlighting using `rewrite.py`.

## Notes

- These scripts assume the existence of an API endpoint that accepts JSON payloads and returns the processed data.
- Make sure to have the necessary dependencies installed before running the scripts.
- Adjust the API endpoint URLs and request payloads based on your specific API requirements.
- Feel free to customize the scripts to suit your specific needs and use cases.

For more detailed information on each script, please refer to their respective sections in this README.
