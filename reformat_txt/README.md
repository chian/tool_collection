# LinuXpert Shell Script

The `go_linux.sh` script is a Bash script that takes a natural language prompt as input and sends it to an API endpoint using curl. The API response, which is expected to be a precise Linux command based on the input prompt, is then displayed in the terminal.

## Usage

```bash
./go_linux.sh "<prompt>"
```

- `<prompt>`: The natural language prompt describing the desired Linux command or task.

## Example

```bash
./go_linux.sh "How do I find files containing the word 'error' in the logs directory?"
```

This command will send the prompt to the API endpoint and display the corresponding Linux command returned by the API.

## Notes

- The script assumes the existence of an API endpoint at `https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/` that accepts a JSON payload containing the prompt and returns the corresponding Linux command.
- The API response is expected to be in JSON format, and the script extracts the `response` field from the JSON using `jq`.
- The script uses `curl` to send the POST request to the API endpoint.
- If the API call fails or the `curl` command is unsuccessful, an error message will be displayed, and the script will exit with a non-zero status code.

Make sure to have `curl` and `jq` installed on your system before running this script.

# Text Rewriting Line by Line

The `rewrite-line.py` script is a Python script that processes a file line by line, applying a specific instruction to each line using an API. It sends each line to the API for processing and stores the original line and the corresponding output in a JSON format.

## Dependencies

- Python 3.x
- argparse
- os
- json
- requests

## Usage

```bash
python rewrite-line.py <input_path> [--specific_instruction <instruction>]
```

- `<input_path>`: Path to the file to be processed line by line (required).
- `--specific_instruction`: Specific instruction to be applied to each line (optional, default: "Format the following text nicely.").

## Example

```bash
python rewrite-line.py input.txt --specific_instruction "Reformat table as a json format."
```

This command will process the `input.txt` file line by line, sending each line to the API with the instruction "Reformat table as a json format." The script will display the processed lines and store them in a JSON format.

## Notes

- The script assumes the existence of an API endpoint at `https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/` that accepts a JSON payload containing the line and the specific instruction and returns the processed line.
- The API response is expected to be in JSON format, and the script extracts the `response` field from the JSON.
- If the API call fails or returns a non-200 status code, an error message will be displayed.
- The script processes the file line by line and stores the original line and the corresponding output in a JSON format.
- The resulting JSON is printed to the console.

Feel free to modify the script to suit your specific requirements and API integration needs.

# Text Rewriting

The `rewrite.py` script is a Python script that processes a file or a directory of files, applying a specific instruction to the content using an API. It breaks down the content into chunks, sends each chunk to the API for processing, and concatenates the processed chunks to generate the final output.

## Dependencies

- Python 3.x
- argparse
- os
- json
- requests

## Usage

```bash
python rewrite.py <input_path> [--specific_instruction <instruction>]
```

- `<input_path>`: Path to the file or directory to be processed (required).
- `--specific_instruction`: Specific instruction to be applied to the content (optional, default: "Format the following text nicely.").

## Example

```bash
python rewrite.py input.txt --specific_instruction "Reformat the content as a structured report."
```

This command will process the `input.txt` file, sending its content to the API with the instruction "Reformat the content as a structured report." The script will display the processed content.

## Notes

- The script assumes the existence of an API endpoint at `https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/` that accepts a JSON payload containing the content and the specific instruction and returns the processed content.
- The API response is expected to be in JSON format, and the script extracts the `response` field from the JSON.
- If the API call fails or returns a non-200 status code, an error message will be displayed.
- The script breaks down the content into chunks of a specified size (default: 3500 words) and sends each chunk to the API for processing.
- The processed chunks are concatenated to generate the final output.
- If a directory is provided as the input path, the script will process all the files in that directory.

Feel free to customize the script based on your specific requirements and API integration needs.
