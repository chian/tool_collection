# xummarize_paper.py: Paper Summarization

This Python script summarizes the content of a file or a directory of files using an API. It breaks down the content into chunks, sends each chunk to the API for summarization, and then concatenates the summaries to generate a final summary.

## Dependencies

- Python 3.x
- argparse
- os
- json
- requests

## Usage

1. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script with the desired arguments:

   ```bash
   python summarize_paper.py <input_path> [--specific_instruction <instruction>]
   ```

   - `<input_path>`: Path to the file or directory containing the files to summarize (required).
   - `--specific_instruction`: Additional prompt for more specific instructions (optional).

3. The script will process each file in the specified directory or the single file provided.

4. For each file, the script will:
   - Read the file content and split it into chunks of a specified size (default: 3500 words).
   - Send each chunk to the API for summarization.
   - Concatenate the summaries of all chunks.
   - Send the concatenated summary to the API for a final summarization.
   - Print the final summary for the file.

5. If the API call is successful, the script will display the final summary for each file.

6. If the API call fails or returns a non-200 status code, an error message will be displayed.

## Example

```bash
python summarize_paper.py ./papers --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techniques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?"
```

This command will summarize the content of all files in the `./papers` directory, providing the specific instruction to highlight answers to the given questions.

## Notes

- The script assumes the existence of an API endpoint that accepts a JSON payload containing the text to summarize and returns the corresponding summary.
- Replace the API endpoint URL (`'https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/'`) with your actual API endpoint.
- The script expects the API response to be a JSON object containing a `'response'` field, which should contain the generated summary.
- Adjust the API request payload and response handling based on your specific API requirements.
- The chunk size (`chunk_size`) can be modified according to your needs. It determines the number of words per chunk.

Feel free to customize the script to fit your specific use case and API integration requirements.
