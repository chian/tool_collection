# read_paper.py Agglomerative Information Extraction and Summarization

This Python script performs agglomerative information extraction and summarization on a directory of text files. It processes the files, splits them into paragraphs or sentences, classifies the sentences into predefined topics using a Language Model (LLM), and generates summaries for each topic and the overall document.

## Features

- Processes multiple text files in a directory
- Splits files into paragraphs or sentences
- Classifies sentences into predefined topics using an LLM
- Generates summaries for each topic and the overall document
- Utilizes concurrent processing for improved performance
- Implements retry mechanisms and error handling for API calls
- Saves and loads checkpoints to resume processing from the last processed file

## Dependencies

- Python 3.x
- argparse
- os
- json
- numpy
- time
- concurrent.futures
- retrying
- langchain
- crewai
- rich
- typing
- llama_cpp

## Usage

1. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script with the desired arguments:

   ```bash
   python read_paper.py --directory <path_to_directory> [--paragraph] [--specific_instruction <instruction>]
   ```

   - `--directory`: Path to the directory containing the text files to be processed (required).
   - `--paragraph`: Enable paragraph processing mode (optional).
   - `--specific_instruction`: Add a specific instruction for more targeted processing (optional).

3. The script will process the files in the specified directory, generate summaries, and display the results in the console.

4. If the script is interrupted or stopped, it will save a checkpoint. When restarted, it will resume processing from the last processed file.

## Example

```bash
python read_paper.py --directory /path/to/text/files --paragraph --specific_instruction "Focus on extracting information related to infectious disease outbreaks."
```

This command will process the text files in the `/path/to/text/files` directory, enable paragraph processing mode, and focus on extracting information related to infectious disease outbreaks.

## Notes

- The script utilizes an LLM for sentence classification and summary generation. Make sure to set up the appropriate API credentials and endpoints for the LLM service being used.
- The script saves checkpoints after processing each file. If the script is interrupted, it will resume from the last processed file when restarted.
- The script includes retry mechanisms and error handling for API calls to handle potential network issues or service unavailability.
- Adjust the `DEBUG` variable to control the verbosity of the output and logging.
- Modify the `NUM_LLM_THREADS` variable to control the number of concurrent threads used for LLM calls.

Feel free to explore and modify the code to suit your specific requirements and use case.
