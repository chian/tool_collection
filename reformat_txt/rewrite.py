#!/usr/bin/env python3
import argparse
import os
import json
import requests

#Example usage:
#summarize_paper ./filename-or-directory --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techiques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?" 

def process_chunk(chunk_text, specific_instruction):
    """Send a request to summarize the chunk using an API."""
    headers = {'Content-Type': 'application/json'}

    data = {"user":"chia",
            "model":"gpt35", 
            "system": "You are a helpful AI assistant.",
            "prompt":[specific_instruction + chunk_text],
            "stop":[],
            "temperature":1.0,
            "top_p":0.7}
    #print(data)
    payload = json.dumps(data)
    
    # Example API call. Replace 'https://example.com/api/summarize' with your actual API endpoint.
    try:
        response = requests.post('https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/',
                                 headers=headers, data=payload)
        #print("Status Code:", response.status_code)
        if response.status_code == 200:
            #print("JSON Response:",response.json())
            summary = response.json()
            summary = summary['response']
            # Assuming the API returns a JSON with a field 'summary'. Adjust as per your API response structure.
        else:
            summary = f"Failed to get summary, status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        summary = f"API request failed: {e}"

    return summary

def process_file(file_path, specific_instruction):
    """Process a single file to summarize its content."""
    print(f"Processing {file_path}...")
    chunk_size = 2000  # Number of words per chunk

    with open(file_path, 'r') as file:
        file_content = file.read()
    word_array = file_content.split()
    total_words = len(word_array)
    print(f"Total words: {total_words}")
    current_word = 0
    cat_summary = ""

    while current_word < total_words:
        chunk = word_array[current_word:current_word+chunk_size]
        chunk_text = " ".join(chunk)
        #print(chunk_text)
        summary = process_chunk(chunk_text,specific_instruction)
        #print(summary)
        cat_summary += summary + " "
        current_word += chunk_size

    # Simulate summarizing the concatenated summaries of all chunks
    print(cat_summary)

def main():
    parser = argparse.ArgumentParser(description='Go through files and reformat according to instructions.',
                                     epilog='Example usage: rewrite ./filename-or-directory --specific_instruction "Reformat table as a json format.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_path', type=str, help='Directory or file path to summarize')
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="Format the following text nicely.", help='Add on prompt for more specific instructions')
    args = parser.parse_args()

    input_path = args.input_path
    specific_instruction = args.specific_instruction

    if os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            process_file(file_path, specific_instruction)
    elif os.path.isfile(input_path):
        process_file(input_path, specific_instruction)
    else:
        print("The input path is not a valid directory or file")
        exit(2)

if __name__ == '__main__':
    main()
