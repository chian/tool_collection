#!/usr/bin/env python3
import argparse
import os
import json
import requests
from langchain.text_splitter import PythonCodeTextSplitter

#Example usage:
#summarize_paper ./filename-or-directory --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techiques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?" 

def process_chunk(chunk_text, specific_instruction):
    """Send a request to summarize the chunk using an API."""
    headers = {'Content-Type': 'application/json'}

    data = {"user":"chia",
            "model":"gpt4", 
            "system": "You are a python coding expert capable of answering any questions about a specific code.",
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

    with open(file_path, 'r') as file:
        file_content = file.read()

    python_splitter = PythonCodeTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = python_splitter.create_documents([file_content])

    cat_answer = ""

    if len(chunks) == 1:
        chunk_text = chunk.page_content
        answer = process_chunk(chunk_text,specific_instruction)
        full_answer = answer
    else:
        for chunk in chunks:
            chunk_text = chunk.page_content
            answer = process_chunk(chunk_text,specific_instruction)
            cat_answer += answer + "\n"
        full_answer = process_chunk(cat_answer,"put all of these findings into a single response - do not lose any information, just organize it better")
    print(f"Final Answer for {file_path}:")
    print(full_answer)
    print("-------------")

def main():
    parser = argparse.ArgumentParser(description='Summarize text from a file or directory of files.',
                                     epilog='Example usage: summarize_paper ./filename-or-directory --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techniques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?"',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_path', type=str, help='Directory or file path to summarize')
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
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
