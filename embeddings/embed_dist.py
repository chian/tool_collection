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
            "system": "You are a scientific expert and are very good at summarizing current research and knowledge.",
            "prompt":["Summarize the following scientific text." + specific_instruction + chunk_text],
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
