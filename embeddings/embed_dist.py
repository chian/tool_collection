#!/usr/bin/env python3
import argparse
import os
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Example usage:
#summarize_paper ./filename-or-directory --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techiques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?" 

def process_chunk(string1, string2):
    """Send a request to summarize the chunk using an API."""
    headers = {'Content-Type': 'application/json'}

    data = {"user":"chia",
            "prompt":[string1,string2],
            }
    #print(data)
    payload = json.dumps(data)
    
    # Example API call. Replace 'https://example.com/api/summarize' with your actual API endpoint.
    try:
        response = requests.post('https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/',
                                 headers=headers, data=payload)
        #print("Status Code:", response.status_code)
        if response.status_code == 200:
            #print("JSON Response:",response.json())
            embedding = response.json()
            embedding = embedding['embedding']
            #print(embedding)
            summary = "Worked!"
            # Assuming the API returns a JSON with a field 'summary'. Adjust as per your API response structure.
        else:
            summary = f"Failed to get summary, status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        summary = f"API request failed: {e}"

    return embedding, summary


def main():
    parser = argparse.ArgumentParser(description='Embed two strings and calculate the cosine similarity.',
                                     epilog='Example usage: embed_dist "string1" "string2"',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('string1', type=str, help='First string')
    parser.add_argument('string2', type=str, help='Second string')
    args = parser.parse_args()

    embedding, summary = process_chunk(args.string1,args.string2)

    if summary=="Worked!":

        # Example embeddings
        embedding1 = np.array(embedding[0])  # Replace these with your actual embeddings
        embedding2 = np.array(embedding[1])  # Replace these with your actual embeddings
        
        # Reshape the embeddings to 2D arrays
        embedding1_reshaped = embedding1.reshape(1, -1)
        embedding2_reshaped = embedding2.reshape(1, -1)

        euclidean_distance = np.linalg.norm(embedding1 - embedding2)
        print(f"Euclidean distance: {euclidean_distance}")
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1_reshaped, embedding2_reshaped)
        
        print(f"Cosine similarity: {similarity[0][0]}")

    else:
        print(summary)

if __name__ == '__main__':
    main()
    
