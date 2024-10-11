#!/usr/bin/env pythonA
import pdb
import argparse
import os, json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING

import re, ast
from rich import print
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.chains import create_extraction_chain_pydantic

from typing import List
from langchain.output_parsers import PydanticOutputParser
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
import openai

from transformers import AutoTokenizer

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='Determine any infectious disease outbreaks in a document.')
    parser.add_argument('--input', type=str, required=True, help='Input file path.')
    args = parser.parse_args()

    #openai_api_key = "your-api-key"
    #openai_api_base = f"http://your-api-server:{args.port}/v1"

    # Set OpenAI's API key and API base to use vLLM's API server.
    #openai_api_key = "EMPTY"

    openai_api_key = "cmsc-35360"
    openai_api_base = f"http://66.55.67.65:80/v1"

    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    def chunk_content(content, words_per_chunk=13000):
        words = content.split()
        return [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

    with open(args.input, 'r', encoding='utf-8') as file:
        content = file.read()
    chunks = chunk_content(content)

    PROMPT_FIND_OUTBREAK_TEXT = PromptTemplate.from_template(
                """
                Identify reports with potential mention of a variant of concern.
                Any report with a match will be considered a positive match.

                Avian influenza virus (highly pathogenic)
                Bacillus anthracis
                Botulinum neurotoxin (For the purposes of this policy, there are no exempt quantities of botulinum neurotoxin. Research involving any quantity of botulinum neurotoxin should be evaluated for DURC potential.)
                Burkholderia mallei
                Burkholderia pseudomallei
                Ebola virus
                Foot-and-mouth disease virus
                Francisella tularensis
                Marburg virus
                Reconstructed 1918 Influenza virus
                Rinderpest virus
                Toxin-producing strains of Clostridium botulinum
                Variola major virus
                Variola minor virus
                Yersinia pestis

                Text from Report:
                {content}

                In a very short and single phrase, list your answer.                                           
                If there is a disease from the above list mentioned, please begin your response with
                "POTENTIAL RISKS:" followed by the names of the infectious agents. If there are no significant
                risks, respond with "FINAL ANSWER: None". Note that "POTENTIAL RISKS:" and "FINAL ANSWER:                                        
                are mutually exclusive and should not occur together.
                
                """
    )

    responses = []
    for content in chunks:
        outbreak_analysis_prompt = PROMPT_FIND_OUTBREAK_TEXT.format(content=content)
        #print(outbreak_analysis_prompt)
        # Using LLM-3 tokenizer to count tokens more accurately for the model
        from token_count import TokenCount
        tc = TokenCount(model_name="gpt-3.5-turbo")
        tokens = tc.num_tokens_from_string(outbreak_analysis_prompt)
        print(f"Total tokens in the prompt according to GPT-3 tokenizer: {tokens}")

        model='llama31-405b-fp8'
        outbreak_analysis_response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": outbreak_analysis_prompt}
            ],
        )
        #import pdb
        #pdb.set_trace()
        responses.append(outbreak_analysis_response.choices[0].message.content)
        #print(outbreak_analysis_response.choices[0].message.content)

    print("*************"+args.input+"*************")
    print("\n".join(responses))
    exit(0)

    list_responses = client.chat.completions.create(
        model="gradientai/Llama-3-70B-Instruct-Gradient-262k",  # or your specific model
        messages=[
            {"role": "user", "content": "Convert the following into a python list: " + response}
            ],
        temperature=0.0,
        max_tokens=100000
    )
    list_r = list_responses.choices[0].message.content
    print(list_r)

    for item in list_r:
        print(item)
        item_response = client.chat.completions.create(
            model="gradientai/Llama-3-70B-Instruct-Gradient-262k",  # or your specific model
            messages=[
            {"role": "user", "content": "Read the below text and pull out all information related to:" + item + "Text:" + content1}
            ],
        temperature=0.0,
        max_tokens=100000
        )
        print(item_response.choices[0].message.content)

    exit(0)

    chat_response = client.chat.completions.create(
        model='gradientai/Llama-3-70B-Instruct-Gradient-262k',
        messages=[
            {"role": "user", "content": "Please generate four hypothesis in the origins of life that could be explored with a self-driving laboratory.  For each example please list the key equipment and instruments that would be needed and the experimental protocols that would need to be automated to test the hypotheses."},
        ],
        temperature=0.0,
        max_tokens=2056,
    )

    print(chat_response.choices[0].message.content)

    specific_instructions = ""

    #This is agentic chunking with paragraphs
    for i in range(len(files)):
        filepath = files[i]
        if os.path.isfile(filepath):
            
            finder_response = find_infectious_disease_outbreaks(content,llm)
            print(finder_response)
            
                 
    param_infectious_disease_chunks(ac, llm)
    print("Number llm calls by AgenticChunker:",ac.llm_call_count)
    
    exit(0)

if __name__ == '__main__':
    main()



