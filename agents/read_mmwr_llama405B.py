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

class InfectiousDiseaseRelevance(BaseModel):
    relevance: str = Field(description="Indicates if the chunk is related to Infectious Disease Outbreaks ('Yes' or 'No')")

    @validator('relevance')
    def relevance_validator(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('relevance must be either "Yes" or "No"')
        return v
    
def param_infectious_disease_chunks(ac, llm):
    LOCAL_DEBUG = False
    
    print("\nInfectious Disease Outbreak Chunks:")

    #grammar = LlamaGrammar.from_json_schema(InfectiousDiseaseRelevance.schema_json())
    
    PROMPT_RELEVANCE = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that analyzes chunk summaries to determine if they are related to Infectious Disease Outbreaks.

                Consider the following criteria when determining if a chunk summary is related to Infectious Disease Outbreaks:
                - Mentions of specific infectious diseases or outbreaks
                - Descriptions of the spread, transmission, or impact of infectious diseases
                - Discussions about prevention, control, or response measures for infectious disease outbreaks
                - References to public health emergencies or pandemics caused by infectious diseases

                Exclude the following types of information:
                - Vaccine side effects or reactions that are not actual outbreaks
                - Discussions about eradication efforts or progress reports without mentioning specific outbreaks
                - Vague or incomplete information about surveillance gaps without mentioning specific diseases or outbreaks
                
                If the chunk summary is related to Infectious Disease Outbreaks based on the above criteria, respond with "Yes".
                If the chunk summary is not related to Infectious Disease Outbreaks, respond with "No".

                Chunk Summary:
                {chunk_summary}
                """
            )
        ]
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=InfectiousDiseaseRelevance)
    format_instructions = pydantic_parser.get_format_instructions()

    PROMPT_THREAT_LEVEL = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that answers questions about Infectious Disease Outbreaks based on 
                given information.

                Summary:
                {chunk_text}

                Original Information:
                {propositions}

                Questions:
                1. What disease is being discussed?
                2. What is the causal species of bacteria or virus of the disease?
                3. What geographical location is it occuring in?
                """
            )
        ]
    )

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
                You are an AI assistant that analyzes documents for text related to ongoing infectious diseases that could 
                potentially spread and returns only that text.

                Consider the following criteria when determining if a part of the text is related to an infectious disease outbreak:
                - Mentions of specific infectious diseases or outbreaks
                - Descriptions of the spread, transmission, or impact of infectious diseases
                - Discussions about prevention, control, or response measures for infectious disease outbreaks
                - References to public health emergencies or pandemics caused by infectious diseases

                Exclude the following types of information:
                - Vaccine side effects or reactions that are not actual outbreaks
                - Discussions about eradication efforts or progress reports without mentioning specific outbreaks
                - Vague or incomplete information about surveillance gaps without mentioning specific diseases or outbreaks
                
                Note that there might be multiple disease outbreaks being discussed within one document. Your job is to filter out
                all the text that is not related to an infectious disease outbreak.

                Text:
                {content}

                In a very short and single phrase, list any disease outbreaks that are a threat for becoming a                                           
                concerning outbreak. Do not include anything that is likely to remain minor. If there is a disease                                       
                that may possibly become a larger disease outbreak, please begin your response with                                                      
                "POTENTIAL RISKS:" followed by the names of the infectious agents. If there are no significant
                risks, begin your response with "FINAL ANSWER:". Note that "POTENTIAL RISKS:" and "FINAL ANSWER:                                        
                are mutually exclusive and should not occur together. Please note that articles about vaccination rates
                DO NOT constitute an ongoing outbreak.
                
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



