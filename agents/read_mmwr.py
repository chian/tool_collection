#!/usr/bin/env python
import pdb
import argparse
import os, json
import numpy as np
import time
import concurrent.futures
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
#from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import textwrap
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import subprocess

import re, ast
from rich import print
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.chains import create_extraction_chain_pydantic

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from langchain_community.llms import LlamaCpp
from llama_cpp.llama_grammar import LlamaGrammar
from functools import wraps

from AgenticChunker import AgenticChunker

from ollama import Client
from langchain_community.llms import Ollama

DEBUG = 1
ID_LIMIT = 5
NUM_LLM_THREADS = 4

# FUNCTION DEFINITIONS
def call_with_timeout(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=120)  # 30 seconds timeout
        except concurrent.futures.TimeoutError:
            print("The API call timed out.")
            raise Exception("API call timed out.")

def retry_on_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                result = func(*args, **kwargs)
                if result is None:
                    print("No result returned. Retrying the API call in 10 minutes...")
                    # Save checkpoint before sleeping
                #    checkpoint_data = {
                #        "agenticChunker": ac.to_json(),
                #        "currentFileIndex": current_file_index,
                #        "currentParagraphIndex": current_paragraph_index,
                #        # Include other relevant data in the checkpoint
                #    }
                #    save_checkpoint(checkpoint_data, "checkpoint.json")
                    time.sleep(360)  # Sleep for 1 hour before retrying
                else:
                    return result
            except Exception as e:
                print(f"An error occurred: {e}. Retrying the API call in 10 minutes...")
                # Save checkpoint before sleeping
                #checkpoint_data = {
                #    "agenticChunker": ac.to_json(),
                #    "currentFileIndex": current_file_index,
                #    "currentParagraphIndex": current_paragraph_index,
                    # Include other relevant data in the checkpoint
                #}
                #save_checkpoint(checkpoint_data, "checkpoint.json")
                time.sleep(360)  # Sleep for 1 hour before retrying
    return wrapper

@retry_on_error
def robust_api_call(func, *args, **kwargs):
    while True:
        try:
            result = call_with_timeout(func, *args, **kwargs)
            if result is None:
                raise Exception("API call failed or timed out.")
            return result
        except ConnectionError as e:
            print(f"ConnectionError: {e}")
            print("Service cannot be found. Pausing for 600 seconds before retrying...")
            # Save checkpoint before sleeping
            checkpoint_data = {
                "agenticChunker": ac.to_json(),
                "currentFileIndex": current_file_index,
                "currentParagraphIndex": current_paragraph_index,
                # Include other relevant data in the checkpoint
            }
            save_checkpoint(checkpoint_data, "checkpoint.json")
            time.sleep(600)

def split_file_into_sentences(file_path,embeddings):
    #Process a single file and perform semantic segmentation.
    print(f"Processing {file_path}...")

    with open(file_path, 'r') as file:
        file_content = file.read()

    # Create a RecursiveCharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=['\n\n', '\n', '. ', '? ', '! ']
    )

    # Split the file content into sentences
    documents = text_splitter.create_documents([file_content])
    sentences = [doc.page_content for doc in documents]
    
    return sentences

def split_file_into_chunks(file_path):
    #Process a single file and perform semantic segmentation.
    print(f"Processing {file_path}...")
    file_content = read_and_sanitize_file(file_path)

    #with open(file_path, 'r') as file:
    #    file_content = file.read()

    #print(file_content)
    # Create a CharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=[" ", "\n", "\t"]
    )
    
    # Split the file content into chunks
    chunks = text_splitter.split_text(file_content)
    #print(chunks)
    return chunks

class Paragraph(BaseModel):
    paragraph: str = Field(description="a paragraph of text from a scientific paper")

class Paragraphs(BaseModel):
    paragraphs: List[Paragraph]

def split_chunk_into_paragraphs(chunk,llm):
    LOCAL_DEBUG = False
    if LOCAL_DEBUG:
        print("Incoming chunk:",chunk)

    template_string = """
    You are an expert and parsing which words belong to a sentence and which sentence belong
    together in a paragraph. You will split up words and sentences sensibly to make coherent
    paragraphs that discuss a single idea or subject.  Ignore partial paragraphs at the start
    and end of the chunk.

    Please sanatize the output for JSON by getting rid of any URLs, fixing typos,
    and removing or adding spaces and puncutation as necessary.
    
    Use the formatting instructions below to provide the answers to user queries.
    
    FORMATTING_INSTRUCTIONS:
    {format_instructions}
    
    Here is the chunk of text to process:
    {chunk}
    """
        
    pydantic_parser = PydanticOutputParser(pydantic_object=Paragraphs)
    format_instructions = pydantic_parser.get_format_instructions()
    #print(format_instructions)

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(chunk=chunk, format_instructions=format_instructions)
    #print(messages)
    

    max_attempts = 3
    attempt = 0
    parsed_output = None

    while attempt < max_attempts and parsed_output is None:
        try:
            output = robust_api_call(llm.invoke, messages)

            if LOCAL_DEBUG:
                print ("OUTCONTENT",output)

            parsed_output = pydantic_parser.parse(output)

            if LOCAL_DEBUG:
                print("TYPE",type(parsed_output))
                print("PARAGRAPHS\n",parsed_output.paragraphs)
                print("-"*80)
                print("PARAGRAPHS")
                for paragraph in parsed_output.paragraphs:
                    text = paragraph.paragraph
                    print(text)
                    print("-"*50)
                    print("Paragraph splitter says:",parsed_output)
            #print(parsed_output)
            #pdb.set_trace()
        except OutputParserException as e:
            print(f"Error parsing output: {e}")
            attempt += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            attempt += 1

    if parsed_output is None:
        print(f"Failed to parse output after {max_attempts} attempts.")
        return []

    return parsed_output.paragraphs

def process_chunk(chunk,llm):
    #print("processing chunk...", chunk)
    paragraph_json = split_chunk_into_paragraphs(chunk, llm)
    #print("Print paragraph json",paragraph_json)
    return paragraph_json
                    
def split_file_into_paragraphs(file_path, llm):
    if DEBUG:
        print("Splitting file into chunks...")
    chunks = split_file_into_chunks(file_path)
    paragraphs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_LLM_THREADS) as executor:
        futures = [executor.submit(process_chunk, chunk, llm) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            #print("1. result = ", result)
            if result:
                for paragraph in result:
                    paragraphs.append(paragraph.paragraph)
                #print("2. result = ",result)
    #print("split file:",paragraphs)
    return paragraphs

class ChunkIDResponse(BaseModel):
    chunk_id: str = Field(description="The chunk ID extracted from the text")

    @validator('chunk_id')
    def chunk_id_validator(cls, v):
        if len(v) != ID_LIMIT or not v.isalnum():
            raise ValueError(f"chunk_id must be a {ID-LIMIT}-character alphanumeric string")
        return v

class InfectiousDiseaseRelevance(BaseModel):
    relevance: str = Field(description="Indicates if the chunk is related to Infectious Disease Outbreaks ('Yes' or 'No')")

    @validator('relevance')
    def relevance_validator(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('relevance must be either "Yes" or "No"')
        return v
    
def merge_sentences(document_content):
    merged_text = {}

    for topic_id, topic_info in document_content.items():
        topic = topic_info['topic']
        sentences = topic_info['sentences']

        # Sort the sentences based on the index
        sorted_sentences = sorted(sentences, key=lambda x: x['index'])

        # Extract the sorted sentences
        merged_sentences = ' '.join([sentence['sentence'] for sentence in sorted_sentences])

        merged_text[topic_id] = {
            'topic': topic,
            'text': merged_sentences
        }

    return merged_text



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

    for chunk_id, chunk in ac.chunks.items():
        chunk_summary = chunk['summary']

        runnable = PROMPT_RELEVANCE | llm
        #llm.grammar = grammar
        
        relevance_response = runnable.invoke({
            "chunk_summary": chunk_summary,
            "format_instruction": format_instructions,
        })
        if LOCAL_DEBUG:
            print("chunk_summary:", chunk_summary)
            print("relevance_response:", relevance_response)

        if "yes" in relevance_response.lower():
            is_related = "yes"
        else:
            is_related = "no"

        if LOCAL_DEBUG:
            print("is_related:", is_related)
        
        if is_related == "yes":
            chunk_text = "\n".join(chunk['propositions'])

            runnable_threat_level = PROMPT_THREAT_LEVEL | llm
            threat_assessment = robust_api_call(runnable_threat_level.invoke, {
                "chunk_text": chunk_text,
                "propositions": chunk['propositions']
            })
            #self.llm_call_count += 1
            
            print(f"Chunk ID: {chunk_id}")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk_summary}")
            print(f"File Path: {chunk['file_path']}")
            print(f"Threat Level Assessment: {threat_assessment}")
            print("Propositions:")
            for proposition in chunk['propositions']:
                print(f"- {proposition}")
            print()

def save_checkpoint(checkpoint_data, checkpoint_file):
    try:
        with open(checkpoint_file, "w") as file:
            json.dump(checkpoint_data, file, indent=4)
        print(f"Checkpoint saved to {checkpoint_file}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def sanitize_for_json(input_string):
    # Replace or escape specific characters that are problematic for JSON
    # This is a basic example; for more complex scenarios, consider using a library
    sanitized_string = input_string.replace('\\', '\\\\').replace('"', '\"').replace('\n', '\\n').replace('\r', 'r').replace('\g','\\g').replace('\)',')').replace('\(','(')
    return sanitized_string

def read_and_sanitize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return sanitize_for_json(content)

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='...',
                                     epilog='Example usage: python read_mmwr.py --input "path/to/files" --paragraph',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='Input file, directory, or wildcard expression for files to be processed.')
    parser.add_argument('--paragraph', action='store_true', help='Enable paragraph processing')
    args = parser.parse_args()

    #parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    #user_query = args.specific_instruction

    specific_instructions = """
        Your summaries should focus on infectious disease outbreaks and assess the potential threat level these
        infectious diseases pose. Take into consideration the deadliness or severity of the disease, the infectivity,
        and mention any potentially vulnerable populations such as the elderly or children and infants. Take a
        fine-grained approach to categorizing disease cases based on threat level, potential for harm, infectious
        agent, and geographical location. For everything that does not fit into the category of infectious disease,
        you should generalize very broadly. You must mention in the category summary the infectious disease, causal
        organism or virus, threat level, and at-risk populations.
        
        Example:
        Input: Proposition: A new strain of influenza virus has been detected in several countries, causing severe
        respiratory illness particularly in older adults and people with underlying health conditions. The virus
        appears to be highly transmissible.

        Output: This chunk contains information about a potentially high threat level outbreak of a new influenza
        virus strain. The virus causes severe illness, especially in vulnerable populations like the elderly and
        those with pre-existing conditions. High transmissibility is noted. Close monitoring and containment
        measures are warranted.

        Input: Proposition: Health authorities are investigating a cluster of Legionnaires' disease cases traced
        back to a cooling tower in an industrial area. Most patients are responding well to antibiotic treatment.

        Output: This chunk discusses a localized outbreak of Legionnaires' disease linked to an environmental
        source. The threat level appears moderate as patients are recovering with proper treatment. Working-age
        adults in or around the affected industrial area are the main population at risk. Remediation of the
        cooling tower should help control the outbreak.

        Input: Proposition: Researchers have published a new machine learning model that can accurately predict
        crop yields based on satellite imagery and weather data.

        Output: This chunk does not contain information about an infectious disease outbreak. It broadly relates
        to applications of machine learning in agriculture.
        """

    # Determine if the input is a directory, a single file, or a wildcard expression
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    elif "*" in args.input:  # Handle wildcard input
        files = glob.glob(args.input)
    else:  # Handle single file input
        files = [args.input]

    # Initialize the ollama client
    llm = Ollama(model="llama3")

    #Load save checkpoint for agentic chunking
    checkpoint_file = "checkpoint.json"
    global ac
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as file:
            checkpoint_data = json.load(file)
            ac = AgenticChunker.from_json(checkpoint_data["agenticChunker"],llm=llm,
                                          chunk_summary_instruction=specific_instructions)
            current_file_index = checkpoint_data["currentFileIndex"]
            current_paragraph_index = checkpoint_data.get("currentParagraphIndex", 0)
    else:
        ac = AgenticChunker(llm, chunk_summary_instruction=specific_instructions)
        current_file_index = 0
        current_paragraph_index = 0

    #This is agentic chunking with paragraphs
    if args.paragraph == True:
        for i in range(current_file_index, len(files)):
            filepath = files[i]
            if os.path.isfile(filepath):
                paragraphs = split_file_into_paragraphs(filepath, llm)
                #print("Main: paragraphs = ",paragraphs)
                for j, paragraph in enumerate(paragraphs[current_paragraph_index:], start=current_paragraph_index):
                    ac.add_proposition(paragraph, filepath)
                    current_paragraph_index = j + 1

                    # Save checkpoint after processing each paragraph
                    if j % 10 == 0:
                        checkpoint_data = {
                            "agenticChunker": ac.to_json(),
                            "currentFileIndex": i,
                            "currentParagraphIndex": current_paragraph_index,
                            # Include other relevant data in the checkpoint
                        }
                        save_checkpoint(checkpoint_data, checkpoint_file)

                # Reset paragraph index for the next file
                current_paragraph_index = 0

                # Update current file index
                current_file_index = i + 1

                # Save checkpoint after finishing each file
                checkpoint_data = {
                    "agenticChunker": ac.to_json(),
                    "currentFileIndex": current_file_index,
                    "currentParagraphIndex": current_paragraph_index,
                    # Include other relevant data in the checkpoint
                }
                save_checkpoint(checkpoint_data, checkpoint_file)
                
    param_infectious_disease_chunks(ac, llm)
    print("Number llm calls by AgenticChunker:",ac.llm_call_count)
    
    exit(0)

if __name__ == '__main__':
    main()

