#!/usr/bin/env python3

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

import uuid, re
from rich import print
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic

DEBUG = True

# FUNCTION DEFINITIONS
def call_with_timeout(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=30)  # 10 seconds timeout
        except concurrent.futures.TimeoutError:
            print("The API call timed out.")
            return None

# Retry mechanism with exponential backoff
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def robust_api_call(func, *args, **kwargs):
    result = call_with_timeout(func, *args, **kwargs)
    if result is None:
        raise Exception("API call failed or timed out.")
    return result

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

class AgenticChunker:
    def __init__(self, llm):
        self.chunks = {}
        self.id_truncate_limit = 5
        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = True
        self.llm = llm
        self.llm_call_count = 0

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition):
        if self.print_logging:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)
        #chunk_id = robust_api_call(self._find_relevant_chunk, proposition)
        if DEBUG:
            print("chunk_id (add_proposition):", chunk_id)

        # If a chunk was found then add the proposition to it
        if chunk_id is not None:
            if self.print_logging:
                print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            #robust_api_call(self.add_proposition_to_chunk, chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print ("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition)
        

    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = robust_api_call(self._update_chunk_summary, self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = robust_api_call(self._update_chunk_title, self.chunks[chunk_id])
            #self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            #self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = robust_api_call(runnable.invoke, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        })
        self.llm_call_count += 1
        """
        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        }).content
        """
        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = robust_api_call(runnable.invoke, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        })
        self.llm_call_count += 1
        """
        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        }).content
        """
        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = robust_api_call(runnable.invoke, {
            "proposition": proposition
        })
        self.llm_call_count += 1
        """
        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        })
        """
        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm
        
        new_chunk_title = robust_api_call(runnable.invoke, {
            "summary": summary
        })
        self.llm_call_count += 1
        """
        new_chunk_title = runnable.invoke({
            "summary": summary
        })
        """
        if DEBUG:
            print("new_chunk_title:",new_chunk_title)
        return new_chunk_title


    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        #new_chunk_id = "aAa" + new_chunk_id + "aAa" #add delimiters to make it easier to find
        if DEBUG:
            print("new_chunk_id (_create_new_chunk)",new_chunk_id)
        new_chunk_summary = robust_api_call(self._get_new_chunk_summary, proposition)
        new_chunk_title = robust_api_call(self._get_new_chunk_title, new_chunk_summary)
        #new_chunk_summary = self._get_new_chunk_summary(proposition)
        #new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk of their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = robust_api_call(runnable.invoke, {
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        })
        self.llm_call_count += 1
        """
        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        })
        """
        if DEBUG:
            print("chunk_found (_find_relevant_chunks):",chunk_found)

        if "no chunk" in chunk_found.lower():
            if DEBUG:
                print("_find_relevant_chunks returning None")
            return None


        # Use an LLM call to extract the chunk ID                                                                                                                  
        extract_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant that extracts chunk IDs from text.
                    The chunk ID is a 5-character alphanumeric string enclosed in parentheses.
                    If the text contains a chunk ID, return the chunk ID in a JSON format.
                    If the text does not contain a chunk ID, return None.
                    """
                ),
                ("user", "Extract the chunk ID from the following text:\n{text}"),
            ]
        )

        attempts = 0
        while attempts < 3:
            try:
                extract_runnable = extract_prompt | self.llm
                chunk_id = extract_runnable.invoke({
                    "text": chunk_found
                })
                self.llm_call_count += 1
                if DEBUG:
                    print("chunk_id (_find_relevant_chunk):", chunk_id)
                if chunk_id.lower() == "none":
                    return None
                parsed_json = json.loads(chunk_id)
                chunk_id = parsed_json.get("chunk_id",None)
                if chunk_id not in self.chunks:
                    # Prompt the agent to provide the correct chunk ID or create a new chunk
                    correct_chunk_id_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
                                The previously provided chunk ID does not match any existing chunks.
                                Please provide the correct chunk ID for the given proposition.
                                If you think a new chunk should be created for this proposition, respond with "Create New Chunk".
                                """
                            ),
                            ("user", "Proposition: {proposition}\nChunk ID: {chunk_id}"),
                        ]
                    )
                    
                    correct_chunk_id_runnable = correct_chunk_id_prompt | self.llm
                    correct_chunk_id = correct_chunk_id_runnable.invoke({
                        "proposition": proposition,
                        "chunk_id": chunk_id
                    })

                    if correct_chunk_id.lower() == "create new chunk":
                        return None
                    else:
                        chunk_id = correct_chunk_id.strip()
                return chunk_id
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}. Attempting again...")
                attempts += 1


        #if chunk_id.lower() == "none":
        #    return None
        
        #chunk_id_match = re.search(r"'aAa'(\w{5})'aAa'", chunk_found)

        #if chunk_id_match:
        #    chunk_id = chunk_id_match.group(1)
            #chunk_id = 'aAa' + chunk_id + 'aAa'
        #    return chunk_id
        
        return None
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")

        # Group propositions by chunk ID
        grouped_propositions = {}
        for chunk_id, chunk in self.chunks.items():
            for prop in chunk['propositions']:
                if chunk_id not in grouped_propositions:
                    grouped_propositions[chunk_id] = []
                grouped_propositions[chunk_id].append(prop)

        # Print propositions grouped by chunk ID
        for chunk_id, propositions in grouped_propositions.items():
            chunk = self.chunks[chunk_id]
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in propositions:
                print(f"    - {prop}")
            print("\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self.get_chunk_outline())

# MAIN CODE BLOCK
def main():

    parser = argparse.ArgumentParser(description='...',
                                     epilog='Example usage: python_debug ...',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    parser.add_argument('--filepath', type=str, nargs='?', default="", help='file to be run')
    args = parser.parse_args()

    user_query = args.specific_instruction

    argo_embedding_wrapper_instance = ArgoEmbeddingWrapper()	
    argo_embedding = ARGO_EMBEDDING(argo_embedding_wrapper_instance)
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 0.0)

    sentences = split_file_into_sentences(args.filepath,argo_embedding)
    print(sentences)

    ac = AgenticChunker(llm)

    ac.add_propositions(sentences)
    ac.pretty_print_chunks()
    #ac.pretty_print_chunk_outline()
    #print (ac.get_chunks(get_type='list_of_strings'))
    print(ac.llm_call_count)

    
    
    exit(0)

    
    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[script_executor],
        tasks=[task_execute_code],
        process=Process.sequential
    )

    # Example code for kicking off the crew process - this is conceptual
    # You would need to provide the actual file path and handle the execution context safely
    command = ["python3", args.filepath]
    result = crew.kickoff()
    
    # Process the result
    print(result)

    exit(0)

if __name__ == '__main__':
    main()

