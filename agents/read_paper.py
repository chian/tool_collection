#!/usr/bin/env python3
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

import uuid, re, ast
from rich import print
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.chains import create_extraction_chain_pydantic

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

DEBUG = 1
NUM_LLM_THREADS = 6

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
    while True:
        try:
            result = call_with_timeout(func, *args, **kwargs)
            if result is None:
                raise Exception("API call failed or timed out.")
            return result
        except ConnectionError as e:
            print(f"ConnectionError: {e}")
            print("Service cannot be found. Pausing for 600 seconds before retrying...")
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

    with open(file_path, 'r') as file:
        file_content = file.read()

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

def classify_sentence(sentences, index, topic_list, agents, topic_ider, llm):
    sentence = sentences[index]
    DEBUG = True
    if DEBUG:
        print(sentence)
    #before_context = " ".join(sentences[max(0, index - 3):index])
    #after_context = " ".join(sentences[index + 1:index + 4])
    #context = f"Before: {before_context}\nAfter: {after_context}"
    task1 = Task(
        description=f"Classify the following text into one of the given topics:\n\nText: {sentence}\n\nTopics: {', '.join([topic for _, topic in topic_list])}",
        expected_output="The most appropriate topic for the given chunk of text",
        agents=agents,
    )
    task2 = Task(
        description=f"Based on the output from the previous task, return the appropriate topic_id from this table:\n\n{topic_list}\n\nReturn only the topic ID in json format. Your final output should contain nothing other than a json ouput with a single entry corresponding to the topic_id.",
        expected_output="The most appropriate topic for the given sentence",
        agents=[topic_ider],
    )
    crew = Crew(
        tasks=[task1, task2],
        agents=agents + [topic_ider],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True
    )

    max_attempts = 3
    attempt = 0
    topic_id = None

    while attempt < max_attempts and topic_id is None:
        try:
            result = crew.kickoff()
            print(result)

            try:
                # Try to parse the result as JSON
                result_json = json.loads(result.strip())
                topic_id = result_json["topic_id"]
            except json.JSONDecodeError:
                # If parsing as JSON fails, assume the result is a plain string
                topic_id = result.strip()
            except KeyError:
                # If the "topic_id" key is not found in the JSON, set topic_id to None
                topic_id = None
            
            # Check if the topic_id matches one of the existing topic IDs
            if topic_id not in [t_id for t_id, _ in topic_list]:
                print(f"Warning: Invalid topic ID '{topic_id}' for sentence: {sentence}")
                topic_id = None
                attempt += 1
            else:
                break

        except Exception as e:
            print(f"An error occurred during crew execution: {str(e)}")
            attempt += 1

    if topic_id is None:
        print(f"Failed to obtain a valid topic ID for sentence: {sentence}")
        return None

    if DEBUG:
        print(topic_id,index,sentence)

    return {
        "topic_id": topic_id,
        "index": index,
        "sentence": sentence,
    }

def miss_deep(llm):
    agent = Agent(
        role="Critical Analyst of Research Findings",
        goal=textwrap.dedent("""
        Evaluate the depth, validity, and implications of research findings, and identify more
        pertinent questions or alternative interpretations.
        """),
        backstory=textwrap.dedent("""
        Miss Deep is renowned for her ability to delve into the complex world of academic research,
        scrutinizing the methodology, data, and conclusions of papers across various fields. With a
        keen eye for detail and a questioning mind, she goes beyond the surface to explore what
        results truly signify, challenging assumptions and proposing new angles for investigation.
        Her expertise lies not just in understanding the data presented but in discerning the
        broader context and potential biases, making her an invaluable asset in the pursuit of
        knowledge.
        """),
        llm=llm
    ) 
    return agent

def joker(llm):
    agent = Agent(
        role = "Master of Clichéd Erudition",
        goal = textwrap.dedent("""
        Restate and reinterpret statements with a blend of cliché and scholarly eloquence, elevating
        common discourse through the art of sophisticated verbosity.
        """),
        backstory = textwrap.dedent("""
        Joker, not confined to mere humor, has mastered the art of dressing words in the finest of
        academic robes, even when the occasion hardly calls for it. With a library's worth of phrases
        at his disposal, he delights in taking the mundane and spinning it into a tapestry of elaborate
        diction and well-trodden sayings. His expertise is not just in the volume of his vocabulary but
        in the skill with which he weaves it into conversation, ensuring that even the simplest of
        statements are transformed into a grandiose exposition.
        """),
        llm=llm,
    )
    return agent

def wendy(llm):
    agent = Agent(
        role = "Supreme Database Organizer",
        goal = textwrap.dedent("""
        Excel in the meticulous organization of information, categorization of diverse entities, and the
        strategic structuring of databases to enhance accessibility and efficiency.
        """),
        backstory = textwrap.dedent("""
        With an innate passion for order and structure, this super administrative assistant has honed
        their skills in the art of organization to perfection. They possess an uncanny ability to sift
        through chaos, identify underlying patterns, and systematically arrange data into intuitive
        categories. Their expertise extends beyond mere filing; they understand the nuances of data
        interrelationships and excel at crafting databases that are not only organized but are also
        optimized for user engagement and query efficiency. Their meticulous attention to detail
        ensures that no piece of information is ever out of place, making them an indispensable asset
        in any data-driven environment.
        """),
        llm = llm,
    )
    return agent

def agent_the_paper(sentences,llm):
    topic_list = [
        ("t0001", "Background Information and Previous State of the Art"),
        ("t0002", "Detailed Problem Statement or Barrier Overcome"),
        ("t0003", "Methodology"),
        ("t0004", "Experiments or Tests Performed and Results"),
        ("t0005", "References"),
        ("t0006", "Publication Information such as Title, Authors, and Affiliation"),
        ("t0007", "Other")
    ]
    """
    agents = [
        Agent(
            role=f"Advocate for {topic}",
            goal=f"Argue why a sentence should be classified as {topic}",
            backstory=f"You are an expert in identifying sentences related to {topic}.",
            llm=llm
        )
        for _, topic in topic_list
    ]
    """
    agents = [miss_deep(llm), joker(llm), wendy(llm)]
    topic_ider = Agent(
        role=f"Topic ID Reporter",
        goal=f"Return a JSON formatted string with only the topic_id field. Your final answer should be only a JSON-formatted string with a topic ID.",
        backstory="You are an expert in extracting topic IDs from text and returning them in JSON format.",
        llm=llm
    )
    
    classified_sentences = {}
    for topic_id, topic in topic_list:
        classified_sentences[topic_id] = {
            "topic": topic,
            "sentences": [],
        }

    global_usage_metrics = {}
    print("About to work on these:",sentences)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_LLM_THREADS) as executor:
        # Submit the classification tasks to the executor
        futures = [executor.submit(classify_sentence, sentences, index, topic_list, agents, topic_ider, llm)
                   for index in range(len(sentences))]
        # Retrieve the results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                topic_id = result["topic_id"]
                classified_sentences[topic_id]["sentences"].append({
                    "index": result["index"],
                    "sentence": result["sentence"],
                })
                
    if DEBUG:
        print(classified_sentences)
    return classified_sentences

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
        specific_instructions_original = """
        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".
        
        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.
        """
        specific_instructions = """
        Your summaries should focus on the uses of large language models (LLMs) in cancer research and take a fine-grained
        approach to categorizing use cases. For everythign that does not fit into that category, you should generalize
        very broadly. You must mention in the category summary whether LLMs are used or not used in a paper.
        
        Example:
        Input: Proposition: LLMs can be used to diagnosis instestinal bowel disease.
        Output: This chunk contains information about using LLMs in disease that are not cancer.

        Input: Proposition: LLMs can be used to embed the electronic health record and make predictions about outcomes
        in patients with breast cancer. First models were embedded using ClinBERT and the those embeddings were passed
        to a linear regression model to predict survival under or over 5 years.
        Output: This chunk contains information about the use of LLMs as embedding models for the electronic health
        record and predicting patient outcomes in breast cancer. The specific models used where ClinBERT and a linear
        regression.

        Input: Proposition: AI vision model can be used to predict extent of disease in breast cancer patients.
        Output: This chunk does not contain information about the use of LLMs in cancer.
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    {specific_instructions}
                    
                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = robust_api_call(runnable.invoke, {
            "proposition": proposition,
            "specific_instructions" : specific_instructions
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

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='...',
                                     epilog='Example usage: python_debug ...',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    parser.add_argument('--directory', type=str, nargs='?', default="", help='directory containing files to be processed')
    args = parser.parse_args()

    user_query = args.specific_instruction

    argo_embedding_wrapper_instance = ArgoEmbeddingWrapper()	
    argo_embedding = ARGO_EMBEDDING(argo_embedding_wrapper_instance)
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 0.5)

    ac = AgenticChunker(llm)

    for filename in os.listdir(args.directory):
        filepath = os.path.join(args.directory, filename)
        if os.path.isfile(filepath):
            paragraphs = split_file_into_paragraphs(filepath, llm)
            print("Main: paragraphs = ",paragraphs)
            classified_sentences = agent_the_paper(paragraphs,llm)
            sorted_sentences = merge_sentences(classified_sentences)
            print(f"Classified Texts for {filename}:", sorted_sentences)

            PROMPT = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are a eloquent discusser of topics. You have a PhD in science and conveny information in
                        expert and educated verbiage that is precise and efficient. You are not prone to unnecessary
                        flourishes of language as those are for the weak-minded. Truth is the balance between precision
                        and conciseness.
                        """,
                    ),
                    ("user",
                     """You will take the below chunk and summarize it without losing any key information. You will
                     highlight hypotheses discussed, assumptions made, points raised, experiments performed, results obtained,
                     and anything else relevant to understanding the chunk of text in its entirety.
                     --Start of current chunk-- {chunk}  --End of current chunk--
                     """,
                     ),
                ]
            )

            for topic_id, topic_info in sorted_sentences.items():
                topic = topic_info['topic']
                text = topic_info['text']
                if len(text) > 10:
                    runnable = PROMPT | llm
                    section_summary = robust_api_call(runnable.invoke, {
                        "chunk": text,
                    })
                    sorted_sentences[topic_id]['summary'] = section_summary

            print(f"Summaries added for {filename}:", sorted_sentences)

            paper_summary = ""
            for topic_id, topic_info in sorted_sentences.items():
                if 'summary' in topic_info:
                    section_summary = topic_info['summary']
                    paper_summary += section_summary + "\n"

            print(f"Paper Summary for {filename}:", paper_summary)
            ac.add_propositions([paper_summary])

    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()

    exit(0)

if __name__ == '__main__':
    main()

