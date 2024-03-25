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

from langchain_community.llms import LlamaCpp
from llama_cpp.llama_grammar import LlamaGrammar
from functools import wraps

DEBUG = 1
ID_LIMIT = 5
NUM_LLM_THREADS = 4

# FUNCTION DEFINITIONS
def call_with_timeout(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=10)  # 10 seconds timeout
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
                    print("No result returned. Retrying the API call in 1 hour...")
                    time.sleep(3600)  # Sleep for 1 hour before retrying
                else:
                    return result
            except Exception as e:
                if "status code: 500" in str(e):
                    print("Received status code 500. Retrying the API call in 1 hour...")
                    time.sleep(3600)  # Sleep for 1 hour before retrying
                elif "API call timed out." in str(e):
                    print("API call timed out. Retrying the API call in 1 hour...")
                    time.sleep(3600)  # Sleep for 1 hour before retrying
                else:
                    raise e
    return wrapper

# Retry mechanism with exponential backoff
#@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
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
    #DEBUG = True
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

#AGENTS
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

def generic(llm,topic_list):
    agents = [
        Agent(
            role=f"Advocate for {topic}",
            goal=f"Argue why a sentence should be classified as {topic}",
            backstory=f"You are an expert in identifying sentences related to {topic}.",
            llm=llm
        )
        for _, topic in topic_list
    ]
    return agents

#TOPICS
def the_paper():
    topic_list = [
        ("t0001", "Background Information and Previous State of the Art"),
        ("t0002", "Detailed Problem Statement or Barrier Overcome"),
        ("t0003", "Methodology"),
        ("t0004", "Experiments or Tests Performed and Results"),
        ("t0005", "References"),
        ("t0006", "Publication Information such as Title, Authors, and Affiliation"),
        ("t0007", "Other")
    ]
    return topic_list    

def agent_parser(sentences,llm):
    topic_list = the_paper()
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
    
class AgenticChunker:
    def __init__(self, llm):
        self.chunks = {}
        self.id_truncate_limit = ID_LIMIT
        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = DEBUG
        self.llm = llm
        self.llm_call_count = 0
    
    def to_json(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        ac = cls(data["llm"])
        ac.chunks = data["chunks"]
        ac.id_truncate_limit = data["id_truncate_limit"]
        ac.generate_new_metadata_ind = data["generate_new_metadata_ind"]
        ac.print_logging = data["print_logging"]
        ac.llm_call_count = data["llm_call_count"]
        return ac
        
    def add_propositions(self, propositions, file_path):
        for proposition in propositions:
            self.add_proposition(proposition, file_path)
    
    def add_proposition(self, proposition,file_path):
        if self.print_logging:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition,file_path)
            return

        chunk_id = self._find_relevant_chunk(proposition)
        #chunk_id = robust_api_call(self._find_relevant_chunk, proposition)
        if DEBUG:
            print("chunk_id (add_proposition):", chunk_id)

        # If a chunk was found then add the proposition to it
        if chunk_id is not None:
            if "create" in chunk_id.lower():
                if self.print_logging:
                    print("Creating a new chunk as suggested by the LLM")
                self._create_new_chunk(proposition, file_path)
            else:
                if self.print_logging:
                    print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
                self.add_proposition_to_chunk(chunk_id, proposition)
                #robust_api_call(self.add_proposition_to_chunk, chunk_id, proposition)
        else:
            if self.print_logging:
                print ("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition,file_path)
        

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
        Your summaries should focus on infectious disease outbreaks and assess the potential threat leavel these
        infectous disease pose. Take into consideration the deadliness or seversity of the disease, the infectivity,
        and mention any potentially vulnerable populations such as the elderly or children and infants. Take a
        fine-grained approach to categorizing disease cases based on threat level, potential for harm, infectious
        agent, and geographical location. For everythign that does not fit into the category of infectious disease,
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


    def _create_new_chunk(self, proposition,file_path):
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
            'chunk_index' : len(self.chunks),
            'file_path' : file_path,
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
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\nFile: {chunk['file_path']}\n\n"""
        
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

        existing_chunk_ids = list(self.chunks.keys())
        for existing_id in existing_chunk_ids:
            if existing_id in chunk_found:
                return existing_id

        return None
        '''    
        if "no chunk" in chunk_found.lower():
            if DEBUG:
                print("_find_relevant_chunks returning None")
                return None

        #Define the grammar for the chunk ID response
        grammar = LlamaGrammar.from_json_schema(ChunkIDResponse.schema_json())
        
        # Use an LLM call to extract the chunk ID                                                                                                       
        extract_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant that extracts chunk IDs from text.
                    The chunk ID is a {ID_LIMIT}-character alphanumeric string enclosed in parentheses.
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
                self.llm.grammar = grammar
                chunk_id_response = extract_runnable.invoke({
                    "ID_LIMIT": ID_LIMIT,
                    "text": chunk_found
                })
                self.llm_call_count += 1
                if DEBUG:
                    print("chunk_id (_find_relevant_chunk):", chunk_id_response)
                parsed_json = json.loads(chunk_id_response)
                chunk_id = parsed_json.get("chunk_id", "")

                if chunk_id == "":
                    return None
                
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
                    self.llm.grammar = None
                    correct_chunk_id_response = correct_chunk_id_runnable.invoke({
                        "proposition": proposition,
                        "chunk_id": chunk_id
                    })

                    if correct_chunk_id_response == "":
                        return None
                    else:
                        chunk_id = correct_chunk_id_response.strip()
                return chunk_id
            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}. Attempting again...")
                attempts += 1
            
        return None
        '''
        
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

def print_infectious_disease_chunks(ac, llm):
    LOCAL_DEBUG = False
    
    print("\nInfectious Disease Outbreak Chunks:")

    grammar = LlamaGrammar.from_json_schema(InfectiousDiseaseRelevance.schema_json())
    
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
                You are an AI assistant that assesses the threat level of Infectious Disease Outbreaks based on given information.

                Consider the following factors when determining the threat level:
                - Nature of the infection: How severe are the symptoms? What is the fatality rate?
                - Likelihood of spread: How easily does the disease transmit? Is it highly contagious?
                - Potential impact: How many people are at risk? Are vulnerable populations affected?

                Provide a threat level assessment based on the above factors using the following scale:
                - Low: The outbreak poses minimal risk to public health.
                - Moderate: The outbreak poses a significant risk and requires attention and monitoring.
                - High: The outbreak poses a severe risk and demands immediate action and containment measures.

                Also, provide a brief explanation for your assessment.

                Chunk Information:
                {chunk_text}
                """
            )
        ]
    )

    for chunk_id, chunk in ac.chunks.items():
        chunk_summary = chunk['summary']

        runnable = PROMPT_RELEVANCE | llm
        llm.grammar = grammar
        
        relevance_response = robust_api_call(runnable.invoke, {
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
                "chunk_text": chunk_text
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

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='...',
                                     epilog='Example usage: python_debug ...',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    parser.add_argument('--directory', type=str, nargs='?', default="", help='directory containing files to be processed')
    parser.add_argument('--paragraph', action='store_true', help='Enable paragraph processing')
    args = parser.parse_args()

    user_query = args.specific_instruction

    argo_embedding_wrapper_instance = ArgoEmbeddingWrapper()	
    argo_embedding = ARGO_EMBEDDING(argo_embedding_wrapper_instance)
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 0.5)

    checkpoint_file = "checkpoint.json"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as file:
            checkpoint_data = json.load(file)
            ac = AgenticChunker.from_json(checkpoint_data["agenticChunker"])
            current_file_index = checkpoint_data["currentFileIndex"]
            # Load other relevant data from the checkpoint
    else:
        ac = AgenticChunker(llm)
        current_file_index = 0

    files = os.listdir(args.directory)
        
    if args.paragraph == False: #organizing papers by topic, i.e., file by file
        for i in range(current_file_index, len(files)):
            filepath = os.path.join(args.directory, files[i])
            if os.path.isfile(filepath):
                paragraphs = split_file_into_paragraphs(filepath, llm)
                print("Main: paragraphs = ",paragraphs)
                classified_sentences = agent_parser(paragraphs,llm)
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
                ac.add_propositions([paper_summary],filepath)

                # Save checkpoint after processing each file
                checkpoint_data = {
                    "agenticChunker": ac.to_json(),
                    "currentFileIndex": i + 1,
                    # Include other relevant data in the checkpoint
                }
                save_checkpoint(checkpoint_data, checkpoint_file)

    if args.paragraph == True:
        for i in range(current_file_index, len(files)):
            filepath = os.path.join(args.directory, files[i])
            if os.path.isfile(filepath):
                paragraphs = split_file_into_paragraphs(filepath, llm)
                #print("Main: paragraphs = ",paragraphs)
                ac.add_propositions(paragraphs,filepath)

                # Save checkpoint after processing each file
                checkpoint_data = {
                    "agenticChunker": ac.to_json(),
                    "currentFileIndex": i + 1,
                    # Include other relevant data in the checkpoint
                }
                save_checkpoint(checkpoint_data, checkpoint_file)
                
    #ac.pretty_print_chunks()
    #ac.pretty_print_chunk_outline()
    print_infectious_disease_chunks(ac, llm)
    print("Number llm calls by AgenticChunker:",ac.llm_call_count)
    
    exit(0)

if __name__ == '__main__':
    main()

