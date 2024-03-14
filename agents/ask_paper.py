#!/usr/bin/env python3

#Function that finds information in a text file for you.
import argparse
import os, json
import numpy as np
import time
import concurrent.futures
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import textwrap
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING

# USER SET PARAMETERS
ENDPOINT = "argo"
user_query = "Who leads the CDC?"
file_name = "cleaned_texts_r3.txt"
RETRIEVE_STOP = 3
search_depth = 10

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

def clean_string(text):
    return text.replace('`', '').replace('\n', '')

def half_self_rag(sub_query,docs):
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 1.0)
    relevance_thinker = Agent(
        role=textwrap.dedent("""
            Agent Role: Relevance Assessor

            Primary Objectives:
            1. Come up with criteria or subquestions that will help you decide whether a document is relevant or irrelevant
            to a question.
            2. Decide is text provided answers the question. If so, it is relevant.
            3. If the text provided does not in any way help answer the question presented, then it is irrelevant.
        """),
        goal="Information Assessment",
        backstory=textwrap.dedent("""
            Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating 
            the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential 
            biases or flaws in thinking.
        """),
        verbose=True,
        llm=llm,
        tools=[],
        allow_delegation=False,
    )

    relevance_classifier = Agent(
        role=textwrap.dedent("""
            Agent Role: Classifier

            Primary Objective: Take prior thoughts and classify the overall relevance of the text. Provide a one word answer that is
            either "RELEVANT" or "IRRELEVANT"
        """),
        goal="Relevance Classification",
        backstory="Expert at taking a prior chain-of-thought and categorizing text provides as either \"RELEVANT\" or \"IRRELEVANT\"",
        verbose=True,
        llm=llm,
        tools=[],
        allow_delegation=False,
    )

    classification_task = Task(
        description="Reading the thoughts above, provide a final one-word answer of \"RELEVANT\" or \"IRRELEVANT\"",
        agent=relevance_classifier
    )

    relevant_docs = []
    answers = []
    doc_index = 0
    num_relevant = 0
    for doc in docs:
        relevance_task = Task(
            description=textwrap.dedent(f"""
            Your task is to identify if the text below:
                    {doc}
                Is relevant to the query:
                    {sub_query}
            
                Explain your thoughts step-by-step and provide the final answer of "RELEVANT" or "IRRELEVANT"
             """),
            agent=relevance_thinker
        )
        crew = Crew(
            agents=[relevance_classifier,relevance_thinker],
            tasks=[relevance_task,classification_task],
            verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
            process=Process.sequential,
        )

        answer = crew.kickoff()
        #print("DOC:", doc)
        #print("ANSWER:", answer)
        #print("######################")
        answers.append(answer)
        if clean_string(answers[doc_index]) == 'RELEVANT':
            relevant_docs.append(doc)
            num_relevant += 1
        time.sleep(5)
        if num_relevant == RETRIEVE_STOP:
            break
        doc_index += 1
    return relevant_docs

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='Have Self-RAG look up answers to a question and then have an agent reason through the answer.',
                                     epilog='Example usage: ask_paper ./filename --specific_instruction "What prompting techniques does this paper use?"',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_path', type=str, help='Text file to answer from')
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    args = parser.parse_args()

    file_name = args.input_path
    user_query = args.specific_instruction

    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            cleaned_text = file.read()
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    argo_embedding_wrapper_instance = ArgoEmbeddingWrapper()  # Assuming this is how you initialize it
    argo_embedding = ARGO_EMBEDDING(argo_embedding_wrapper_instance)
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 1.0)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splits = text_splitter.create_documents([cleaned_text])

    relevant_docs = half_self_rag(user_query,splits)

    query_executor = Agent(
        role=textwrap.dedent("""
        Agent Role: Information Synthesis

        Key Responsibilities:
        - Synthesize information from diverse sources to provide a comprehensive understanding of the disease and its impact.
        - Adhere to the principles of clarity and conciseness in reporting findings.
        - Your final answer MUST be a correct response to the original user-query
        """),
        goal="Information Searcher",
        backstory="Your final answer MUST be a correct response to the original user-query.",
        verbose=True,
        llm=llm,
        tools=[],
        allow_delegation=False,
    )

    question_task = Task(
        description=textwrap.dedent(f"""
        Your task is to use the following information: 
        {relevant_docs}
        in the context of the original user request:
        {user_query}
        If not, then say not and what is information is missing.
        If so, then answer the question succinctly.
        """),
        agent=query_executor,
    )

    crew = Crew(
        agents=[query_executor],
        tasks=[question_task],
        verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
        process=Process.sequential,
    )

    sub_answer = crew.kickoff()
    print(sub_answer)


if __name__ == '__main__':
    main()
