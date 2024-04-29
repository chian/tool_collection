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
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import subprocess

# USER SET PARAMETERS
ENDPOINT = "argo"

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

@tool("Python Code Executor")
def python_code_executor(code: str, command_args: str = "") -> str:
    """Executes Python code and returns the output or the error."""
    try:
        # Execute the python code using subprocess
        completed_process = subprocess.run(
            ["python", "-c", code] + command_args.split(), 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return completed_process.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

def reverse_llm_output(output):
    """Reverses the output of an LLM discussion."""
    # Assuming the output is a string, reverse it
    return output[::-1]

def assess_code_purpose_and_reverse_if_needed(original_code, modified_code, llm):
    """Assesses if the purpose of the modified code has changed. If so, reverses changes."""
    # This is a placeholder for the actual logic to assess code purpose
    # Let's assume it returns True if the purpose has changed, False otherwise
    purpose_changed = False  # Placeholder for actual assessment logic
    if purpose_changed:
        # If the purpose has changed, reverse the modifications by using the original code
        return original_code
    else:
        # If the purpose hasn't changed, proceed with the modified code
        return modified_code

# MAIN CODE BLOCK
def main():


    parser = argparse.ArgumentParser(description='...',
                                     epilog='Example usage: python_debug ...',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    parser.add_argument('--filepath', type=str, nargs='?', default="bug_example.py", help='file to be run')
    parser.add_argument('--command_args', type=str, nargs='?', default="", help='command args')
    args = parser.parse_args()

    user_query = args.specific_instruction

    # Read the content of the file specified in args.filepath
    with open(args.filepath, 'r') as file:
        original_code = file.read()

    execute_code = args.filepath + " " + args.command_args

    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt35', temperature = 1.0)

    # Example usage
    # Assuming the decorator approach was used to create the tool
    script_executor = Agent(
        role='Python Code Execution Specialist',
        goal='Execute Python code and return results or errors.',
        backstory='A specialist capable of running Python code snippets securely.',
        tools=[python_code_executor],
        llm=llm,
        verbose=True
    )

    debugger = Agent(
        role='Debugger',
        goal='Read code, analyze error messages, and suggest debugging steps',
        backstory='Expert in analyzing Python error messages and suggesting fixes.',
        tools=[],
        llm=llm,
        verbose=True
    )

    code_writer = Agent(
        role='Python Programmer',
        goal='Read code and debugging suggestions, then rewrite code incorporating the suggested fixes',
        backstory='Expert in implementing detailed Python code according to instructions.',
        tools=[],
        llm=llm,
        verbose=True
    )
    
    purpose_preserver = Agent(
        role='Purpose Preserver',
        goal='Understand the original purpose of the code and reject changes that alter that purpose.',
        backstory='An agent trained to analyze code and its modifications to ensure the core intent remains unchanged.',
        tools=[],  # Tools would be specific to code analysis and natural language understanding
        llm=llm,
        verbose=True
    )
    
    # Task to execute the original code
    task_execute_original_code = Task(
        description='Execute the original Python code to understand its purpose. The command line for executing this code is ' + execute_code,
        expected_output='Output or behavior of the original code.',
        agent=script_executor
    )

    # Task to debug the original code
    task_debug_original_code = Task(
        description='Analyze any error messages from executing the Python code and suggest debugging steps. ' + original_code,
        expected_output='Suggestions for addressing the error given',
        agent=debugger,
        #context=[task_execute_original_code]  # Depends on the output of executing the original code
    )

    # Task to revise the original code based on debugging suggestions
    task_revise_original_code = Task(
        description='Revise the Python code based on the debugging suggestions. ' + original_code,
        expected_output='Revised Python code incorporating the debugging suggestions',
        agent=code_writer,
        #context=[task_debug_original_code]  # Depends on the debugging suggestions
    )

    # Task to assess if the purpose of the code has changed after modifications
    task_assess_code_purpose = Task(
        description='Assess the purpose of the original code: ' + original_code,
        expected_output='Determination if the code purpose has changed.',
        agent=purpose_preserver,
        #context=[task_revise_original_code]  # Depends on the revised code
    )

    # Update the crew configuration to include the new tasks
    crew = Crew(
        agents=[script_executor, debugger, code_writer, purpose_preserver],
        tasks=[task_assess_code_purpose, task_execute_original_code, task_debug_original_code, task_revise_original_code],
        process=Process.sequential,
        #manager_llm=llm,
        full_output=True,
        verbose=True,
    )

    # Execute tasks in the correct order and save the modified code
    # This is a conceptual example; you'll need to adapt it based on how your task execution and result retrieval are implemented
    result = crew.kickoff()
    
    # Assuming 'output' contains the revised code
    modified_code = result.get('output')  # This line is conceptual and depends on your implementation

    # Save the modified code to a file
    #with open('modified_code.py', 'w') as file:
    #    file.write(modified_code)

    # Process the result
    print(modified_code)

    exit(0)

if __name__ == '__main__':
    main()
