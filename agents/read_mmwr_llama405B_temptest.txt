#!/usr/bin/env pythonA
import pdb
import argparse
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
import sys, os
import collections
import string

# Add the project directory to the sys.path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../argo'))
sys.path.append(project_path)
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

def group_similar_answers(answer_counts):
    grouped_answers = collections.defaultdict(int)
    for answer, count in answer_counts.items():
        # Remove any leading/trailing whitespace, convert to lowercase, and remove punctuation
        normalized_answer = answer.strip().lower().translate(str.maketrans('', '', string.punctuation))
        
        # Use a regular expression to extract the potential risks
        match = re.search(r'potential risks:\s*(.*)', normalized_answer)
        if match:
            risks = match.group(1)
            risks = re.sub(r'[\(\)]', '', risks)  # Remove parentheses
            risks = re.sub(r'\s+', ' ', risks).strip()  # Normalize whitespace
            risks = risks.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            grouped_answers[risks] += count
        else:
            grouped_answers[normalized_answer] += count
    
    return grouped_answers

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

    def chunk_content(content, words_per_chunk=10000):
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

    PROMPT_SUMMARIZE_TEXT = PromptTemplate.from_template(
                """
                Summarize the final responses from the following reports into a single report.

                Text from Report:
                {content}

                In a very short and single phrase, list your answer.                                           
                If there is a disease from the above list mentioned, please begin your response with
                "POTENTIAL RISKS:" followed by the names of the infectious agents. If there are no significant
                risks, respond with "NO POTENTIAL RISKS". Note that "POTENTIAL RISKS:" and "NO POTENTIAL RISKS"                                        
                are mutually exclusive and should not occur together. 
                
                A mention of any POTENTIAL RISKS overrides any items listed as "FINAL ANSWER: None". The mention
                of any risk means there is a risk and you need to report it.
                """
    )

    PROMPT_NUMINFECTED_TEXT = PromptTemplate.from_template(
                """
                Please identify the number of people infected by the disease or diseases reported in the risk assessment below:
                
                Disease Report:
                {disease}

                Using the information from the following report.

                Information:
                {content}

                In a very short and single phrase, list your answer. Do not report on anything other than what is listed
                in the initial Disease Report as a potential risk.                                           
                If the disease in the Disease Report is not included in the Information section above, then return
                "NOT REPORTED". If the information is present, then please report a number for each disease separately.
                """
    )

    PROMPT_LOCATION_TEXT = PromptTemplate.from_template(
                """
                Please identify the outbreak location using latitude and longitude for the disease or diseases 
                reported in the risk assessment below:
                
                Disease Report:
                {disease}

                Using the information from the following report.

                Information:
                {content}

                In a very short and single phrase, list your answer. Do not report on anything other than what is listed
                in the initial Disease Report as a potential risk. Use the following examples for latitude-longitude
                reporting format: 

                46.7296° N, 94.6859° W; 
                35.7596° N, 79.0193° W; 
                40.3838° N, 81.1086° W; 
                35.5175° N, 86.5804° W; 
                43.7844° N, 88.7879° W; 
                38.5976° N, 80.4549° W

                If the disease in the Disease Report is not included in the Information section above, then return
                "NOT REPORTED". If the information is present, then please report the location of each outbreak separately.
                """
    )

    num_iterations = 20  # Number of times to run the process
    results = []
    result_counts = collections.Counter()

    for i in range(num_iterations):
        responses = []
        for content in chunks:
            outbreak_analysis_prompt = PROMPT_FIND_OUTBREAK_TEXT.format(content=content)
            from token_count import TokenCount
            tc = TokenCount(model_name="gpt-3.5-turbo")
            tokens = tc.num_tokens_from_string(outbreak_analysis_prompt)

            model = 'llama31-405b-fp8'
            outbreak_analysis_response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=1.0,
                messages=[
                    {"role": "user", "content": outbreak_analysis_prompt}
                ],
            )
            #print(outbreak_analysis_response)
            response = outbreak_analysis_response.choices[0].message.content
            responses.append(response)

            if 1:
                outbreak_number_prompt = PROMPT_NUMINFECTED_TEXT.format(disease=response, content=content)
            
                model = 'llama31-405b-fp8'
                outbreak_number_response = client.chat.completions.create(
                    model=model,
                    max_tokens=4096,
                    temperature=1.0,
                    messages=[
                        {"role": "user", "content": outbreak_number_prompt}
                    ],
                )
                number = outbreak_number_response.choices[0].message.content
                print("NUMBER:", number)
                #responses.append(outbreak_analysis_response.choices[0].message.content)

            if 1:
                outbreak_location_prompt = PROMPT_LOCATION_TEXT.format(disease=response, content=content)
            
                model = 'llama31-405b-fp8'
                outbreak_location_response = client.chat.completions.create(
                    model=model,
                    max_tokens=4096,
                    temperature=1.0,
                    messages=[
                        {"role": "user", "content": outbreak_location_prompt}
                    ],
                )
                location = outbreak_location_response.choices[0].message.content
                print("LOCATION:", location)
                #responses.append(outbreak_analysis_response.choices[0].message.content)

        final_responses = "\n".join(responses)
        print(final_responses)

        summarize_prompt = PROMPT_SUMMARIZE_TEXT.format(content=final_responses)
        final_response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": summarize_prompt}
            ],
        )

        result = final_response.choices[0].message.content
        results.append(result)

        # Update the running count
        result_counts.update([result])

        if sys.stdout.isatty() and False:
            # Print the current state of the counts, overwriting the previous line
            sys.stdout.write("\r")
        sys.stdout.write(f"Iteration {i+1}/{num_iterations} - Current Counts: {dict(result_counts)}")
        sys.stdout.write(f"\n")
        sys.stdout.flush()

    # Final results
    #print("\nFinal Results:")
    grouped_answers = group_similar_answers(result_counts)
    #for answer, count in sorted(grouped_answers.items(), key=lambda x: x[1], reverse=True):
    #print(f"{count}: {answer}")
    top_grouped_answers = sorted(grouped_answers.items(), key=lambda x: x[1], reverse=True)[:2]
    
    for answer, count in top_grouped_answers:   
        print(f"{count}: {answer}")

    total_runs = sum(grouped_answers.values())
    most_common_results = [(answer, count) for answer, count in top_grouped_answers]

    #most_common_results = result_counts.most_common(2)
    #total_runs = sum(result_counts.values())

    # Prepare the output line
    output_line = []
    for result, count in most_common_results:
        certainty_estimate = count / total_runs * 100
        output_line.append(f"{certainty_estimate:.2f}% for \"{result}\"")
    output_line.append(f"Total tests run: {num_iterations}")

    if sys.stdout.isatty() and False:
        # Print the final formatted line
        sys.stdout.write("\r")
    sys.stdout.write("     ".join(output_line) + " " * (80 - len("     ".join(output_line))))
    sys.stdout.write(f"\n")
    sys.stdout.flush()

if __name__ == '__main__':
    main()



