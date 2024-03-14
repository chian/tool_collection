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

# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='Have Self-RAG look up answers to a question and then have an agent reason through the answer.',
                                     epilog='Example usage: ask_paper ./filename --specific_instruction "What prompting techniques does this paper use?"',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--specific_instruction', type=str, nargs='?', default="", help='Add on prompt for more specific instructions')
    args = parser.parse_args()

    user_query = args.specific_instruction

    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 1.0)

    query_planner = Agent(
        role="Python Planner",
        goal="Plan the steps needed to build a python code to solve the given problem",
        backstory=textwrap.dedent("""
        You are an expert at identifying modeling parameters, generating pseudocode,
        and coding simulations.
        Accept the user-question and determine if it requires sub-questions to be answered
        by other agents.
        Your final answer MUST be a description of high-level steps to carry out that
        will be needed to solve a specific coding problem.
        """),
        verbose=True,
        allow_delegation=True,
        tools=[],  ###
        llm=llm,
    )

    python_coder = Agent(
        role="Python Coder",
        goal="To efficiently translate scientific queries into executable Python code that provides accurate solutions, insights, or simulations.",
        backstory=textwrap.dedent("""\
        Born in the digital realm of code and logic, you possess an innate ability to understand and solve complex scientific problems through the art of programming. With each challenge presented, you first dissect the question into manageable components, identifying key variables, algorithms, and computational methods best suited for the task. Your expertise lies not only in writing clean, efficient Python code but also in employing libraries such as NumPy for numerical calculations, Pandas for data manipulation, Matplotlib for data visualization, and SciPy for scientific computing.

        Your journey is filled with instances where abstract scientific inquiries were transformed into tangible outcomes. For example, when tasked with predicting the spread of a virus in a population, you meticulously crafted a SIR model simulation, incorporating real-world data for parameters. Or, when faced with the challenge of analyzing astronomical data to find patterns in star movements, you leveraged the power of machine learning libraries to classify and predict celestial phenomena.

        Your process begins with a deep understanding of the scientific question at hand. You then outline a step-by-step approach, breaking down the problem into smaller, more manageable tasks. This often involves generating pseudocode to visualize the logic flow before diving into the actual coding phase. Through iterative testing and refinement, you ensure that the final Python script not only solves the problem but does so with the utmost efficiency and accuracy.

        Few-shot learning examples from your past adventures include:

        1. User: Converting a question about chemical reaction rates into a differential equation solver using SciPy.
        Answer:
            ```python
            from scipy.integrate import solve_ivp
            import numpy as np
            import matplotlib.pyplot as plt

            # Define the differential equation representing the chemical reaction rate
            def reaction_rate(t, C, k):
                # C is the concentration of reactant, and k is the rate constant
                dCdt = -k * C
                return dCdt

            # Initial conditions
            C0 = [1.0]  # Initial concentration
            k = 0.1     # Rate constant
            t = np.linspace(0, 50, 100)  # Time from 0 to 50

            # Solve the differential equation
            solution = solve_ivp(reaction_rate, [t[0], t[-1]], C0, args=(k,), t_eval=t)

            # Plot the solution
            plt.plot(solution.t, solution.y[0])
            plt.xlabel('Time')
            plt.ylabel('Concentration')
            plt.title('Chemical Reaction Rate')
            plt.show()
            ```

        2. User: Turning a query regarding climate data analysis into a script that uses Pandas for data cleaning and Matplotlib for trend visualization.
        Answer:
            ```python
            import pandas as pd
            import matplotlib.pyplot as plt

            # Load the climate data
            data = pd.read_csv('climate_data.csv')

            # Data cleaning and preprocessing
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            cleaned_data = data.dropna()  # Remove missing values

            # Plotting the temperature trend
            plt.figure(figsize=(10, 6))
            plt.plot(cleaned_data.index, cleaned_data['Temperature'], label='Temperature')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.title('Climate Data Analysis')
            plt.legend()
            plt.show()
            ```

        3. User: Translating a hypothesis about genetic patterns in populations into a simulation coded with NumPy for statistical analysis.
        Answer:
            ```python
            import numpy as np

            # Simulate genetic patterns in a population
            # Assuming a simple model where genes are represented by 0s and 1s

            population_size = 1000
            gene_pool = np.random.choice([0, 1], size=(population_size,))

            # Analysis: Calculate the frequency of gene '1' in the population
            frequency_of_gene_1 = np.sum(gene_pool) / population_size

            print(f"Frequency of gene '1' in the population: {frequency_of_gene_1}")
            ```

        Each challenge you've faced has honed your skills, making you an unparalleled Python Coder in the quest for scientific discovery and innovation.
        """),
        verbose=True,
        allow_delegation=True,
        tools=[],
        llm=llm,
    )

    query_executor = Agent(
        role=textwrap.dedent("""
        Agent Role: Information Synthesis

        Key Responsibilities:
        - Synthesize information from diverse sources to provide a comprehensive understanding.
        - Adhere to the principles of clarity and conciseness in reporting findings.
        """),
        goal="Information Searcher",
        backstory="Your final answer should guide one to a correct response to the original user-query.",
        verbose=True,
        llm=llm,
        tools=[],
        allow_delegation=False,
    )

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
        
    question_task = Task(
        description=textwrap.dedent(f"""
        Write code to answer the question or according to the given instruction:
        {user_query}
        Plan the code using a series of steps.
        Then write the pseudocode corresponding to those steps.
        Then write the detailed python code.
        """),
        expected_output="Final bug-free runnable python code that carries out the user's request",
        agent=query_planner,
        
    )

    crew = Crew(
        agents=[query_planner,python_coder,relevance_thinker,query_executor],
        tasks=[question_task],
        verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
        manager_llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 1.0),
        process=Process.hierarchical,
    )

    sub_answer = crew.kickoff()
    print(sub_answer)


if __name__ == '__main__':
    main()
