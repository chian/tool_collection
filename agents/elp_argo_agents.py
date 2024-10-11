#eleepalou; 6/6/24
#multiple agents conversing; implementing argo

# current question: "Which of the following is true about the order of a finite group and the order of an element? 
# Pick the letter associated with the correct answer: 
# a) a group's order refers to its cardinality, 
# b) a group's order and an element's order are found in the same way, 
# c) an element's order is found by applying the identity element to it, 
# d) None of the other answers are true"


#same error 422 that ive been getting, i get when i try to run coding_agent.py as well (...)


import argparse
import os, json
import numpy as np
import requests
import time
import concurrent.futures
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import textwrap

from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt4", help="the model to use")
    parser.add_argument("question", type=str, help="the question being asked")
    args = parser.parse_args()

    question = args.question

    # user_query = args.specific_instruction

    argo_wrapper_instance = ArgoWrapper()
    # #adapt temperature - put higher for less accuracy?? (will take fewer commands?)
    llm = ARGO_LLM(argo=argo_wrapper_instance, model_type='gpt4', temperature = 1.0)


    #agents based off 6 hats thinking method
    #green hat: generate ideas to solve problem from creative perspective - think outside of the box, supply creative alternative ideas
    agentGreen = Agent(role = "Cutting edge researcher",
                      goal = """Provide out of the box solutions and creative alternative ideas to what is being mentioned by the other researchers""",
                      backstory = """You are an experienced researcher that is most interested in innovative, new ways of approaching problems 
                      and is currently in a group specializing in computer science and math""",
                      allow_delegation = True,
                      verbose = True,
                      llm = llm)

    #changed red hats goal, etc because i was running into the issue of no agent giving a definitive answer to the mult choice question - instead were all too verbose and just discussing the concepts in the question
    #red hat: straightforward, gives clear and consice answer (*changed bc emotional impact isnt useful for this - used to be: gauge emotional impact and get gut feelings)
    agentRed = Agent(role = "Straightforward researcher",
                      goal = """Listen to all the information presented by your fellow researchers and come to a direct and concise 
                      answer for the question originally posed""",
                      backstory = """You are a math and computer science researcher who likes to create and present 
                      clear, straightforward, and brief answers and solutions""",
                      allow_delegation = False, #cannot delegate bc it needs to give a definitive answer
                      verbose = False,
                      llm = llm)


    #white hat: understand issues and present facts
    agentWhite = Agent(role = "Analytical researcher",
                      goal = """Logically present the facts that you know and pursue the clearest, logical solution""",
                      backstory = """You are an excellent math and computer science researcher who centers their thinking and problem solving skills around the value of logic and direct reasoning and facts""",
                      allow_delegation = True,
                      verbose = True,
                      llm = llm)

    #evaluate ideas by listing their benefits and look for positive outcomes
    agentYellow = Agent(role = "Optimistic researcher",
                      goal = """Provide positive outcomes and list the benefits that you know""",
                      backstory = """You are an optimistic math and computer science researcher that likes to focus on the positives, 
                                     like what they know, and they evaluate potential solutions by focusing on the positive aspects and benefits of each""",
                      allow_delegation = True,
                      verbose = True,
                      llm = llm)

    #used to manage the process, facilitate conversation, summarize discussion
    agentBlue = Agent(role = "Research group leader",
                      goal = """Listen and manage the discussion between your group's researchers, directing attention to the ideas that make the most sense""",
                      backstory = """You are are research group leader specialized in math and computer science that likes to facilitate productive conversations between the researchers in your group.""",
                      allow_delegation = True,
                      verbose = True,
                      llm = llm)


    #evaluate ideas by listing drawbacks and predict negative outcomes
    agentBlack = Agent(role = "Pessimistic researcher",
                      goal = """Provide negative outcomes and list the disadvantages that you know""",
                      backstory = """You are a pessimistic math and computer science researcher that prefers to focus on the negatives, 
                                     like what they know is wrong or do not know, and they evaluate potential solutions by focusing on the negative aspects and disadvantages of each""",
                      allow_delegation = True,
                      verbose = True,
                      llm = llm)
    
    taskBlue = Task(description=("Pose the following question to your researchers: %s Help to moderate the discussion to keep it civil and productive." % question),
             agent = agentBlue,
             expected_output="Natural language comment about the responses from other researchers")

    taskGen = Task(description=question,
             agents = [agentGreen, agentRed, agentWhite, agentYellow, agentBlack],
             expected_output="A multiple choice selection.")


    crew = Crew(
        agents=[agentBlue, agentRed, agentWhite, agentYellow, agentGreen, agentBlack],
        tasks=[taskBlue, taskGen],
        verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
        manager_llm = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 1.0),
        process=Process.hierarchical,
    )

    response = llm.invoke("What is your name?")
    print(response)
    exit(0)
    res = crew.kickoff()
    print(res)

if __name__ == '__main__':
    main()