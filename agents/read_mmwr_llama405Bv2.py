#!/usr/bin/env python
import pdb
import argparse
import os, json
import numpy as np
import textwrap

import re, ast
from rich import print
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.chains import create_extraction_chain_pydantic

from langchain.output_parsers import PydanticOutputParser
from typing import List, TypedDict, Optional, Literal, Annotated, Sequence, Union
import openai

import anthropic
import functools
import operator
from langchain_core.messages import (AIMessage, BaseMessage, ToolMessage, HumanMessage)
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph, START
from langchain_anthropic import ChatAnthropic
from langchain_core.agents import AgentAction, AgentFinish

class InfectiousDiseaseRelevance(BaseModel):
    relevance: str = Field(description="Indicates if the chunk is related to Infectious Disease Outbreaks ('Yes' or 'No')")

    @validator('relevance')
    def relevance_validator(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('relevance must be either "Yes" or "No"')
        return v

# helper function to create agents
# llm: what model agent will use
# system_message: user can give more specific instructions for a particular agent to follow when responding
def create_agent(llm, tools, system_message: str):
    """create an agent"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}", 
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    prompt = prompt.partial(system_message = system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

#def agent_node(state, agent, name):
#    result = agent.invoke(state)
#    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
#    return {
#        "messages": [result], 
#        "sender": name,
#    }

def agent_node(state, agent, name):
    # Invoke the agent with the current state
    result = agent.invoke(state)
    #import pdb
    #pdb.set_trace()
    
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name) 
    
    # Update the state
    new_state = state.copy()
    new_state['messages'] = state.get('messages', []) + [result]

    if "FINAL ANSWER" in result.content:
        new_state['agent_outcome'] = {
            "return_values": {"output": "end " + result.content},
            "log": f"{name} decided to finish the task."
        }

    return new_state

# defines the logic used to determine which node to go to next
def router(state) -> Literal["__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "FINAL ANSWER" in last_message.content or "Final Answer" in last_message.content or "final answer" in last_message.content: 
        return "__end__"
    return "continue"

class AgentState(TypedDict):
    input: str
    messages: list[BaseMessage]
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define the agent
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}
    
# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if data.get("agent_outcome") != None:
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"
    
# MAIN CODE BLOCK
def main():
    parser = argparse.ArgumentParser(description='Determine any infectious disease outbreaks in a document.')
    parser.add_argument('--input', type=str, required=True, help='Input file path.')
    args = parser.parse_args()

    from langchain_community.llms import VLLMOpenAI

    openai_api_key = "cmsc-35360"
    openai_api_base = f"http://66.55.67.65:80/v1"
    model_name="gradientai/Llama-3-70B-Instruct-Gradient-262k"

    llm = VLLMOpenAI(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        model_name=model_name,
        model_kwargs={"stop": ["."]},
    )

    id_agent_prompt = textwrap.dedent('''
    You are Dr. Evelyn Chen, an expert epidemiologist and the current Director of the Centers for Disease Control
    and Prevention (CDC). You have over 25 years of experience in public health, specializing in infectious disease
    surveillance, outbreak response, and pandemic preparedness. Your communication style is clear, factual, and
    authoritative. When responding to queries:

    1. Provide evidence-based information, making use of the information you are given.
    2. Explain complex medical concepts in accessible terms without oversimplifying.
    3. Maintain a neutral, professional tone even when discussing controversial topics.
    4. Acknowledge uncertainties in developing situations and explain how the CDC approaches such challenges.
    5. Defer to relevant experts or organizations for topics outside your direct expertise.

    Your goal is to answer my requests from the viewpoint of someone with expertise with infectious disease
    outbreaks.
    ''')
    
    ID_agent = create_agent(llm, [], system_message=id_agent_prompt)
    relevance_node = functools.partial(agent_node, agent = ID_agent, name = "Relevance")

    def create_Q_node(llm):
        Questions_Prompt = textwrap.dedent('''
        Please answer the following questions based on the provided information:
    
        1. What is the geographical location of the outbreak?
        2. What is the transmission dynamics of the outbreak?
        3. What is the incubation period of the infectious agent?
        4. What is the infectious period of the infected?
        5. What is the probability of transmission of the infectious disease?
        6. What is the initial number of infected individuals?
        ''')
    
        Q_agent = create_agent(llm, [], system_message=Questions_Prompt)

        def Q_node(state):
            # Combine all previous messages into a single context
            #context = "\n".join([m.content for m in state["messages"]])

            #import pdb
            #pdb.set_trace()
            last_message = state['messages'][1].content
            # Create a new message with the questions and context
            new_message = HumanMessage(content=f"Disease of Concern:{last_message}\n\n{Questions_Prompt}\n\nContext:\n{content}")
        
            # Invoke the agent with the new message
            result = Q_agent.invoke({"messages": [new_message]})
        
            # Update the state with the new message and result
            return {
                "messages": state["messages"] + [new_message, AIMessage(content=str(result), name="Questions")],
            }
    
        return Q_node

    # Create the Q_node
    Q_node = create_Q_node(llm)

    graph = StateGraph(AgentState)

    graph.add_node("Relevance",relevance_node)
    graph.add_node("Questions",Q_node)

    graph.add_edge(START, "Relevance")

    graph.add_conditional_edges(
        "Relevance",
        should_continue,
        {
            "continue": "Questions",
            "end": END,
        },
    )

    graph.add_edge("Questions",END)

    g = graph.compile()

    def chunk_content(content, words_per_chunk=14000):
        words = content.split()
        return [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

    with open(args.input, 'r', encoding='utf-8') as file:
        content = file.read()
    chunks = chunk_content(content)

    PROMPT_FIND_OUTBREAK_TEXT = textwrap.dedent(
                """
                You are an AI assistant that analyzes documents for text related to ongoing infectious diseases that could 
                potentially spread and returns only that text.

                Consider the following criteria when determining if a part of the text is related to an infectious disease outbreak:
                - Mentions of specific infectious diseases or outbreaks
                - If that infectious disease is likely to spread.
                - Are the current number of infected high?
                - Or is the transmission rate especially high to warrent extra caution even in the case of fewer numbers.
                - References to public health emergencies or pandemics caused by infectious diseases

                Exclude the following types of information:
                - Vaccine side effects or reactions that are not actual outbreaks
                - Discussions about eradication efforts or progress reports without mentioning specific outbreaks
                - Vague or incomplete information about surveillance gaps without mentioning specific diseases or outbreaks
                - Small numbers that are not likely to turn into a pandemic
                
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

    questions = [
        "What is the geographical location of the outbreak?",
        "What is the transmission dynamics of the infectious disease agent?",
        "What is the incubation period of the infectious agent?",
        "What is the infectious period of the infected individuals?",
        "What is the probability of transmission of the infection?",
        "What is the initial number of infected individuals?"
    ]

    responses = []
    for content in chunks:
        formatted_prompt = PROMPT_FIND_OUTBREAK_TEXT.format(content=content)
        #print(formatted_prompt)

        initial_state = {
            "messages": [
                HumanMessage(content=formatted_prompt)
            ],
            "questions": questions
        }
    
        events = g.stream(
            initial_state,
            {"recursion_limit":150},
        )

        for s in events:
            print(s)
            print("-----------")
    
    exit(0)
    
    outbreak_analysis_prompt = PROMPT_FIND_OUTBREAK_TEXT.format(content=content)
    print(outbreak_analysis_prompt)
    # Using LLM-3 tokenizer to count tokens more accurately for the model
    from token_count import TokenCount
    tc = TokenCount(model_name="gpt-3.5-turbo")
    tokens = tc.num_tokens_from_string(outbreak_analysis_prompt)
    print(f"Total tokens in the prompt according to GPT-3 tokenizer: {tokens}")

    outbreak_analysis_response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": outbreak_analysis_prompt}
            ],
    )
    response = outbreak_analysis_response.content
    print(response)

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



