import re
from collections import Counter
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
from typing import Literal
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING

def researcher_agents(topic, llm):
    agents = [
        Agent(
            role=f"Researcher {i+1}",
            goal=f"Generate an innovative and groundbreaking hypothesis on the topic: {topic}",
            backstory=f"You are a passionate researcher with expertise in various scientific fields.",
            llm=llm,
        )
        for i in range(3)
    ]

    task = Task(
        description=f"Generate an innovative and groundbreaking hypothesis on the topic: {topic}",
        expected_output="A well-formulated hypothesis that is both innovative and groundbreaking",
        agents=agents,
    )
    
    crew = Crew(
        agents=agents,
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )
    
    result = crew.kickoff()
    return result

class JudgeOutput(BaseModel):
    chosen_hypothesis: Literal["Hypothesis 1", "Hypothesis 2"]
    explanation: str

def judge_hypothesis(hypothesis1, hypothesis2, llm):
    judge = Agent(
        role="Hypothesis Judge",
        goal="Evaluate and critique the given hypotheses based on explanative power and existing data.",
        backstory="You are an expert in evaluating scientific hypotheses using Karl Popper's concept of explanative power.",
        llm=llm,
    )

    task = f"""
    Given the following two hypotheses:

    Hypothesis 1: {hypothesis1}

    Hypothesis 2: {hypothesis2}

    Provide a detailed critique of each hypothesis, considering their explanative power and existing data. Choose the best hypothesis and explain your reasoning.

    Output your answer in the following format:
    ```
    {{
        "chosen_hypothesis": "Hypothesis 1" or "Hypothesis 2",
        "explanation": "Your detailed explanation and reasoning goes here"
    }}
    ```
    """

    result = judge.task(task)
    judge_output = JudgeOutput.parse_raw(result)
    return judge_output

def penalize_overused_words(hypothesis):
    overused_word_groups = {
        "rigor": ["rigor", "rigorous", "meticulous", "thorough"],
        "innovative": ["innovative", "groundbreaking", "pioneering", "cutting-edge"],
        "original": ["original", "unique", "novel", "unprecedented"],
        "creative": ["creative", "imaginative", "inventive", "ingenious"],
        "significant": ["significant", "remarkable", "noteworthy", "outstanding"],
        "robust": ["robust", "reliable", "dependable", "consistent"],
        "comprehensive": ["comprehensive", "extensive", "in-depth", "exhaustive"],
        "state-of-the-art": ["state-of-the-art", "advanced", "sophisticated", "cutting-edge"],
        "breakthrough": ["breakthrough", "game-changing", "transformative", "disruptive"],
        "paradigm-shifting": ["paradigm-shifting", "revolutionary", "trailblazing", "landmark"]
    }
    
    word_counts = Counter(re.findall(r'\b\w+\b', hypothesis.lower()))
    
    penalty_scores = {group: sum(word_counts[word] for word in words) for group, words in overused_word_groups.items()}

    for group, score in penalty_scores.items():
        if score > 3:
            penalty_string = "The hypothesis heavily relies on cliched and overused science words, which severely weakens its impact and credibility. "
            penalty_string += f"The excessive use of words related to '{group}' ({score} occurrences) is particularly egregious and demonstrates a lack of originality. "
    
            penalty_string += "This hypothesis is a prime example of unimaginative and hackneyed writing that fails to contribute anything novel or substantial to the scientific discourse. It is a subpar and uninspired piece of work that should be heavily penalized."
    
    return penalty_string


def improvement_discussion(chosen_hypothesis, explanation, llm):
    reviewer_agent = Agent(
        role="Reviewer",
        goal="Analyze the judge's feedback and identify areas for improvement in the chosen hypothesis.",
        backstory="You are an experienced researcher who can critically examine hypotheses and provide constructive feedback.",
        llm=llm,
    )

    developer_agent = Agent(
        role="Developer",
        goal="Propose specific changes and improvements to the chosen hypothesis based on the reviewer's analysis.",
        backstory="You are a skilled researcher who can refine hypotheses and incorporate feedback effectively.",
        llm=llm,
    )

    reviewer_task = Task(
        description=f"The judge has chosen {chosen_hypothesis} with the following explanation:\n{explanation}\n\nAnalyze the judge's feedback and identify areas where the chosen hypothesis can be improved. Provide your analysis in a clear and concise manner.",
        expected_output="A clear analysis of the judge's feedback, highlighting areas for improvement",
        agents=[reviewer_agent],
    )

    developer_task = Task(
        description=f"Based on the reviewer's analysis, propose specific changes and improvements to {chosen_hypothesis}. Be specific and provide concrete suggestions.",
        expected_output="Concrete suggestions for improving the chosen hypothesis",
        agents=[developer_agent],
    )

    crew = Crew(
        tasks=[reviewer_task, developer_task],
        agents=[reviewer_agent, developer_agent],
        process=Process.sequential,
        manager_llm=llm,
        verbose=True,
    )

    result = crew.kickoff()
    return result


def implement_improvements(chosen_hypothesis, improvement_suggestions, llm):
    implementer_agent = Agent(
        role="Implementer",
        goal="Refine the chosen hypothesis based on the improvement suggestions.",
        backstory="You are a skilled researcher who can effectively implement feedback and improve hypotheses.",
        llm=llm,
    )

    implementer_task = Task(
        description=f"Refine the following hypothesis based on the provided improvement suggestions:\n\nHypothesis: {chosen_hypothesis}\n\nImprovement Suggestions:\n{improvement_suggestions}\n\nGenerate an updated hypothesis that incorporates the improvements.",
        expected_output="An updated hypothesis that incorporates the improvement suggestions",
        agents=[implementer_agent],
    )

    crew = Crew(
        tasks=[implementer_task],
        agents=[implementer_agent],
        process=Process.sequential,
        manager_llm=llm,
        verbose=True,
    )

    result = crew.kickoff()
    return result

# Example usage
topic = "The potential applications of quantum computing in solving causal inference problems"
argo_wrapper_instance = ArgoWrapper()
llm = ARGO_LLM(argo=argo_wrapper_instance, model_type='gpt4', temperature=0.5)

hypothesis1 = "HYPOTHESIS 1:\n"
hypothesis1 += researcher_agents(topic, llm)
print(f"Hypothesis 1: {hypothesis1}")
penalty1 = penalize_overused_words(hypothesis1)
print(f"Hypothesis 1 Penalty: {penalty1}")
hypothesis1 += "\n" + penalty1

hypothesis1 = "HYPOTHESIS 2:\n"
hypothesis2 += researcher_agents(topic, llm)
print(f"Hypothesis 2: {hypothesis2}")
penalty2 = penalize_overused_words(hypothesis2)
print(f"Hypothesis 2 Penalty: {penalty2}")
hypothesis2 += "\n" + penalty2

attempts = 0
while attempts <= 10:
    judge_output = judge_hypothesis(hypothesis1, hypothesis2, llm)
    chosen_hypothesis = judge_output.chosen_hypothesis
    explanation = judge_output.explanation
    print("\nJudge's Evaluation:")
    print(explanation)

    if chosen_hypothesis == "Hypothesis 1":
        hypothesis = hypothesis1
    elif chosen_hypothesis == "Hypothesis 2":
        hypothesis = hypothesis2
    else:
        print("Error in choosing hypothesis")
        exit(0)
        
    improvement_suggestions = improvement_discussion(hypothesis, explanation, llm)
    print("\nImprovement Suggestions:")
    print(improvement_suggestions)
    updated_hypothesis = implement_improvements(chosen_hypothesis, improvement_suggestions, llm)
    print("\nUpdated Hypothesis:")
    print(updated_hypothesis)
