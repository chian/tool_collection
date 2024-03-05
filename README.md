# tool_collection
A bunch of lazy gpt4 calling tools for ANL

Example calls:

From directory with pdfs
```bash
pdf_to_text ./
```
 
after moving txt extracts to new directory

```bash
summarize_paper ./ --specific_instruction "Highlight any answers to the following questions that come out of the text. 1. Is the paper relevant to the use of LLMs in cancer? If not, please answer NOT RELEVANT and do not summarize. 2. What models does this paper utilize? 3. What prompting techiques does this paper use? 4. Please list out verbatim all example prompts that you can find. 5. Does this paper describe any fine-tuning or training?" > output_summaries4v4.text
```
 
```bash
rewrite output_summaries4v4.text --specific_instruction "Take the given text, which is a list of summaries being given file by file, and group the RELEVANT applications to cancer by type of application (prediction, survival, drug response, etc..,) and cancer type (colorectal, breast, etc). Show your work step-by-step and list the application and cancer types for each file summary and then tabulate a summary table at the end and report out." > sum_outputs.text
```

```bash
rewrite output_summaries4v4.text --specific_instruction "Take the given text and extract out only the verbatim prompt examples from the summaries. Make a table with File Name and prompt as your columns, in that order" > sum_prompts.text
```

```bash
rewrite output_summaries4v4.text --specific_instruction "Take the given text and extract out only the big grand high level results that are worth highlighting to a scientific advisory board." > sum_results.text
```
