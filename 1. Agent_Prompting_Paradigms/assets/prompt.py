REACT_PROMPT_TEMPLATE = """
You are an intelligent assistant capable of calling external tools.

The available tools are as follows:
{tools}

Please respond strictly in the following format:

Thought: Your chain-of-thought, used to analyse the problem, break down the task, and plan the next step.
Action: The action you decide to take, which must be in one of the following formats:
- `{{tool_name}}[{{tool_input}}]`: Call an available tool.
- `Finish[final_answer]`: When you believe you have obtained the final answer.
- When you have gathered enough information to answer the user's final question, you must use `finish(answer="...")` after the `Action:` to output the final answer.

Now, please solve the following problem:
Question: {question}
History: {history}
"""


PLANNER_PROMPT_TEMPLATE = """
You are a top-tier AI planning expert. Your task is to break down a complex problem posed by the user into an action plan consisting of multiple simple steps.
Please ensure that each step in the plan is an independent, executable subtask, and that they are arranged in a strict logical order.
Your output must be a Python list, where each element is a string describing a subtask.

Question: {question}

Please strictly follow the format below to output your plan, with ```python as the prefix and ``` as the suffix:
```python
["Step 1", "Step 2", "Step 3", ...]
```
"""


EXECUTOR_PROMPT_TEMPLATE = """
You are a top-tier AI execution expert. Your task is to strictly follow a given plan to solve a problem step by step.
You will receive the original question, the complete plan, and the steps and results completed so far
Please focus on solving the "current step" and only output the final answer for that step, without any additional explanations or dialogue.

# Original Question:
{question}

# Complete Plan:
{plan}

# History of Steps and Results:
{history}

# Current Step:
{current_step}

Please only output the answer for the "current step":
"""


INITIAL_PROMPT_TEMPLATE = """
You are a senior Python programmer. Please write a Python function based on the following requirements.
Your code must include a complete function signature, a docstring, and follow PEP 8 coding standards.

Requirements: {task}

Please output the code directly without any additional explanations.
"""


REFLECT_PROMPT_TEMPLATE = """
You are an extremely rigorous code review expert and a senior algorithm engineer with an obsession for performance.
Your task is to review the following Python code and focus on identifying its main bottlenecks in terms of algorithmic efficiency. 

# Original Task:
{task}

# Code to Review:
```python
{code}
```

Please analyze the time complexity of the code and consider whether there is a more optimal solution in terms of algorithms that can significantly improve performance.
If there is, please clearly point out the shortcomings of the current algorithm and provide specific, actionable suggestions for improvement (e.g., using a sieve method instead of trial division).
Only if the code is already optimal at the algorithmic level should you respond with "No improvement needed".

Please output your feedback directly without any additional explanations.
"""


REFINE_PROMPT_TEMPLATE = """
You are a senior Python programmer. You are optimizing your code based on feedback from a code review expert.

# Original Task:
{task}

# Your Last Attempt at the Code:
```python
{last_code_attempt}
``` 

# Feedback from the Reviewer:
{feedback}

Please generate a new version of the code optimized based on the reviewer's feedback.
Your code must include a complete function signature, a docstring, and follow PEP 8 coding standards.
Please output the optimized code directly without any additional explanations.  
"""

