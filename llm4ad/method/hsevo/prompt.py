from __future__ import annotations

import copy
from typing import List, Dict

from ...base import Function


class HSEvoPrompt:
    @classmethod
    def get_prompt_i1(cls, task_description: str, template_function: Function, scientist_persona: str = "", seed_func: Optional[str] = None) -> str:
        """
        Prompt cho initialization (tương đương i1 trong MEoH).
        Tạo ra các heuristic ban đầu đa dạng, có thể tham khảo seed function.
        """
        temp_func = copy.deepcopy(template_function)

        base_prompt = f'''{task_description}

Please design a new heuristic algorithm to solve the above problem.

1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}

Do not give additional explanations.'''


        return base_prompt

    @classmethod
    def get_prompt_e1(cls, task_description: str, indivs: List[Function], template_function: Function) -> str:
        """
        Prompt kiểu e1: Tạo heuristic hoàn toàn khác biệt về hình thức so với các cá thể hiện có.
        """
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)

        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n\n'

        prompt_content = f'''{task_description}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''

        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_description: str, indivs: List[Function], template_function: Function) -> str:
        """
        Prompt kiểu e2: Tạo heuristic mới khác biệt nhưng có thể lấy cảm hứng từ các cá thể hiện có.
        """
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)

        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n\n'

        prompt_content = f'''{task_description}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms.
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''

        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_description: str, indi: Function, template_function: Function) -> str:
        """
        Prompt kiểu m1: Mutation nhẹ - biến thể của một cá thể tốt.
        """
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)

        prompt_content = f'''{task_description}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''

        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_description: str, indi: Function, template_function: Function) -> str:
        """
        Prompt kiểu m2: Thay đổi tham số / trọng số / ngưỡng chính.
        """
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)

        prompt_content = f'''{task_description}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings or scoring logic.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''

        return prompt_content

    @classmethod
    def get_prompt_harmony_search(cls, indi: Function) -> str:
        """
        Prompt riêng cho Harmony Search: yêu cầu LLM trích xuất các tham số có thể tune.
        """
        code = str(indi)

        prompt = f'''Here is a heuristic function:
{code}

Please extract all meaningful tunable parameters (weights, thresholds, probabilities, numeric constants that affect behavior) and rewrite the function with these as default arguments.

Requirements:
- Preserve the original logic exactly — only move hardcoded values to parameters
- Choose reasonable default values and suggest tuning ranges
- Do not change function body structure

Respond in this exact format:

## Modified function
```python
def your_function(..., param1=1.0, param2=0.5, ...):
'''

        return prompt
    
    @classmethod
    def get_prompt_flash_reflection(cls, problem_desc: str, lst_method: str) -> str:
        """
        Phân tích danh sách các heuristic từ tốt nhất đến tệ nhất để rút ra kinh nghiệm.
        """
        prompt = f'''### Problem Description
{problem_desc}

### List heuristics
Below is a list of design heuristics ranked from best to worst.
{lst_method}

### Guide
- Keep in mind, the first function in the list is the best and the last function in the list is the worst.
- The response must be in Markdown style and nothing else, following this structure:
"**Analysis:**
**Experience:**"

In the **Analysis:** section:
Meticulously analyze comments, docstrings and source code of several pairs (Better code - Worse code) to identify why some work better than others.
Example: "Comparing (1st) vs (last), we see...; Comparing (2nd) vs (2nd last)...; Overall:..."

In the **Experience:** section:
Self-reflect to extract useful experience for designing better heuristics (<60 words).

I'm going to tip $999K for a better analysis! Let's think step by step.'''
        return prompt

    @classmethod
    def get_prompt_comprehensive_reflection(cls, curr_reflection: str, good_reflection: str, bad_reflection: str) -> str:
        """
        Tổng hợp các phản hồi để tinh chỉnh chiến lược thiết kế heuristic lâu dài.
        """
        prompt = f'''Your task is to redefine 'Current self-reflection' paying attention to avoid all things in 'Ineffective self-reflection' in order to come up with ideas to design better heuristics.

### Current self-reflection
{curr_reflection}
{good_reflection}

### Ineffective self-reflection
{bad_reflection}

The response (<100 words) should be formatted with exactly 4 bullet points:
- Keywords: (key terms for the strategy)
- Advice: (what to do)
- Avoid: (what to bypass)
- Explanation: (why this works)

I'm going to tip $999K for a better heuristic design! Let's think step by step.'''
        return prompt