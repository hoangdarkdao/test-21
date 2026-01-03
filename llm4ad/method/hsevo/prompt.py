from __future__ import annotations

import copy
from typing import List, Dict, Optional

from ...base import Function


class HSEvoPrompt:
    """HSEvo prompt generation for different evolutionary operations.
    
    This class provides methods to generate prompts that match the original
    HSEvo system structure from hsevo/HSEvo, with all prompts embedded directly.
    """
    
    # System prompts
    SYSTEM_GENERATOR = """{scientist_persona} Your task is to design heuristics that can effectively solve optimization problems. Your response outputs Python code and nothing else. Format your code as a Python code string: "```python ... ```"."""

    SYSTEM_REFLECTOR = """You are an expert in the domain of optimization heuristics. Your task is to provide useful advice based on analysis to design better heuristics."""

    SYSTEM_HARMONY_SEARCH = """You are an expert in code review. Your task is to extract all threshold, weight, or hardcoded variables from the function and make them default parameters."""

    # User prompts
    USER_GENERATOR = """{scientist_persona} Your task is to write a `{func_name}` function for {problem_desc}

## Function description:
{func_desc}"""

    SEED_PROMPT = """## Seed function:{seed_func}
## Your task:
Refer to the format of a trivial design above. Be very creative and give new independent `{func_name}`. Output code only and enclose your code with Python code block: ```python ... ```."""

    CROSSOVER_PROMPT = """{user_generator}

### Better code
{code_method1}
### Worse code
{code_method2}
### Analyze & experience
- {analyze}
- {exp}

Your task is to write an improved function `{func_name}` by COMBINING elements of two above heuristics base Analyze & experience.
Output the code within a Python code block: ```python ... ```, has comment and docstring (<50 words) to description key idea of heuristics design.

I'm going to tip $999K for a better heuristics! Let's think step by step."""

    MUTATION_PROMPT = """{user_generator}

Current heuristics:
{elitist_code}

Now, think outside the box write a mutated function `{func_name}` better than current version.
You can use some hints below:
- {reflection}

Output code only and enclose your code with Python code block: ```python ... ```.
I'm going to tip $999K for a better solution!"""

    FLASH_REFLECTION_PROMPT = """### List heuristics
Below is a list of design heuristics ranked from best to worst.
{lst_method}
### Guide
- Keep in mind, list of design heuristics ranked from best to worst. Meaning the first function in the list is the best and the last function in the list is the worst.
- The response in Markdown style and nothing else has the following structure:
"**Analysis:**
**Experience:**"
In there:
+ Meticulously analyze comments, docstrings and source code of several pairs (Better code - Worse code) in List heuristics to fill values for **Analysis:**.
Example: "Comparing (best) vs (worst), we see ...;  (second best) vs (second worst) ...; Comparing (1st) vs (2nd), we see ...; (3rd) vs (4th) ...; Comparing (second worst) vs (worst), we see ...; Overall:"

+ Self-reflect to extract useful experience for design better heuristics and fill to **Experience:** (<60 words).

I'm going to tip $999K for a better heuristics! Let's think step by step."""

    COMPREHENSIVE_REFLECTION_PROMPT = """Your task is to redefine 'Current self-reflection' paying attention to avoid all things in 'Ineffective self-reflection' in order to come up with ideas to design better heuristics.

### Current self-reflection
{curr_reflection}
{good_reflection}

### Ineffective self-reflection
{bad_reflection}

Response (<100 words) should have 4 bullet points: Keywords, Advice, Avoid, Explanation.
I'm going to tip $999K for a better heuristics! Let's think step by step."""

    HARMONY_SEARCH_PROMPT = """[code]
{code_extract}
Extract all threshold, weight, or hardcoded numeric variables as tunable parameters. Requirements:
1. PRESERVE original function logic - only change default parameter values, never modify function body
2. Use reasonable decimal ranges (NO scientific notation like 1e-5, NO infinity values)
3. Parameters must be meaningful for optimization (not array indices or boolean flags)

Response format (two parts):
## Part 1: Output code with threshold, weight, or hardcoded variables as default parameters
```python
# Your modified function here - SAME logic, only parameter defaults changed
```

## Part 2: parameter_ranges dictionary  
```python
parameter_ranges = {{
    'param_name': [min_value, max_value]  # Must have exactly 2 different values
}}
```

I'm going to tip $999K for a better solution!"""

    def get_system_generator_prompt(self, scientist_persona: str = "") -> str:
        """Generate system prompt for code generation.
        
        Args:
            scientist_persona: Persona for diverse generation
            
        Returns:
            Formatted system prompt string
        """
        return self.SYSTEM_GENERATOR.format(scientist_persona=scientist_persona)

    def get_system_reflector_prompt(self) -> str:
        """Generate system prompt for reflection operations.
        
        Returns:
            System reflector prompt string
        """
        return self.SYSTEM_REFLECTOR

    def get_system_harmony_search_prompt(self) -> str:
        """Generate system prompt for harmony search operations.
        
        Returns:
            System harmony search prompt string
        """
        return self.SYSTEM_HARMONY_SEARCH

    def get_user_generator_prompt(self, scientist_persona: str, func_name: str, 
                                 problem_desc: str, func_desc: str) -> str:
        """Generate user prompt for code generation.
        
        Args:
            scientist_persona: Persona for diverse generation
            func_name: Name of the function to generate
            problem_desc: Description of the problem
            func_desc: Description of the function
            
        Returns:
            Formatted user generator prompt string
        """
        return self.USER_GENERATOR.format(
            scientist_persona=scientist_persona,
            func_name=func_name,
            problem_desc=problem_desc,
            func_desc=func_desc
        )

    def get_seed_prompt(self, seed_func: str, func_name: str) -> str:
        """Generate seed prompt.
        
        Args:
            seed_func: The seed function code
            func_name: Name of the function
            
        Returns:
            Formatted seed prompt string
        """
        return self.SEED_PROMPT.format(
            seed_func=seed_func,
            func_name=func_name
        )

    def get_crossover_prompt(self, user_generator: str, code_method1: str, code_method2: str,
                           analyze: str, exp: str, func_name: str) -> str:
        """Generate crossover prompt for combining two parent algorithms.
        
        Args:
            user_generator: User generator prompt
            code_method1: Code for method 1 (better)
            code_method2: Code for method 2 (worse)
            analyze: Analysis from flash reflection
            exp: Experience from comprehensive reflection
            func_name: Name of the function
            
        Returns:
            Formatted crossover prompt
        """
        return self.CROSSOVER_PROMPT.format(
            user_generator=user_generator,
            code_method1=code_method1,
            code_method2=code_method2,
            analyze=analyze,
            exp=exp,
            func_name=func_name
        )

    def get_mutation_prompt(self, user_generator: str, reflection: str, 
                          elitist_code: str, func_name: str) -> str:
        """Generate mutation prompt for modifying the elite individual.
        
        Args:
            user_generator: User generator prompt
            reflection: Comprehensive reflection
            elitist_code: Elite individual code
            func_name: Name of the function
            
        Returns:
            Formatted mutation prompt
        """
        return self.MUTATION_PROMPT.format(
            user_generator=user_generator,
            reflection=reflection,
            elitist_code=elitist_code,
            func_name=func_name
        )

    def get_flash_reflection_prompt(self, problem_desc: str, lst_method: str, 
                                  schema_reflection: Dict[str, str]) -> str:
        """Generate flash reflection prompt for analyzing recent algorithms.
        
        Args:
            problem_desc: Description of the problem
            lst_method: List of methods to analyze
            schema_reflection: Schema for reflection response
            
        Returns:
            Formatted flash reflection prompt
        """
        return self.FLASH_REFLECTION_PROMPT.format(
            problem_desc=problem_desc,
            lst_method=lst_method,
            schema_reflection=schema_reflection
        )

    def get_comprehensive_reflection_prompt(self, curr_reflection: str, 
                                          good_reflection: str, bad_reflection: str) -> str:
        """Generate comprehensive reflection prompt for long-term learning.
        
        Args:
            curr_reflection: Current reflection
            good_reflection: Good experiences
            bad_reflection: Bad experiences
            
        Returns:
            Formatted comprehensive reflection prompt
        """
        return self.COMPREHENSIVE_REFLECTION_PROMPT.format(
            curr_reflection=curr_reflection,
            good_reflection=good_reflection,
            bad_reflection=bad_reflection
        )

    def get_harmony_search_prompt(self, code_extract: str) -> str:
        """Generate harmony search prompt for parameter tuning.
        
        Args:
            code_extract: Code to analyze for parameter tuning
            
        Returns:
            Formatted harmony search prompt
        """
        return self.HARMONY_SEARCH_PROMPT.format(code_extract=code_extract)

    @classmethod
    def create_instruct_prompt(cls, prompt: str, system_prompt: str = None) -> List[Dict]:
        """Create instruction prompt for LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            List of message dictionaries
        """
        if system_prompt is None:
            system_prompt = cls.SYSTEM_GENERATOR
            
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]

    @classmethod
    def get_initial_prompt(cls, task_description: str, template_function: Function, 
                          scientist_persona: str = "", seed_func: str = None) -> str:
        """Generate prompt for initial population creation (legacy method).
        
        Args:
            task_description: Description of the optimization problem
            template_function: Template function to complete
            scientist_persona: Persona for diverse generation
            seed_func: Optional seed function code for reference
            
        Returns:
            Formatted prompt string
        """
        user_gen = cls.USER_GENERATOR.format(
            scientist_persona=scientist_persona,
            func_name=template_function.name,
            problem_desc=task_description,
            func_desc=template_function.docstring or ""
        )
        
        seed_prompt = cls.SEED_PROMPT.format(
            seed_func=seed_func if seed_func else "",
            func_name=template_function.name
        )
        
        return user_gen + "\n\n" + seed_prompt 