from __future__ import annotations

import re
import logging
from typing import Tuple, List, Dict, Optional

from .prompt import HSEvoPrompt
from ...base import LLM, SampleTrimmer, Function, Program


class HSEvoSampler:
    # Constants for extraction configuration
    MIN_FUNCTION_LINES = 5     # Minimum lines to consider for function extraction
    
    # Compiled regex patterns for extracting algorithm thoughts from LLM responses
    _THOUGHT_PATTERNS = [
        # Extract content between <think> and </think> tags
        re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE),
    ]
    
    # Comprehensive parameter extraction patterns for all formats (no scientific notation)
    _PARAM_PATTERNS = [
        # Double quoted parameters with flexible whitespace
        re.compile(r'"(\w+)"\s*:\s*\(\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)'),  # "param": (min, max)
        re.compile(r'"(\w+)"\s*:\s*\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]'),  # "param": [min, max]
        # Single quoted parameters with flexible whitespace 
        re.compile(r"'(\w+)'\s*:\s*\(\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)"),  # 'param': (min, max)
        re.compile(r"'(\w+)'\s*:\s*\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]"),  # 'param': [min, max]
        # Unquoted parameters with flexible whitespace
        re.compile(r'(\w+)\s*:\s*\(\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)'),    # param: (min, max)
        re.compile(r'(\w+)\s*:\s*\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]'),    # param: [min, max]
    ]
    
    # Function and parameter extraction patterns
    # Extract Python code blocks from markdown-style code fences (with and without closing fence)
    _CODE_BLOCK_PATTERNS = [
        # Standard: ```python ... ```
        re.compile(r'```python\s*(.*?)```', re.DOTALL),
        # Missing closing fence: ```python ... (to end or next section)
        re.compile(r'```python\s*(.*?)(?=##|$)', re.DOTALL),
        # Alternative: ```python ... (to parameter_ranges section)
        re.compile(r'```python\s*(.*?)(?=parameter_ranges|$)', re.DOTALL | re.IGNORECASE),
        # Raw function without markdown fencing: def ... (to parameter_ranges or end)
        re.compile(r'(def\s+\w+.*?)(?=parameter_ranges|##|$)', re.DOTALL | re.IGNORECASE),
    ]
    # Extract parameter_ranges dictionary assignments
    _PARAM_RANGES_PATTERN = re.compile(r'parameter_ranges\s*=\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL)

    def __init__(self, llm: LLM, template_program: str | Program):
        """HSEvo sampler for LLM interaction and code generation.
        
        Args:
            llm: LLM instance for generating responses
            template_program: Template program for code generation
        """
        self._llm = llm
        self._template_program = template_program
        self._sample_trimmer = SampleTrimmer(llm)
        self._prompt_generator = HSEvoPrompt()

    def _process_response_to_function(self, response: str) -> Tuple[Optional[str], Optional[Function]]:
        """Process LLM response to extract thought and function.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (thought, function) where either can be None if extraction fails
        """
        logging.debug(f"HSEvoSampler._process_response_to_function: Processing response of length {len(response)}")
        
        # Step 1: Extract thought
        thought = self.__class__.extract_thought_from_response(response)
        logging.debug(f"HSEvoSampler: Thought extraction {'successful' if thought else 'failed'}")
        
        # Step 2: Extract code
        code = SampleTrimmer.trim_preface_of_function(response)
        logging.debug(f"HSEvoSampler: Code trimming result - length: {len(code) if code else 0}")
        
        # Step 3: Convert to function
        try:
            function = SampleTrimmer.sample_to_function(code, self._template_program)
            logging.debug(f"HSEvoSampler: Function conversion {'successful' if function else 'failed'}")
        except Exception as e:
            logging.warning(f"HSEvoSampler: Function conversion failed with exception: {e}")
            function = None
        
        # Step 4: Set function attributes
        if function is not None:
            function.algorithm = thought if thought else ""
            try:
                function.entire_code = str(SampleTrimmer.sample_to_program(code, self._template_program))
                logging.debug(f"HSEvoSampler: Function attributes set successfully")
            except Exception as e:
                logging.warning(f"HSEvoSampler: Failed to set entire_code attribute: {e}")
        
        return thought, function

    def get_thought_and_function(self, prompt: str, scientist_persona: str = "") -> Tuple[str, Function]:
        """Get algorithm thought and function from LLM response.
        
        Args:
            prompt: Input prompt for LLM
            scientist_persona: Scientist persona for system prompt
            
        Returns:
            Tuple of (thought, function) where thought is the algorithm description
            and function is the generated Function object
        """
        logging.debug(f"HSEvoSampler.get_thought_and_function called with persona: '{scientist_persona[:50] if scientist_persona else 'default'}'")
        
        # Create message with proper system prompt
        system_prompt = self._prompt_generator.get_system_generator_prompt(scientist_persona)
        messages = self._prompt_generator.create_instruct_prompt(prompt, system_prompt)
        logging.debug(f"HSEvoSampler: Sending request to LLM with {len(messages)} messages")
        response = self._llm.draw_sample(messages)
        logging.debug(f"HSEvoSampler: Received LLM response of length {len(response)}")
        
        result = self._process_response_to_function(response)
        logging.debug(f"HSEvoSampler: Processing result - thought: {'✓' if result[0] else '✗'}, function: {'✓' if result[1] else '✗'}")
        
        return result

    def get_thought_and_function_with_response(self, prompt: str, scientist_persona: str = "") -> Tuple[str, Function, str]:
        """Get algorithm thought and function from LLM response, also returning raw response.
        
        Args:
            prompt: Input prompt for LLM
            scientist_persona: Scientist persona for system prompt
            
        Returns:
            Tuple of (thought, function, raw_response) where thought is the algorithm description,
            function is the generated Function object, and raw_response is the original LLM response
        """
        logging.debug(f"HSEvoSampler.get_thought_and_function_with_response called with persona: '{scientist_persona[:50] if scientist_persona else 'default'}'")
        
        # Create message with proper system prompt
        system_prompt = self._prompt_generator.get_system_generator_prompt(scientist_persona)
        messages = self._prompt_generator.create_instruct_prompt(prompt, system_prompt)
        
        logging.debug(f"HSEvoSampler: Sending request to LLM with {len(messages)} messages")
        response = self._llm.draw_sample(messages)
        logging.debug(f"HSEvoSampler: Received LLM response of length {len(response)}")
        
        thought, function = self._process_response_to_function(response)
        logging.debug(f"HSEvoSampler: Processing result - thought: {'✓' if thought else '✗'}, function: {'✓' if function else '✗'}")
        
        return thought, function, response

    def get_multiple_functions(self, prompts: List[str], scientist_persona: str = "") -> List[Tuple[str, Function]]:
        """Get multiple functions from multiple prompts.
        
        Args:
            prompts: List of prompts for LLM
            scientist_persona: Scientist persona for system prompt
            
        Returns:
            List of (thought, function) tuples
        """
        # Create messages with proper system prompts
        system_prompt = self._prompt_generator.get_system_generator_prompt(scientist_persona)
        messages_list = [self._prompt_generator.create_instruct_prompt(prompt, system_prompt) for prompt in prompts]
        
        responses = self._llm.draw_samples(messages_list)
        return [self._process_response_to_function(response) for response in responses]

    def get_harmony_search_response(self, prompt: str) -> str:
        """Get raw response for harmony search parameter extraction.
        
        Args:
            prompt: Harmony search prompt
            
        Returns:
            Raw LLM response
        """
        # Create message with proper system prompt
        system_prompt = self._prompt_generator.get_system_harmony_search_prompt()
        messages = self._prompt_generator.create_instruct_prompt(prompt, system_prompt)
        
        return self._llm.draw_sample(messages)

    def get_flash_reflection_response(self, prompt: str) -> str:
        """Get response for flash reflection analysis.
        
        Args:
            prompt: Flash reflection prompt
            
        Returns:
            Raw LLM response
        """
        # Create message with proper system prompt
        system_prompt = self._prompt_generator.get_system_reflector_prompt()
        messages = self._prompt_generator.create_instruct_prompt(prompt, system_prompt)
        
        return self._llm.draw_sample(messages)

    def get_comprehensive_reflection_response(self, prompt: str) -> str:
        """Get response for comprehensive reflection analysis.
        
        Args:
            prompt: Comprehensive reflection prompt
            
        Returns:
            Raw LLM response
        """
        # Create message with proper system prompt
        system_prompt = self._prompt_generator.get_system_reflector_prompt()
        messages = self._prompt_generator.create_instruct_prompt(prompt, system_prompt)
        
        return self._llm.draw_sample(messages)

    @classmethod
    def extract_thought_from_response(cls, response: str) -> Optional[str]:
        """Extract algorithm thought/description from LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            Extracted thought or None if not found
        """
        try:
            # Try to extract from <think> tags first (primary method)
            think_pattern = cls._THOUGHT_PATTERNS[0]
            matches = think_pattern.findall(response)
            if matches:
                thought = matches[0].strip()
                if thought:
                    return thought
            
            # Fallback: Extract text before code blocks or function definitions
            # Look for text before ```python or def keywords
            lines = response.split('\n')
            thought_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                # Stop when we hit code blocks or function definitions
                if (line_stripped.startswith('```') or 
                    line_stripped.startswith('def ') or
                    'def ' in line_stripped):
                    break
                # Skip empty lines at the beginning
                if not thought_lines and not line_stripped:
                    continue
                thought_lines.append(line)
            
            if thought_lines:
                thought = '\n'.join(thought_lines).strip()
                # Only return non-trivial thoughts (more than just whitespace/punctuation)
                if thought and len(thought) > 3 and any(c.isalnum() for c in thought):
                    return thought
            
            return None
        except Exception as e:
            logging.warning(f"Error extracting thought from response: {e}")
            return None

    @classmethod
    def _extract_parameters_from_patterns(cls, response: str) -> Dict[str, Tuple[float, float]]:
        """Extract parameters using compiled regex patterns.
        
        Args:
            response: LLM response string
            
        Returns:
            Dictionary of parameter names to (min, max) tuples
        """
        parameter_ranges = {}
        
        for pattern in cls._PARAM_PATTERNS:
            matches = pattern.findall(response)
            if matches:
                # Process all matches from this pattern
                for match in matches:
                    param_name, min_val, max_val = match
                    converted_params = cls._convert_and_validate_parameter(param_name, min_val, max_val)
                    if converted_params:
                        parameter_ranges[param_name] = converted_params
                        
                # If we found valid parameters, don't try other patterns
                if parameter_ranges:
                    break
                    
        return parameter_ranges
    
    @classmethod
    def _convert_and_validate_parameter(cls, param_name: str, min_val: str, max_val: str) -> Optional[Tuple[float, float]]:
        """Convert string values to float and validate parameter ranges.
        
        Args:
            param_name: Name of the parameter
            min_val: Minimum value as string
            max_val: Maximum value as string
            
        Returns:
            Tuple of (min_float, max_float) if valid, None otherwise
        """
        try:
            min_float = float(min_val)
            max_float = float(max_val)
            
            # Validate parameters according to harmony search requirements
            if cls._validate_parameter_values(param_name, min_float, max_float):
                return (min_float, max_float)
            else:
                return None
                
        except ValueError:
            logging.warning(f"Could not convert parameter values to float: {param_name}={min_val},{max_val}")
            return None

    @classmethod
    def _validate_parameter_values(cls, param_name: str, min_val: float, max_val: float) -> bool:
        """Validate parameter values for harmony search.
        
        Args:
            param_name: Name of the parameter
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            True if values are valid, False otherwise
        """
        # Check for identical values (no optimization range)
        if min_val == max_val:
            logging.warning(f"Parameter '{param_name}' has identical min/max values: ({min_val}, {max_val}) - skipping")
            return False
        
        # Ensure proper range ordering
        if min_val > max_val:
            logging.warning(f"Parameter '{param_name}' has min > max: ({min_val}, {max_val}) - skipping")
            return False
        
        return True

    @classmethod
    def _extract_function_block(cls, response: str) -> str:
        """Extract function block from response using robust extraction strategy.
        
        Args:
            response: LLM response string
            
        Returns:
            Function block string (empty string if none found)
        """
        # Strategy 1: Try multiple code block patterns (handles missing closing fence)
        pattern_names = ["standard_markdown", "missing_closing_fence", "to_parameter_section", "raw_function"]
        for i, pattern in enumerate(cls._CODE_BLOCK_PATTERNS):
            code_blocks = pattern.findall(response)
            for block in code_blocks:
                block_stripped = block.strip()
                # Valid function block: contains 'def' but excludes parameter_ranges definitions
                if 'def ' in block_stripped and 'parameter_ranges' not in block_stripped:
                    logging.debug(f"Function extracted using pattern '{pattern_names[i]}' - length: {len(block_stripped)}")
                    return block_stripped
        
        # Strategy 2: Line-by-line extraction (fallback for no markdown fencing)
        line_by_line_result = cls._extract_function_line_by_line(response)
        if line_by_line_result:
            logging.debug(f"Function extracted using line-by-line fallback - length: {len(line_by_line_result)}")
        else:
            logging.debug("No function found with any extraction method")
        return line_by_line_result
    
    @classmethod
    def _extract_function_line_by_line(cls, response: str) -> str:
        """Extract function definition line by line as fallback method.
        
        Args:
            response: LLM response string
            
        Returns:
            Function block string
        """
        lines = response.split('\n')
        function_lines = []
        in_function = False
        empty_line_count = 0
        
        for line in lines:
            # Start collecting when we see a function definition
            if line.strip().startswith('def '):
                in_function = True
                empty_line_count = 0
            
            if in_function:
                # Stop if we encounter parameter_ranges section or another major section
                if line.strip().lower().startswith('parameter_ranges') or line.strip().startswith('##'):
                    break
                
                function_lines.append(line)
                
                # Track empty lines but don't stop immediately
                if line.strip() == '':
                    empty_line_count += 1
                    # Stop after 2 consecutive empty lines (more robust)
                    if empty_line_count >= 2 and len(function_lines) > cls.MIN_FUNCTION_LINES:
                        break
                else:
                    empty_line_count = 0
        
        # Clean up trailing empty lines
        while function_lines and function_lines[-1].strip() == '':
            function_lines.pop()
        
        return '\n'.join(function_lines) if function_lines else ""

    @classmethod
    def _extract_parameters_from_dict_section(cls, response: str) -> Dict[str, Tuple[float, float]]:
        """Extract parameters from parameter_ranges dictionary section.
        
        Args:
            response: LLM response string
            
        Returns:
            Dictionary of parameter names to (min, max) tuples
        """
        parameter_ranges = {}
        
        if 'parameter_ranges' not in response.lower():
            return parameter_ranges
            
        param_section_match = cls._PARAM_RANGES_PATTERN.search(response)
        if not param_section_match:
            return parameter_ranges
            
        dict_content = param_section_match.group(1)
        
        # Try all parameter patterns (now consolidated)
        for pattern in cls._PARAM_PATTERNS:
            dict_params = pattern.findall(dict_content)
            if dict_params:
                for param_name, min_val, max_val in dict_params:
                    converted_params = cls._convert_and_validate_parameter(param_name, min_val, max_val)
                    if converted_params:
                        parameter_ranges[param_name] = converted_params
                # If we found parameters with this pattern, don't try other patterns
                if parameter_ranges:
                    break
        
        return parameter_ranges

    @classmethod
    def extract_harmony_search_params(cls, response: str) -> Tuple[Dict[str, Tuple[float, float]], str]:
        """Extract harmony search parameters and function block from response with optimized flow.
        
        Args:
            response: LLM response containing parameter ranges and function template
            
        Returns:
            Tuple of (parameter_ranges_dict, function_block_string)
        """
        if not response or not response.strip():
            logging.warning("Empty or whitespace-only response provided for parameter extraction")
            return {}, ""
            
        try:
            # Extract parameter ranges using optimized two-stage approach
            parameter_ranges = cls._extract_parameters_from_patterns(response)
            
            # If no parameters found with patterns, try dictionary section extraction
            if not parameter_ranges:
                parameter_ranges = cls._extract_parameters_from_dict_section(response)
            
            # Extract function block using optimized extraction strategies
            function_block = cls._extract_function_block(response)
            
            # Log extraction summary for debugging
            if not parameter_ranges and not function_block:
                logging.debug("No parameters or function block found in response")
            elif not parameter_ranges:
                logging.debug("Function block found but no parameters extracted")
            elif not function_block:
                logging.debug(f"Parameters found ({list(parameter_ranges.keys())}) but no function block")
            
            return parameter_ranges, function_block
            
        except Exception as e:
            logging.error(f"Error extracting harmony search parameters: {e}")
            return {}, "" 