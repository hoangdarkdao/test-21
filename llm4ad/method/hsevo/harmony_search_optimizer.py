"""
Optimized Harmony Search implementation for HSEvo with clean string extraction,
robust function conversion, and comprehensive logging.

CRITICAL DESIGN NOTE:
Harmony search creates functions with tuned default parameter values in their signatures.
When converting these functions to Program objects, we MUST preserve the original signature
with the tuned parameters, NOT replace it with the template signature. This is handled
by the _harmony_function_to_program method which preserves the function.args attribute.

Example:
Template: def heuristic(items, alpha=0.5, beta=0.3)
Tuned:   def heuristic(items, alpha=0.742, beta=0.891)
         ^-- These tuned values must be preserved in the Program object
"""

import copy
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from ...base import Function, Program, TextFunctionProgramConverter
from .sampler import HSEvoSampler


class HarmonySearchLogger:
    """Specialized logger for harmony search operations."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.iteration_logs = []
        self.parameter_logs = []
        self.function_logs = []
        self.error_logs = []
        self.summary_stats = {
            'total_iterations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'parameter_extraction_success': 0,
            'parameter_extraction_failed': 0,
            'function_creation_success': 0,
            'function_creation_failed': 0,
            'best_score': float('-inf'),
            'best_parameters': {},
            'score_improvements': []
        }
    
    @staticmethod
    def format_score(score: Optional[float]) -> str:
        """Format score for display, handling None and -inf values."""
        if score is None:
            return "N/A"
        elif score == float('-inf'):
            return "-inf"
        elif score == float('inf'):
            return "inf"
        else:
            return f"{score:.6f}"
    
    def log_parameter_extraction(self, response: str, success: bool, 
                                parameters: Dict = None, function_block: str = None):
        """Log parameter extraction results."""
        entry = {
            'timestamp': time.time(),
            'success': success,
            'response_length': len(response),
            'parameters_found': len(parameters) if parameters else 0,
            'function_block_found': bool(function_block),
            'parameters': parameters if success else None
        }
        self.parameter_logs.append(entry)
        
        if success:
            self.summary_stats['parameter_extraction_success'] += 1
            if self.debug_mode:
                print(f"✓ Extracted parameters: {list(parameters.keys())}")
        else:
            self.summary_stats['parameter_extraction_failed'] += 1
            if self.debug_mode:
                # Analyze why extraction failed and provide detailed error info
                failure_reason = self._analyze_extraction_failure(response, parameters, function_block)
                print(f"✗ Parameter extraction failed - {failure_reason}")
                self.error_logs.append({
                    'type': 'parameter_extraction',
                    'timestamp': time.time(),
                    'failure_reason': failure_reason,
                    'response_preview': response[:200] + "..." if len(response) > 200 else response
                })
                
    def _analyze_extraction_failure(self, response: str, parameters: Dict, function_block: str) -> str:
        """Analyze why parameter extraction failed and return detailed reason."""
        issues = []
        
        # Check for single values (same min/max)
        same_value_patterns = [
            r'"?\w+"?\s*:\s*\((\d+(?:\.\d+)?),\s*\1\)',  # (value, value)
            r'"?\w+"?\s*:\s*\[(\d+(?:\.\d+)?),\s*\1\]'   # [value, value]
        ]
        for pattern in same_value_patterns:
            matches = re.findall(pattern, response)
            if matches:
                issues.append(f"identical min/max values detected: {matches[:3]}")
                break
        
        # Check if no parameters found at all
        if not parameters:
            # Look for potential parameter patterns that might be malformed
            potential_params = re.findall(r'\b\w+\s*[:\=]\s*[\[\(][^\]\)]*[\]\)]', response)
            if potential_params:
                issues.append(f"malformed parameter format detected: {potential_params[:3]}")
            else:
                issues.append("no parameter ranges found in response")
        
        # Check if function block missing
        if not function_block:
            if '```python' not in response and 'def ' not in response:
                issues.append("no function definition found")
            else:
                issues.append("function format not recognized")
        
        if not issues:
            issues.append("unknown format issue")
            
        return "Harmony search requires: decimal notation (0.1, 1.0), different min/max values. Issues: " + "; ".join(issues)
    
    def log_function_creation(self, parameters: Dict, harmony: np.ndarray, 
                            success: bool, function_str: str = None):
        """Log function creation results."""
        param_dict = {list(parameters.keys())[i]: harmony[i] for i in range(len(harmony))}
        entry = {
            'timestamp': time.time(),
            'parameters': param_dict,
            'success': success,
            'function_length': len(function_str) if function_str else 0
        }
        self.function_logs.append(entry)
        
        if success:
            self.summary_stats['function_creation_success'] += 1
        else:
            self.summary_stats['function_creation_failed'] += 1
            if self.debug_mode:
                print(f"✗ Function creation failed for parameters: {param_dict}")
    
    def log_iteration(self, iteration: int, parameters: Dict, harmony: np.ndarray, 
                     score: Optional[float], best_score: float, accepted: bool):
        """Log harmony search iteration."""
        param_dict = {list(parameters.keys())[i]: harmony[i] for i in range(len(harmony))}
        entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            'parameters': param_dict,
            'score': score,
            'best_score': best_score,
            'accepted': accepted,
            'successful_evaluation': score is not None
        }
        self.iteration_logs.append(entry)
        
        if score is not None:
            self.summary_stats['successful_evaluations'] += 1
            if score > self.summary_stats['best_score']:
                self.summary_stats['best_score'] = score
                self.summary_stats['best_parameters'] = param_dict.copy()
                self.summary_stats['score_improvements'].append((iteration, score))
                if self.debug_mode:
                    score_str = self.format_score(score)
                    print(f"🎯 New best score: {score_str} at iteration {iteration}")
        else:
            self.summary_stats['failed_evaluations'] += 1
        
        if self.debug_mode:
            status = "✓" if score is not None else "✗"
            score_str = self.format_score(score)
            best_str = self.format_score(best_score)
            print(f"{status} Iter {iteration}: Score={score_str}, Best={best_str}")
    
    def log_summary(self, iterations_completed: int, samples_used: int, 
                   initial_score: float, final_score: float):
        """Log final summary."""
        self.summary_stats['total_iterations'] = iterations_completed
        self.summary_stats['samples_used'] = samples_used
        self.summary_stats['initial_score'] = initial_score
        self.summary_stats['final_score'] = final_score
        self.summary_stats['score_improvement'] = final_score - initial_score
        
        print(f"\n=== Harmony Search Summary ===")
        final_str = self.format_score(final_score)
        print(f"Best score of HS: {final_str}")
        print(f"Best parameters: {self.summary_stats['best_parameters']}")
        
        # Show parameter tuning scores (only successful evaluations)
        if self.iteration_logs:
            print("Parameter tuning scores:")
            successful_iterations = [log for log in self.iteration_logs if log['score'] is not None]
            if successful_iterations:
                # Show up to 5 best scores to keep it concise but meaningful
                sorted_iterations = sorted(successful_iterations, key=lambda x: x['score'], reverse=True)
                for i, log in enumerate(sorted_iterations[:5]):
                    score_str = self.format_score(log['score'])
                    params_str = ", ".join([f"{k}={v:.3f}" for k, v in log['parameters'].items()])
                    print(f"  {i+1}. Score: {score_str} | {params_str}")
                if len(successful_iterations) > 5:
                    print(f"  ... and {len(successful_iterations)-5} more parameter combinations")
        
        print(f"==============================\n")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return self.summary_stats.copy()


class FunctionCreator:
    """Clean function creation with parameter substitution.
    
    CRITICAL: This class creates functions with specific default parameter values
    that have been tuned by harmony search. These signatures MUST be preserved
    when converting to Program objects, as they contain the optimization results.
    """
    
    @staticmethod
    def create_function_with_parameters(func_template: str, parameter_ranges: Dict, 
                                      harmony: np.ndarray, debug_mode: bool = False) -> Optional[str]:
        """Create function string with specific parameter values.
        
        This method substitutes the tuned parameter values into the function template,
        creating a function with specific default values. These default values are
        the result of harmony search optimization and must be preserved.
        
        Args:
            func_template: Function template string
            parameter_ranges: Dictionary of parameter ranges
            harmony: Array of parameter values (tuned by harmony search)
            
        Returns:
            Function string with substituted parameters or None if failed
        """
        try:
            func_str = func_template
            param_names = list(parameter_ranges.keys())
            
            # Method 1: Replace placeholders {param_name}
            substitutions_made = False
            for i, param_name in enumerate(param_names):
                placeholder = '{' + param_name + '}'
                value = harmony[i]
                if placeholder in func_str:
                    func_str = func_str.replace(placeholder, f"{value:.6f}")
                    substitutions_made = True
            
            # Method 2: Replace default values in function signature if no placeholders found
            if not substitutions_made:
                original_func_str = func_str
                func_str = FunctionCreator._replace_signature_defaults(
                    func_str, param_names, harmony, debug_mode
                )
                # Check if replacement actually happened
                if func_str != original_func_str:
                    substitutions_made = True
            
            # Method 3: If no parameters found in signature, append them to signature
            if not substitutions_made and not FunctionCreator._has_parameters_in_signature(func_str, param_names):
                func_str = FunctionCreator._append_parameters_to_signature(
                    func_str, param_names, harmony
                )
            
            return func_str
            
        except Exception as e:
            print(f"Error in create_function_with_parameters: {e}")
            return None
    
    @staticmethod
    def _has_parameters_in_signature(func_str: str, param_names: List[str]) -> bool:
        """Check if parameters exist in function signature."""
        for param_name in param_names:
            if param_name in func_str:
                return True
        return False
    
    @staticmethod
    def _replace_signature_defaults(func_str: str, param_names: List[str], 
                                   harmony: np.ndarray, debug_mode: bool = False) -> str:
        """Replace default values in function signature with improved pattern matching."""
        # Extract only the function signature to avoid replacing parameters in function body
        func_def_pattern = r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:)'
        func_def_match = re.search(func_def_pattern, func_str)
        
        if not func_def_match:
            if debug_mode:
                print(f"      ❌ Could not find function signature")
            return func_str
        
        original_signature = func_def_match.group(1)
        modified_signature = original_signature
        
        for i, param_name in enumerate(param_names):
            value = harmony[i]
            # More specific patterns to handle typed parameters
            patterns = [
                # Pattern 1: param_name: type = value (with comma or closing paren)
                rf'(\b{param_name}\s*:\s*[^=]*=\s*)[^,)]+([,)])',
                # Pattern 2: param_name = value (no type annotation)
                rf'(\b{param_name}\s*=\s*)[^,)]+([,)])',
                # Pattern 3: param_name: type = value (at end of parameters)
                rf'(\b{param_name}\s*:\s*[^=]*=\s*)[^,)]+(\s*\))',
                # Pattern 4: param_name = value (at end of parameters)
                rf'(\b{param_name}\s*=\s*)[^,)]+(\s*\))',
            ]
            
            replacement = rf'\g<1>{value:.6f}\g<2>'
            matched = False
            for j, pattern in enumerate(patterns):
                if re.search(pattern, modified_signature):
                    modified_signature = re.sub(pattern, replacement, modified_signature)
                    matched = True
                    break
            
            if not matched and debug_mode:
                print(f"      ❌ No pattern matched for {param_name}")
        
        # Replace only the signature part in the original function string
        func_str = func_str.replace(original_signature, modified_signature)
        return func_str
    
    @staticmethod
    def _append_parameters_to_signature(func_str: str, param_names: List[str], 
                                       harmony: np.ndarray) -> str:
        """Append parameters to function signature if they don't exist."""
        try:
            # Find function definition line
            func_def_pattern = r'(def\s+\w+\s*\([^)]*)\)(\s*:.*)'
            match = re.search(func_def_pattern, func_str, re.DOTALL)
            
            if match:
                before_closing_paren = match.group(1)
                after_closing_paren = match.group(2)
                
                # Add parameters
                param_strs = []
                for i, param_name in enumerate(param_names):
                    value = harmony[i]
                    param_strs.append(f"{param_name}={value:.6f}")
                
                # Check if there are existing parameters
                if before_closing_paren.strip().endswith('('):
                    # No existing parameters
                    new_params = ', '.join(param_strs)
                else:
                    # Existing parameters, add comma
                    new_params = ', ' + ', '.join(param_strs)
                
                # Reconstruct function definition
                new_func_def = before_closing_paren + new_params + ')' + after_closing_paren
                func_str = func_str.replace(match.group(0), new_func_def)
        
        except Exception as e:
            print(f"Error appending parameters to signature: {e}")
        
        return func_str


class OptimizedHarmonySearch:
    """Optimized harmony search with clean implementation."""
    
    def __init__(self, hm_size: int = 5, hmcr: float = 0.9, par: float = 0.3,
                 bandwidth: float = 0.1, max_iter_hs: int = 10, debug_mode: bool = False):
        self.hm_size = hm_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.max_iter_hs = max_iter_hs
        self.debug_mode = debug_mode
        self.logger = HarmonySearchLogger(debug_mode)
    
    def optimize_function(self, candidate_function: Function, llm_response: str,
                         evaluator, template_program: Program, 
                         total_samples: int, max_samples: int) -> Tuple[Optional[Function], int]:
        """Optimize function using harmony search.
        
        Args:
            candidate_function: Function to optimize
            llm_response: LLM response with parameters and function template
            evaluator: Function evaluator
            template_program: Template program for conversion
            total_samples: Current total samples used
            max_samples: Maximum samples allowed
            
        Returns:
            Tuple of (optimized_function, samples_used)
        """
        # Extract parameters and function template using improved sampler extraction
        parameters, func_template = HSEvoSampler.extract_harmony_search_params(llm_response)
        
        self.logger.log_parameter_extraction(
            llm_response, bool(parameters and func_template), parameters, func_template
        )
        
        if not parameters or not func_template:
            return None, 0
        
        # Initialize harmony search
        bounds = list(parameters.values())
        harmony_memory = self._initialize_harmony_memory(bounds)
        
        best_harmony = None
        best_score = float('-inf')
        best_program = None  # Store the best program with preserved signature
        samples_used = 0
        
        print(f"Harmony Search: Optimizing {len(parameters)} parameters: {list(parameters.keys())}")
        
        # Main optimization loop
        for iteration in range(self.max_iter_hs):
            if total_samples + samples_used >= max_samples:
                break
            
            # Create new harmony
            new_harmony = self._create_new_harmony(harmony_memory, bounds)
            
            # Evaluate harmony (returns score and stores program internally)
            score, program = self._evaluate_harmony_with_program(
                new_harmony, parameters, func_template, evaluator, template_program
            )
            
            if score is not None:
                samples_used += 1
            
            # Update best harmony and program
            if score is not None and score > best_score:
                best_score = score
                best_harmony = new_harmony.copy()
                best_program = program  # Store the program with preserved signature
            
            # Update harmony memory
            accepted = self._update_harmony_memory(harmony_memory, new_harmony, score, bounds)
            
            # Log iteration
            self.logger.log_iteration(iteration + 1, parameters, new_harmony, 
                                    score, best_score, accepted)
        
        # Create best function from the best program
        best_function = None
        if best_program is not None and best_harmony is not None:
            best_function = self._create_optimized_function_from_program(
                best_program, best_harmony, parameters, candidate_function, best_score
            )
        
        # Log summary
        self.logger.log_summary(
            iteration + 1, samples_used, 
            candidate_function.score if candidate_function.score else 0, 
            best_score
        )
        
        return best_function, samples_used
    
    def _initialize_harmony_memory(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Initialize harmony memory with random values."""
        harmony_memory = np.zeros((self.hm_size, len(bounds)))
        for i, (lower, upper) in enumerate(bounds):
            harmony_memory[:, i] = np.random.uniform(lower, upper, self.hm_size)
        return harmony_memory
    
    def _create_new_harmony(self, harmony_memory: np.ndarray, 
                          bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Create new harmony using HMCR and PAR."""
        new_harmony = np.zeros(len(bounds))
        
        for i in range(len(bounds)):
            if np.random.rand() < self.hmcr:
                # Select from harmony memory
                new_harmony[i] = harmony_memory[np.random.randint(self.hm_size), i]
                
                # Apply pitch adjustment
                if np.random.rand() < self.par:
                    adjustment = (np.random.uniform(-1, 1) * 
                                (bounds[i][1] - bounds[i][0]) * self.bandwidth)
                    new_harmony[i] += adjustment
                    # Ensure bounds
                    new_harmony[i] = np.clip(new_harmony[i], bounds[i][0], bounds[i][1])
            else:
                # Random selection
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        
        return new_harmony
    
    def _evaluate_harmony_with_program(self, harmony: np.ndarray, parameters: Dict, 
                                     func_template: str, evaluator, template_program: Program) -> Tuple[Optional[float], Optional[Program]]:
        """Evaluate a harmony and return both score and program with preserved signature."""
        try:
            if self.debug_mode:
                param_dict = {list(parameters.keys())[i]: harmony[i] for i in range(len(harmony))}
                print(f"🎵 Evaluating harmony: {param_dict}")
            
            # Create function with parameters
            func_str = FunctionCreator.create_function_with_parameters(
                func_template, parameters, harmony, False  # Disable verbose logging
            )
            
            if not func_str:
                self.logger.log_function_creation(parameters, harmony, False)
                if self.debug_mode:
                    print(f"   ❌ Failed to create function")
                return None, None
            
            self.logger.log_function_creation(parameters, harmony, True, func_str)
            
            if self.debug_mode:
                print(f"🔧 Generated function string:\n{func_str}\n")
                
                # Check if the generated function makes logical sense for parameter tuning
                param_names = list(parameters.keys())
                logic_lines = [line.strip() for line in func_str.split('\n') 
                              if line.strip() and not line.strip().startswith('#') 
                              and not line.strip().startswith('"""') and not line.strip().startswith('Args:')
                              and not line.strip().startswith('Returns:') and 'def ' not in line]
                
                meaningful_usage = False
                for line in logic_lines:
                    # Skip variable assignments and return statements that just return the parameter
                    if 'return' in line and any(f"return {param}" in line for param in param_names):
                        continue
                    # Check if parameters are used in computations
                    for param in param_names:
                        if param in line:
                            meaningful_usage = True
                            break
                
                if not meaningful_usage and len(logic_lines) > 1:
                    print(f"   ⚠️  WARNING: Function may not meaningfully use tuned parameters!")
                    print(f"   📝 Logic lines: {logic_lines[:3]}")  # Show first few logic lines
            
            # Convert to function and evaluate
            function = TextFunctionProgramConverter.text_to_function(func_str)
            if not function:
                if self.debug_mode:
                    print(f"   ❌ Failed to convert to Function object")
                    print(f"   📋 Function string that failed to convert:\n{func_str}")
                return None, None
            
            # CRITICAL: Use harmony-specific function_to_program that preserves signature
            program = self._harmony_function_to_program(function, template_program)
            if not program:
                if self.debug_mode:
                    print(f"   ❌ Failed to convert to Program")
                return None, None
            
            # Temporarily disable evaluator debug mode to reduce verbose output
            original_debug_mode = evaluator._debug_mode
            evaluator._debug_mode = False
            
            # Extract parameter values to pass to evaluator
            param_kwargs = {list(parameters.keys())[i]: harmony[i] for i in range(len(harmony))}
            
            if self.debug_mode:
                print(f"   🔧 Evaluating with parameters: {param_kwargs}")
                
                # Test if function produces different outputs with different parameter values
                try:
                    # Create test data
                    import numpy as np
                    test_item = 1.0
                    test_bins = np.array([10.0, 8.0, 5.0])
                    
                    # Get the function from the program
                    if hasattr(program, 'functions') and program.functions:
                        func_str = str(program)
                        # Execute the function with the tuned parameters
                        globals_dict = {'np': np}
                        exec(func_str, globals_dict)
                        func_name = program.functions[0].name
                        test_func = globals_dict[func_name]
                        
                        # Test with tuned parameters
                        result1 = test_func(test_item, test_bins, **param_kwargs)
                        
                        # Test with different parameter values (if we have multiple parameters)
                        if len(param_kwargs) > 1:
                            modified_kwargs = param_kwargs.copy()
                            first_param = list(param_kwargs.keys())[0]
                            modified_kwargs[first_param] *= 2  # Double the first parameter
                            result2 = test_func(test_item, test_bins, **modified_kwargs)
                            
                            # Check if results are different
                            if np.array_equal(result1, result2):
                                print(f"   ⚠️  CRITICAL: Function output doesn't change with different parameters!")
                                print(f"   📊 Result1: {result1}")
                                print(f"   📊 Result2: {result2}")
                            else:
                                print(f"   ✅ Function output changes with different parameters")
                except Exception as e:
                    print(f"   🔍 Parameter sensitivity test failed: {e}")
            
            score, _ = evaluator.evaluate_program_record_time(program, **param_kwargs)
            
            # Restore original debug mode
            evaluator._debug_mode = original_debug_mode
            
            if self.debug_mode:
                print(f"   🎯 Score: {score}")
                
                # Check if the function is actually using the parameters
                if hasattr(program, 'functions') and program.functions:
                    func_str = str(program.functions[0])
                    uses_params = any(param in func_str for param in param_kwargs.keys())
                    if not uses_params:
                        print(f"   ⚠️  WARNING: Function may not be using tuned parameters!")
                    else:
                        # Check if parameters are actually affecting the computation
                        param_usage = []
                        for param in param_kwargs.keys():
                            if f"{param} " in func_str or f"{param})" in func_str or f"{param}," in func_str:
                                param_usage.append(f"{param}=✓")
                            else:
                                param_usage.append(f"{param}=✗")
                        print(f"   📊 Parameter usage: {', '.join(param_usage)}")
            
            return score, program
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Harmony evaluation failed: {e}")
            return None, None

    def _harmony_function_to_program(self, function: Function, template_program: Program) -> Program:
        """Convert function to program while preserving the function signature for harmony search.
        
        This is CRITICAL for harmony search because we modify the function signature to include
        specific parameter values that were tuned. We need to ensure the Program object contains
        the actual function text with tuned parameters, not just the template.
        """
        try:
            # Create a new program that preserves the exact function signature and body
            program = copy.deepcopy(template_program)
            
            if not program or not program.functions:
                return None
            
            # Replace the template function with the harmony-tuned function
            program.functions[0] = function
            
            # Ensure the function has the required attributes
            if not hasattr(function, 'name') or not function.name:
                function.name = template_program.functions[0].name
            
            if not hasattr(function, 'return_type') or not function.return_type:
                function.return_type = template_program.functions[0].return_type
            
            return program
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Program conversion failed: {e}")
            return None

    def _update_harmony_memory(self, harmony_memory: np.ndarray, new_harmony: np.ndarray,
                             score: Optional[float], bounds: List[Tuple[float, float]]) -> bool:
        """Update harmony memory if new harmony is better."""
        if score is None:
            return False
        
        # Find worst harmony in memory (simplified)
        worst_idx = np.random.randint(self.hm_size)  # Simplified selection
        harmony_memory[worst_idx] = new_harmony
        return True
    
    def _create_optimized_function_from_program(self, best_program: Program, best_harmony: np.ndarray,
                                               parameters: Dict, original_function: Function,
                                               best_score: float) -> Optional[Function]:
        """Create optimized function from the best program."""
        try:
            if self.debug_mode:
                param_dict = {list(parameters.keys())[i]: best_harmony[i] for i in range(len(best_harmony))}
                print(f"🏆 Best: {param_dict}, Score: {best_score}")
            
            # Extract function string from the best program
            function_str = str(best_program)
            
            if not function_str:
                return None
            
            function = TextFunctionProgramConverter.text_to_function(function_str)
            if not function:
                return None
            
            # Set function attributes including tuned parameters info
            function.score = best_score
            function.algorithm = f"Harmony Search optimized: {original_function.algorithm if original_function.algorithm else 'base'}"
            
            # Add information about the tuned parameters for debugging/logging
            param_dict = {list(parameters.keys())[i]: best_harmony[i] for i in range(len(best_harmony))}
            function.tuned_parameters = param_dict
            
            return function
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Failed to create optimized function: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get harmony search statistics."""
        return self.logger.get_statistics() 