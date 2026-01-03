from __future__ import annotations

import concurrent.futures
import copy
import random
import re
import time
import traceback
from threading import Thread
from typing import Optional, Literal, List, Dict, Tuple
import numpy as np

from .population import HSEvoPopulation
from .profiler import HSEvoProfiler
from .prompt import HSEvoPrompt
from .sampler import HSEvoSampler
from .resume import HSEvoResume
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


class HSEvo:
    """Hybrid Search Evolution (HSEvo) algorithm for automated heuristic design.
    
    HSEvo combines evolutionary algorithms with harmony search and LLM-based reflection
    to evolve high-performing heuristic algorithms. It uses three main operators:
    - Crossover: 2-parent crossover operation for combining successful algorithms
    - Mutation: 1-parent mutation operation for exploring algorithm variations
    - Harmony search: Parameter tuning operation for fine-tuning algorithms
    
    These operators can be enabled/disabled via configuration parameters.
    """
    
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: int = 10,
                 init_pop_size: int = 10,
                 selection_num: int = 2,
                 mutation_rate: float = 0.8,
                 elite_individuals: int = 1,
                 # Harmony search parameters
                 hm_size: int = 5,
                 hmcr: float = 0.9,
                 par: float = 0.3,
                 bandwidth: float = 0.1,
                 max_iter_hs: int = 5,
                # Evolution parameters
                use_crossover: bool = True,
                use_mutation: bool = True,
                use_harmony_search: bool = True,
                 # Execution parameters
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 temperature: float = 0.7,
                 timeout: int = 60,
                 # Reflection parameters
                 flash_reflection: bool = True,
                 comprehensive_reflection: bool = True,
                 # Seed function parameters
                 seed_func: Optional[str] = None,
                 eval_seed_func: bool = False,
                 *,
                 resume_mode: bool = False,
                 resume_log_dir: Optional[str] = None,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Initialize HSEvo algorithm.
        
        Args:
            llm: LLM instance for code generation
            evaluation: Evaluation instance for fitness calculation
            profiler: Profiler for logging (optional)
            max_generations: Maximum number of generations
            max_sample_nums: Maximum number of function evaluations
            pop_size: Population size
            init_pop_size: Initial population size
            selection_num: Number of individuals for selection
            mutation_rate: Mutation rate
            elite_individuals: Number of elite individuals to preserve in selection
            hm_size: Harmony memory size
            hmcr: Harmony memory consideration rate
            par: Pitch adjustment rate
            bandwidth: Bandwidth for pitch adjustment
            max_iter_hs: Maximum harmony search iterations
            use_crossover: Whether to use crossover operator
            use_mutation: Whether to use mutation operator
            use_harmony_search: Whether to use harmony search operator
            num_samplers: Number of parallel samplers
            num_evaluators: Number of parallel evaluators
            temperature: LLM temperature
            timeout: Evaluation timeout
            flash_reflection: Whether to use flash reflection
            comprehensive_reflection: Whether to use comprehensive reflection
            seed_func: Optional seed function code for reference
            eval_seed_func: Whether to evaluate seed function and add to population if successful
            resume_mode: Whether to resume from previous experiment
            resume_log_dir: Directory to resume from
            debug_mode: Whether to enable debug mode
            multi_thread_or_process_eval: Evaluation execution mode
            **kwargs: Additional arguments for SecureEvaluator
        """
        # Store configuration
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._init_pop_size = init_pop_size
        self._selection_num = selection_num
        self._mutation_rate = mutation_rate
        self._elite_individuals = elite_individuals
        
        # Harmony search parameters
        self._hm_size = hm_size
        self._hmcr = hmcr
        self._par = par
        self._bandwidth = bandwidth
        self._max_iter_hs = max_iter_hs
        
        # Evolution parameters
        self._use_crossover = use_crossover
        self._use_mutation = use_mutation
        self._use_harmony_search = use_harmony_search
        
        # Execution parameters
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._temperature = temperature
        self._timeout = timeout
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        
        # Reflection parameters
        self._flash_reflection_enabled = flash_reflection
        self._comprehensive_reflection_enabled = comprehensive_reflection
        
        # Seed function parameters
        self._seed_func = seed_func
        self._eval_seed_func = eval_seed_func
        
        # Resume parameters
        self._resume_mode = resume_mode
        self._resume_log_dir = resume_log_dir
        
        # Function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        
        # Initialize components
        self._population = HSEvoPopulation(pop_size=self._pop_size)
        llm.debug_mode = debug_mode
        self._sampler = HSEvoSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, timeout=timeout, **kwargs)
        self._profiler = profiler
        self._prompt_generator = HSEvoPrompt()
        
        # Statistics
        self._tot_sample_nums = 0
        self._generation = 1
        
        # Reflection state
        self._good_experiences: List[str] = []
        self._bad_experiences: List[str] = []
        self._comprehensive_memory = ""
        self._flash_memory = {"analyze": "", "exp": ""}
        
        # Scientist personas for diverse generation
        self._scientist_personas = [
            "You are an expert in the domain of optimization heuristics.",
            "You are Albert Einstein, relativity theory developer.",
            "You are Isaac Newton, the father of physics.",
            "You are Marie Curie, pioneer in radioactivity.",
            "You are Nikola Tesla, master of electricity.",
            "You are Galileo Galilei, champion of heliocentrism.",
            "You are Stephen Hawking, black hole theorist.",
            "You are Richard Feynman, quantum mechanics genius.",
            "You are Rosalind Franklin, DNA structure revealer.",
            "You are Ada Lovelace, computer programming pioneer."
        ]
        
        # Multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )
        
        # Resume functionality
        if self._resume_mode and self._resume_log_dir:
            self._resume_handler = HSEvoResume(self._resume_log_dir)
            if self._resume_handler.can_resume():
                self._restore_from_checkpoint()
        
        # Pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _restore_from_checkpoint(self):
        """Restore experiment state from checkpoint."""
        if not hasattr(self, '_resume_handler'):
            return
            
        print("Restoring HSEvo experiment from checkpoint...")
        
        # Restore population
        restored_samples = self._resume_handler.restore_population(
            self._population, self._template_program_str
        )
        self._tot_sample_nums = restored_samples
        
        # Restore reflections
        good_exp, bad_exp, comp_memory = self._resume_handler.restore_reflections()
        self._good_experiences = good_exp
        self._bad_experiences = bad_exp
        if comp_memory:
            self._comprehensive_memory = comp_memory
        
        # Restore statistics
        stats = self._resume_handler.restore_experiment_statistics()
        self._generation = stats['generations'] if stats['generations'] > 0 else 1
        
        print(f"Restored {restored_samples} samples, {self._generation} generations")
        print(f"Population size: {len(self._population)}")

    def _sample_evaluate_register(self, prompt: str, operation_type: str = "unknown", scientist_persona: str = "", accumulative: bool = False) -> Function:
        print(f"Check prompt:\n{prompt}")
        if self._debug_mode:
            print(f"\n=== HSEvo {operation_type.upper()} PROMPTS ===")
            system_prompt = self._sampler._prompt_generator.get_system_generator_prompt(scientist_persona)
            print(f"System: {system_prompt}")
            print(f"User: {prompt}")
            print("=== END PROMPTS ===\n")
        
        # Step 1: Sampling
        sample_start = time.time()
        raw_response = None
        try:
            # Try new method that returns raw response for debugging
            if hasattr(self._sampler, 'get_thought_and_function_with_response'):
                thought, func, raw_response = self._sampler.get_thought_and_function_with_response(prompt, scientist_persona)
            else:
                # Fallback to original method
                thought, func = self._sampler.get_thought_and_function(prompt, scientist_persona)
            sample_time = time.time() - sample_start
        except Exception as e:
            print(f"❌ {operation_type}: Sampling failed - {e}")
            if self._debug_mode:
                traceback.print_exc()
            return None
        
        # Step 2: Validation
        if func is None:
            print(f"❌ {operation_type}: No function extracted")
            if raw_response is not None:
                print(f"   Raw LLM response ({len(raw_response)} chars):")
                print(f"   {'-'*50}")
                print(f"   {raw_response}")
                print(f"   {'-'*50}")
            return None
        
        # Step 3: Program conversion and evaluation
        try:
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
            if program is None:
                print(f"❌ {operation_type}: Function to program conversion failed")
                return None
            
            eval_future = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program
            )
            score, eval_time = eval_future.result()
        except Exception as e:
            print(f"❌ {operation_type}: Evaluation failed - {e}")
            if self._debug_mode:
                traceback.print_exc()
            return None
        
        # Step 4: Set attributes and register
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought if thought else ""
        func.sample_time = sample_time
        
        try:
            if self._profiler is not None:
                self._profiler.register_function(func)
                if hasattr(self._profiler, 'register_population'):
                    self._profiler.register_population(self._population)
            
            self._tot_sample_nums += 1
            if accumulative:
                self._population.add_individual_temporary(func)
            else:
                self._population.register_function(func)
        except Exception as e:
            print(f"⚠️  {operation_type}: Registration failed - {e}")
            if self._debug_mode:
                traceback.print_exc()
            return None
        
        # Step 8: Final summary
        thought_preview = thought[:50] + ('...' if len(thought) > 50 else '') if thought else 'None'
        print(f"✅ {operation_type}: Complete! Score={score}, Thought='{thought_preview}'")
        
        return func

    def _init_population(self):
        """Initialize population with diverse algorithms."""
        print("Initializing HSEvo population...")
        
        # Evaluate and add seed function to population if enabled
        self._evaluate_seed_function()
        
        # Generate initial individuals with different personas
        initial_prompts = []
        for i in range(self._init_pop_size):
            persona = self._scientist_personas[i % len(self._scientist_personas)]
            prompt = self._prompt_generator.get_initial_prompt(
                self._task_description_str, 
                self._function_to_evolve, 
                persona,
                self._seed_func
            )
            initial_prompts.append(prompt)
        
        # Generate functions in parallel
        for i, prompt in enumerate(initial_prompts):
            if self._tot_sample_nums >= self._max_sample_nums:
                break
            persona = self._scientist_personas[i % len(self._scientist_personas)]
            self._sample_evaluate_register(prompt, f"initialization_{i+1}", persona)
        
        print(f"Initialized population with {len(self._population)} individuals")
        
        # Log population statistics
        if self._profiler and hasattr(self._profiler, 'register_population'):
            self._profiler.register_population(self._population)

    def _evaluate_seed_function(self):
        """Evaluate seed function and add it to population if successful."""
        if not self._eval_seed_func or not self._seed_func:
            return
            
        print("Evaluating seed function...")
        
        try:
            # Convert seed function string to Function object
            seed_function = TextFunctionProgramConverter.text_to_function(self._seed_func)
            
            if seed_function is None:
                print("❌ Seed function evaluation: Failed to parse seed function")
                return
            
            # Convert to program and evaluate
            program = TextFunctionProgramConverter.function_to_program(seed_function, self._template_program)
            
            if program is None:
                print("❌ Seed function evaluation: Function to program conversion failed")
                return
            
            # Evaluate the seed function
            eval_future = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program
            )
            score, eval_time = eval_future.result()
            
            # If evaluation is successful (score is not None), add to population
            if score is not None:
                seed_function.score = score
                seed_function.evaluate_time = eval_time
                seed_function.algorithm = "Seed function"
                seed_function.sample_time = 0.0  # No sampling time for seed function
                
                # Register to population
                self._population.register_function(seed_function)
                self._tot_sample_nums += 1
                
                # Register with profiler
                if self._profiler is not None:
                    self._profiler.register_function(seed_function)
                    if hasattr(self._profiler, 'register_population'):
                        self._profiler.register_population(self._population)
                
                print(f"✅ Seed function evaluation: Success! Score={score}")
                return True
            else:
                print("❌ Seed function evaluation: Failed - evaluation returned None score")
                return False
                
        except Exception as e:
            print(f"❌ Seed function evaluation: Failed with exception - {e}")
            if self._debug_mode:
                traceback.print_exc()
            return False

    def _flash_reflection(self) -> Dict[str, str]:
        """Perform flash reflection on recent algorithms."""
        if not self._flash_reflection_enabled:
            return {"analyze": "", "exp": ""}
        
        # Get diverse sample for analysis
        sample_algorithms = self._population.get_diversity_sample(5)
        
        if len(sample_algorithms) < 2:
            return {"analyze": "", "exp": ""}
        
        try:
            # Generate flash reflection
            # Format algorithms for analysis
            algorithms_text = ""
            for i, alg in enumerate(sample_algorithms):
                suffix = "th" if 11 <= i + 1 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get((i + 1) % 10, "th")
                algorithms_text += f"[Heuristics {i + 1}{suffix}]\n"
                algorithms_text += f"{str(alg)}\n\n"
            
            prompt = self._prompt_generator.get_flash_reflection_prompt(
                self._task_description_str, 
                algorithms_text,
                {"analyze": "str", "exp": "str"}
            )
            
            if self._debug_mode:
                print(f"\n=== HSEvo FLASH REFLECTION SYSTEM PROMPT ===")
                system_prompt = self._sampler._prompt_generator.get_system_reflector_prompt()
                print(system_prompt)
                print(f"=== END FLASH REFLECTION SYSTEM PROMPT ===\n")
                
                print(f"\n=== HSEvo FLASH REFLECTION USER PROMPT ===")
                print(prompt)
                print(f"=== END FLASH REFLECTION USER PROMPT ===\n")
            
            response = self._sampler.get_flash_reflection_response(prompt)
            
            # Parse response
            analysis, experience = self._parse_flash_reflection(response)
            
            result = {"analyze": analysis, "exp": experience}
            self._flash_memory = result
            
            # Log reflection
            if self._profiler and hasattr(self._profiler, 'log_reflection'):
                self._profiler.log_reflection('flash', response, True)
            
            if self._debug_mode:
                print(f"Flash reflection - Analysis: {analysis[:100]}...")
                print(f"Flash reflection - Experience: {experience[:100]}...")
            
            return result
            
        except Exception as e:
            if self._debug_mode:
                print(f"Flash reflection failed: {e}")
            return {"analyze": "", "exp": ""}

    def _comprehensive_reflection(self) -> str:
        """Perform comprehensive reflection for long-term learning."""
        if not self._comprehensive_reflection_enabled:
            return self._comprehensive_memory
        
        try:
            # Generate comprehensive reflection
            good_reflection = '\n\n'.join(self._good_experiences) if self._good_experiences else "None"
            bad_reflection = '\n\n'.join(self._bad_experiences) if self._bad_experiences else "None"
            current_reflection = self._flash_memory.get("exp", "")
            
            prompt = self._prompt_generator.get_comprehensive_reflection_prompt(
                current_reflection,
                good_reflection,
                bad_reflection
            )
            
            if self._debug_mode:
                print(f"\n=== HSEvo COMPREHENSIVE REFLECTION SYSTEM PROMPT ===")
                system_prompt = self._sampler._prompt_generator.get_system_reflector_prompt()
                print(system_prompt)
                print(f"=== END COMPREHENSIVE REFLECTION SYSTEM PROMPT ===\n")
                
                print(f"\n=== HSEvo COMPREHENSIVE REFLECTION USER PROMPT ===")
                print(prompt)
                print(f"=== END COMPREHENSIVE REFLECTION USER PROMPT ===\n")
            
            response = self._sampler.get_comprehensive_reflection_response(prompt)
            
            self._comprehensive_memory = response
            
            # Log reflection
            if self._profiler and hasattr(self._profiler, 'log_reflection'):
                self._profiler.log_reflection('comprehensive', response, True)
            
            if self._debug_mode:
                print(f"Comprehensive reflection: {response[:100]}...")
            
            return response
            
        except Exception as e:
            if self._debug_mode:
                print(f"Comprehensive reflection failed: {e}")
            return self._comprehensive_memory

    def _crossover(self) -> List[Function]:
        """Perform crossover operations."""
        if not self._use_crossover:
            return []
        
        parent_pairs = self._population.select_for_crossover(self._selection_num)
        if not parent_pairs:
            return []
        
        offspring = []
        
        for parent1, parent2 in parent_pairs:
            if self._tot_sample_nums >= self._max_sample_nums:
                break
            
            try:
                # Generate crossover prompt
                # Create user generator prompt
                persona = self._scientist_personas[0]
                user_generator = f"{persona} Your task is to write a {self._function_to_evolve_name} function for {self._task_description_str}"
                
                # Function signatures not needed for crossover prompt
                
                # Get clean code
                code_method1 = str(parent1) if parent1 else ""
                code_method2 = str(parent2) if parent2 else ""
                
                prompt = self._prompt_generator.get_crossover_prompt(
                    user_generator,
                    code_method1,
                    code_method2,
                    self._flash_memory.get("analyze", ""),
                    self._comprehensive_memory,
                    self._function_to_evolve_name
                )
                
                # Generate offspring
                child = self._sample_evaluate_register(prompt, "crossover", persona, accumulative=True)
                
                if child is not None:
                    offspring.append(child)
                
                # Log crossover
                if self._profiler and hasattr(self._profiler, 'log_crossover'):
                    self._profiler.log_crossover(
                        parent1.score if parent1.score else 0,
                        parent2.score if parent2.score else 0,
                        child.score if child else None
                    )
                
            except Exception as e:
                if self._debug_mode:
                    print(f"Crossover failed: {e}")
                continue
        
        return offspring

    def _mutation(self) -> List[Function]:
        """Perform mutation operations."""
        if not self._use_mutation:
            return []
        
        elite = self._population.get_elite()
        if elite is None:
            return []
        
        offspring = []
        num_mutations = max(1, int(self._pop_size * self._mutation_rate))
        
        for i in range(num_mutations):
            if self._tot_sample_nums >= self._max_sample_nums:
                break
            
            try:
                # Generate mutation prompt
                # Create user generator prompt
                persona = self._scientist_personas[0]
                user_generator = f"{persona} Your task is to write a {self._function_to_evolve_name} function for {self._task_description_str}"
                
                # Function signature not needed for mutation prompt
                
                # Get clean code
                elitist_code = str(elite) if elite else ""
                
                prompt = self._prompt_generator.get_mutation_prompt(
                    user_generator,
                    self._comprehensive_memory,
                    elitist_code,
                    self._function_to_evolve_name
                )
                
                # Generate mutated offspring
                child = self._sample_evaluate_register(prompt, f"mutation_{i+1}", persona, accumulative=True)
                
                if child is not None:
                    offspring.append(child)
                
                # Log mutation
                if self._profiler and hasattr(self._profiler, 'log_mutation'):
                    self._profiler.log_mutation(
                        elite.score if elite.score else 0,
                        child.score if child else None
                    )
                
            except Exception as e:
                if self._debug_mode:
                    print(f"Mutation failed: {e}")
                continue
        
        return offspring



    def _harmony_search(self) -> Optional[Function]:
        """Perform harmony search parameter tuning using optimized implementation."""
        # Select an individual for parameter tuning
        valid_individuals = self._population.get_valid_individuals()
        if not valid_individuals:
            print("Harmony Search: No valid individuals available")
            return None
        
        # Log all individuals in current population with their scores
        print(f"Harmony Search: Current population ({len(valid_individuals)} individuals):")
        for i, individual in enumerate(valid_individuals, 1):
            tried_hs_status = "✓" if (hasattr(individual, 'tried_hs') and individual.tried_hs) else "✗"
            score_str = f"{individual.score:.6f}" if individual.score is not None else "N/A"
            print(f"  [{i:2d}] Score: {score_str} | HS tried: {tried_hs_status}")
        
        # Select the best current individual that hasn't been tried with harmony search yet
        untried_individuals = [
            individual for individual in valid_individuals
            if not hasattr(individual, 'tried_hs') or not individual.tried_hs
        ]
        
        if untried_individuals:
            # Select the best scoring individual among those not yet tried
            candidate = max(untried_individuals, key=lambda x: x.score)
            print(f"Harmony Search: Selected best untried individual (score: {candidate.score:.6f})")
        else:
            # If all have been tried, select the best overall individual
            candidate = max(valid_individuals, key=lambda x: x.score)
            print(f"Harmony Search: All individuals tried, re-selecting best overall (score: {candidate.score:.6f})")
        
        # Mark as tried
        candidate.tried_hs = True
        
        try:
            # Generate harmony search prompt
            prompt = self._prompt_generator.get_harmony_search_prompt(str(candidate))
            response = self._sampler.get_harmony_search_response(prompt)
            
            if self._debug_mode:
                print(f"\n=== HSEvo HARMONY SEARCH SYSTEM PROMPT ===")
                system_prompt = self._sampler._prompt_generator.get_system_harmony_search_prompt()
                print(system_prompt)
                print(f"=== END HARMONY SEARCH SYSTEM PROMPT ===\n")
                
                print(f"\n=== HSEvo HARMONY SEARCH USER PROMPT ===")
                print(prompt)
                print(f"=== END HARMONY SEARCH USER PROMPT ===\n")
                
                print(f"=== HSEvo HARMONY SEARCH RESPONSE ===")
                print(response)
                print(f"=== END HARMONY SEARCH RESPONSE ===\n")
            
            # Use optimized harmony search implementation
            from .harmony_search_optimizer import OptimizedHarmonySearch
            
            optimizer = OptimizedHarmonySearch(
                hm_size=self._hm_size,
                hmcr=self._hmcr,
                par=self._par,
                bandwidth=self._bandwidth,
                max_iter_hs=self._max_iter_hs,
                debug_mode=self._debug_mode
            )
            
            # Perform optimization
            best_function, samples_used = optimizer.optimize_function(
                candidate, response, self._evaluator, self._template_program,
                self._tot_sample_nums, self._max_sample_nums
            )
            
            # Update sample count
            self._tot_sample_nums += samples_used
            
            # CRITICAL: The harmony search function has already been evaluated with preserved tuned parameters
            # We must register it directly to the population WITHOUT going through _sample_evaluate_register
            # which would lose the tuned parameters by converting with the standard method
            if best_function is not None:
                # Set timing information for compatibility
                best_function.sample_time = 0.0  # Harmony search time is tracked separately
                best_function.evaluate_time = 0.0  # Already evaluated during harmony search
                
                # Register directly to population (preserving tuned parameters) - use accumulative during generation
                self._population.add_individual_temporary(best_function)
                
                # Register with profiler
                if self._profiler is not None:
                    self._profiler.register_function(best_function)
                    if hasattr(self._profiler, 'register_population'):
                        self._profiler.register_population(self._population)
                
                if self._debug_mode:
                    print(f"Harmony Search: Successfully optimized function with score {best_function.score}")
                    if hasattr(best_function, 'tuned_parameters'):
                        print(f"Harmony Search: Tuned parameters: {best_function.tuned_parameters}")
            else:
                # Get specific error details from optimizer
                error_detail = getattr(optimizer, 'last_error', 'Unknown error')
                print(f"⚠️  Harmony Search: Score is None - {error_detail}")
            
            # Log to profiler if available
            if self._profiler and hasattr(self._profiler, 'log_harmony_search'):
                stats = optimizer.get_statistics()
                for improvement in stats.get('score_improvements', []):
                    iteration, score = improvement
                    self._profiler.log_harmony_search(iteration, stats['best_parameters'], score)
            
            return best_function
            
        except Exception as e:
            print(f"Harmony Search: Failed with exception: {e}")
            if self._debug_mode:
                traceback.print_exc()
        
        return None



    def _parse_flash_reflection(self, response: str) -> Tuple[str, str]:
        """Parse flash reflection response into analysis and experience."""
        try:
            # Look for Analysis and Experience sections
            analysis = ""
            experience = ""
            
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if '**Analysis:**' in line or 'Analysis:' in line:
                    current_section = 'analysis'
                    continue
                elif '**Experience:**' in line or 'Experience:' in line:
                    current_section = 'experience'
                    continue
                elif line.startswith('**') or line.startswith('##'):
                    current_section = None
                    continue
                
                if current_section == 'analysis' and line:
                    analysis += line + " "
                elif current_section == 'experience' and line:
                    experience += line + " "
            
            return analysis.strip(), experience.strip()
        except Exception:
            return "", ""

    def _continue_loop(self) -> bool:
        """Check if evolution should continue."""
        if self._max_sample_nums is not None and self._tot_sample_nums >= self._max_sample_nums:
            return False
        if self._max_generations is not None and self._generation >= self._max_generations:
            return False
        return True

    def _update_experiences(self):
        """Update experience lists based on performance."""
        current_best = self._population.best_score
        previous_best = getattr(self, '_previous_best', float('-inf'))
        
        experience = self._flash_memory.get("exp", "")
        if experience:
            if current_best > previous_best:
                self._good_experiences.append(experience)
            else:
                self._bad_experiences.append(experience)
        
        self._previous_best = current_best

    def run(self):
        """Run the HSEvo algorithm."""
        print("Starting HSEvo algorithm...")
        
        # Print operator configuration
        enabled_operators = []
        if self._use_crossover:
            enabled_operators.append("Crossover")
        if self._use_mutation:
            enabled_operators.append("Mutation")
        if self._use_harmony_search:
            enabled_operators.append("Harmony Search")
        
        print(f"Enabled operators: {', '.join(enabled_operators) if enabled_operators else 'None'}")
        
        if not enabled_operators:
            raise ValueError("At least one operator (crossover, mutation, or harmony search) must be enabled for HSEvo to run.")
        
        if not self._resume_mode:
            # Initialize population
            self._init_population()
        
        # Main evolution loop
        while self._continue_loop():
            print(f"\n--- HSEvo Generation {self._generation} ---")
            print(f"Samples: {self._tot_sample_nums}/{self._max_sample_nums}")
            print(f"Population size: {len(self._population)}")
            
            # Check if we have enough valid individuals
            valid_individuals = self._population.get_valid_individuals()
            if len(valid_individuals) < 2:
                print("Not enough valid individuals for evolution")
                break
            
            # Flash reflection
            self._flash_reflection()
            
            # Comprehensive reflection
            self._comprehensive_reflection()
            
            # Evolutionary operations
            all_offspring = []
            
            # Crossover operator
            if self._use_crossover:
                if self._debug_mode:
                    print("Performing crossover operation...")
                crossover_offspring = self._crossover()
                all_offspring.extend(crossover_offspring)
                if self._debug_mode:
                    print(f"Crossover operation generated {len(crossover_offspring)} offspring")
            
            # Mutation operator
            if self._use_mutation:
                if self._debug_mode:
                    print("Performing mutation operation...")
                mutation_offspring = self._mutation()
                all_offspring.extend(mutation_offspring)
                if self._debug_mode:
                    print(f"Mutation operation generated {len(mutation_offspring)} offspring")
            
            # Harmony search operator
            if self._use_harmony_search:
                if self._debug_mode:
                    print("Performing harmony search operation...")
                harmony_individual = self._harmony_search()
                if harmony_individual:
                    all_offspring.append(harmony_individual)
                    if self._debug_mode:
                        print("Harmony search operation generated 1 offspring")
                elif self._debug_mode:
                    print("Harmony search operation generated 0 offspring")
            
            # Show population before selection
            pre_selection_size = len(self._population)
            print(f"Population before selection: {pre_selection_size} individuals")
            
            # Survival selection (empty offspring list since we use accumulative mode now)
            self._population.survival_selection([], elite_individuals=self._elite_individuals)
            
            # Show population after selection
            post_selection_size = len(self._population)
            print(f"Population after selection: {post_selection_size} individuals (target: {self._pop_size}, elite: {self._elite_individuals})")
            
            # Update experiences
            self._update_experiences()
            
            # Log generation statistics
            if self._profiler and hasattr(self._profiler, 'register_population'):
                self._profiler.register_population(self._population)
            
            # Advance generation
            self._generation += 1
            self._population.next_generation()
            
            # Print generation summary
            stats = self._population.get_statistics()
            print(f"Best score: {stats['best_score']}")
            print(f"Valid individuals: {stats['valid_count']}")
        
        # Finish
        print(f"\nHSEvo completed after {self._generation} generations")
        print(f"Total samples: {self._tot_sample_nums}")
        print(f"Best score: {self._population.best_score}")
        
        if self._profiler is not None:
            self._profiler.finish()
        
        # Return best individual
        return self._population.best_individual 