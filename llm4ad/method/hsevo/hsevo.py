from __future__ import annotations

import concurrent.futures
import copy
import random
import time
import traceback
from threading import Thread
from typing import Dict, Optional, List, Tuple, Literal
import numpy as np

from .population import HSEvoPopulation
from .profiler import HSEvoProfiler
from .prompt import HSEvoPrompt
from .sampler import HSEvoSampler
from .resume import resume_hsevo  # Sử dụng hàm resume có sẵn
from ...base import (
    Evaluation, LLM, Function, Program,
    TextFunctionProgramConverter as tfpc, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
from .harmony_search_optimizer import OptimizedHarmonySearch

class HSEvo:
    """
    Hybrid Search Evolution (HSEvo) - Automated Heuristic Design Framework
    Multi-objective: maximize score, minimize evaluate_time
    """

    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 20,
                 max_sample_nums: Optional[int] = 300,
                 pop_size: int = 10,
                 init_pop_size: int = 10,
                 selection_num: int = 3,
                 mutation_rate: float = 0.7,
                 elite_individuals: int = 2,
                 # Harmony search params
                 use_harmony_search: bool = True,
                 hm_size: int = 5,
                 hmcr: float = 0.9,
                 par: float = 0.3,
                 bandwidth: float = 0.05,
                 max_iter_hs: int = 6,
                 # Operators control
                 use_crossover: bool = True,
                 use_mutation: bool = True,
                 # Execution
                 num_samplers: int = 2,
                 num_evaluators: int = 2,
                 temperature: float = 0.75,
                 timeout: int = 60,
                 # Reflection
                 use_flash_reflection: bool = True,
                 use_comprehensive_reflection: bool = False,
                 # Seed & resume
                 seed_func: Optional[str] = None,
                 eval_seed_func: bool = False,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description

        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._init_pop_size = init_pop_size
        self._selection_num = selection_num
        self._mutation_rate = mutation_rate
        self._elite_individuals = elite_individuals

        # Harmony search
        self._use_harmony_search = use_harmony_search
        self._hm_size = hm_size
        self._hmcr = hmcr
        self._par = par
        self._bandwidth = bandwidth
        self._max_iter_hs = max_iter_hs

        # Operators
        self._use_crossover = use_crossover
        self._use_mutation = use_mutation

        # Execution
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._temperature = temperature
        self._timeout = timeout
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # Reflection
        self._use_flash_reflection = use_flash_reflection
        self._use_comprehensive_reflection = use_comprehensive_reflection

        # Seed
        self._seed_func = seed_func
        self._eval_seed_func = eval_seed_func

        # Resume
        self._resume_mode = resume_mode

        # Core components
        self._function_to_evolve = tfpc.text_to_function(self._template_program_str)
        self._template_program = tfpc.text_to_program(self._template_program_str)

        self._population = HSEvoPopulation(pop_size=self._pop_size)
        self._sampler = HSEvoSampler(llm, self._template_program)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, timeout=timeout, **kwargs)
        self._profiler = profiler

        # Statistics
        self._tot_sample_nums = 0
        self._generation = 0

        # Multi-thread executor
        if multi_thread_or_process_eval == 'thread':
            self._eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._eval_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        # Logging
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    def _sample_evaluate_register(self, prompt: str, operation: str = "unknown") -> bool:
        """Lấy sample → evaluate → register (accumulative mode)"""
        try:
            # 1. Lấy kết quả từ Sampler
            thought, func = self._sampler.get_thought_and_function(prompt)
            if func is None:
                if self._debug_mode: print("⚠️ Sampler returned None function.")
                return False

            # 2. Chuyển đổi sang Program
            program = tfpc.function_to_program(func, self._template_program)
            if program is None:
                if self._debug_mode: print("⚠️ Failed to convert function to program.")
                return False

            # 3. Đánh giá (Evaluate)
            # Lưu ý: future.result() trả về (score, eval_time)
            future = self._eval_executor.submit(
                self._evaluator.evaluate_program_record_time, program
            )
            result = future.result()
            
            if result is None or result[0] is None:
                if self._debug_mode: print(f"⚠️ Evaluation failed for {operation}.")
                return False
            
            score, eval_time = result
            
            # 4. Gán thông tin cho đối tượng function
            func.score = score
            func.evaluate_time = eval_time
            func.algorithm = thought 
            func.sample_time = 0.0          

            # 5. Đăng ký vào population
            self._population.register_function(func)
            
            # 6. Profiler (Đoạn này dễ gây lỗi nhất nên cần bọc kín)
            if self._profiler is not None:
                try:
                    self._profiler.register_function(prompt, func, program=str(program))
                    
                    # Kiểm tra và đăng ký population nếu là HSEvoProfiler
                    if isinstance(self._profiler, HSEvoProfiler):
                        self._profiler.register_population(self._population)
                        
                    self._tot_sample_nums += 1
                except Exception as profiler_err:
                    print(f"❌ Profiler Error (but continuing): {profiler_err}")
                    if self._debug_mode:
                        traceback.print_exc() 
            return True

        except Exception as e:
            print(f"❌ Critical Error in _sample_evaluate_register: {e}")
            traceback.print_exc() # Luôn in traceback để biết lỗi ở dòng nào
            return False
        
    def _init_population(self):
        """Khởi tạo population bằng i1 prompt - chạy tuần tự"""
        print(f"Initializing population (target ~{self._init_pop_size})...")

        prompts = [
            HSEvoPrompt.get_prompt_i1(
                self._task_description_str,
                self._function_to_evolve,
                scientist_persona=random.choice([
                    "", "You are an expert heuristic designer."
                ])
            )
            for _ in range(self._init_pop_size)  
        ]

        valid_count = 0

        for i, prompt in enumerate(prompts, 1):
            if self._tot_sample_nums >= self._max_sample_nums:
                print(f"Đạt giới hạn max_sample_nums ({self._max_sample_nums}), dừng khởi tạo.")
                break

            print(f"  [{i}/{len(prompts)}] Sampling & evaluating...", end=" ", flush=True)

            success = self._sample_evaluate_register(prompt, "init")

            if success:
                valid_count += 1
                print(f"OK (valid: {valid_count})")
            else:
                print("FAILED")

        print(f"Initialization finished. Valid individuals: {len(self._population.get_valid_individuals())}")
        print(f"  Tổng số mẫu đã thử: {self._tot_sample_nums}")
    
    def _evolutionary_step(self):
        if not self._continue():
            return

        # Crossover
        if self._use_crossover:
            pairs = self._population.select_for_crossover(self._selection_num)
            for p1, p2 in pairs:
                if random.random() < 0.85:  # crossover probability
                    prompt = HSEvoPrompt.get_prompt_e2(
                        self._task_description_str, [p1, p2], self._function_to_evolve
                    )
                    self._sample_evaluate_register(prompt, "crossover")

        # Mutation
        if self._use_mutation:
            for _ in range(max(1, int(self._mutation_rate * self._pop_size))):
                parent = self._population.selection(tournament_size=3)
                if parent is None:
                    continue

                # Chọn loại mutation ngẫu nhiên
                if random.random() < 0.6:
                    prompt = HSEvoPrompt.get_prompt_m1(self._task_description_str, parent, self._function_to_evolve)
                else:
                    prompt = HSEvoPrompt.get_prompt_m2(self._task_description_str, parent, self._function_to_evolve)

                self._sample_evaluate_register(prompt, "mutation")

        if self._use_harmony_search:
            candidate = self._population.get_elite()
            if candidate:
                prompt = HSEvoPrompt.get_prompt_harmony_search(candidate)
                response = self._sampler.get_harmony_search_response(prompt)
                print("Harmony Search triggered (implementation pending)")

    def _continue(self) -> bool:
        """Kiểm tra điều kiện dừng"""
        if self._max_sample_nums is not None and self._tot_sample_nums >= self._max_sample_nums:
            return False
        if self._max_generations is not None and self._generation >= self._max_generations:
            return False
        return True
    
    def _flash_reflection(self) -> Dict[str, str]:
        """Thực hiện Flash Reflection trên các thuật toán gần đây."""
        
        # Lấy mẫu đa dạng từ quần thể (thường là 5 cá thể)
        sample_algorithms = self._population.get_diversity_sample(5)
        
        if len(sample_algorithms) < 2:
            return {"analyze": "", "exp": ""}
        
        try:
            algorithms_text = ""
            
            for i, alg in enumerate(sample_algorithms):
                suffix = "th" if 11 <= i + 1 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get((i + 1) % 10, "th")
                algorithms_text += f"[Heuristics {i + 1}{suffix}]\n{str(alg)}\n\n"
            
            prompt = HSEvoPrompt.get_prompt_flash_reflection(
                self._task_description_str, 
                algorithms_text            
            )

            # Gọi qua sampler để lấy text phân tích (temperature=0.0)
            response = self._sampler.get_flash_reflection_response(prompt)
            
            # Parse kết quả trả về từ LLM
            analysis, experience = self._parse_flash_reflection(response)
            
            self._flash_memory = {"analyze": analysis, "exp": experience}
            
            if self._profiler and hasattr(self._profiler, 'log_reflection'):
                self._profiler.log_reflection('flash', response, True)
                
            return self._flash_memory
        except Exception as e:
            if self._debug_mode: print(f"Flash reflection failed: {e}")
            return {"analyze": "", "exp": ""}
    
    def _parse_flash_reflection(self, response: str) -> Tuple[str, str]:
        """Tách phản hồi của LLM thành phần Phân tích và Kinh nghiệm."""
        analysis = ""
        experience = ""
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'Analysis:' in line:
                current_section = 'analysis'
                continue
            elif 'Experience:' in line:
                current_section = 'experience'
                continue
            
            if current_section == 'analysis' and line:
                analysis += line + " "
            elif current_section == 'experience' and line:
                experience += line + " "
        return analysis.strip(), experience.strip()
    
    def _harmony_search(self) -> Optional[Function]:
        """Thực hiện Harmony Search để tinh chỉnh tham số."""
        if not self._use_harmony_search:
            return None

        valid_individuals = self._population.get_valid_individuals()
        if not valid_individuals: return None

        # Chọn cá thể tốt nhất chưa thử HS
        untried = [ind for ind in valid_individuals if not hasattr(ind, 'tried_hs') or not ind.tried_hs]
        candidate = max(untried, key=lambda x: x.score) if untried else max(valid_individuals, key=lambda x: x.score)
        
        candidate.tried_hs = True

        try:
            # ĐÚNG: Gọi trực tiếp từ self._prompt_generator
            prompt = HSEvoPrompt.get_prompt_harmony_search(candidate)
            
            # Sampler trả về raw response bao gồm code mẫu và dải tham số
            response = self._sampler.get_harmony_search_response(prompt)
            
            # Sử dụng lớp Optimizer chuyên dụng
            from .harmony_search_optimizer import OptimizedHarmonySearch
            
            optimizer = OptimizedHarmonySearch(
                hm_size=self._hm_size, hmcr=self._hmcr, par=self._par,
                bandwidth=self._bandwidth, max_iter_hs=self._max_iter_hs,
                debug_mode=self._debug_mode
            )
            
            # Tiến hành tối ưu hóa tham số số học
            best_func, samples_used = optimizer.optimize_function(
                candidate, response, self._evaluator, self._template_program,
                self._tot_sample_nums, self._max_sample_nums
            )
            
            self._tot_sample_nums += samples_used
            
            if best_func is not None:
                # Đăng ký trực tiếp vào population (chế độ tạm thời trong gen)
                self._population.add_individual_temporary(best_func)
                
                if self._profiler:
                    self._profiler.register_function(best_func)
                return best_func
                
        except Exception as e:
            if self._debug_mode: 
                print(f"Harmony Search failed: {e}")
                traceback.print_exc()
        return None
    
    def run(self):
        """Chạy toàn bộ quy trình HSEvo"""
        print("HSEvo started.")

        if self._resume_mode:
            resume_hsevo(self)
            print(f"Resumed at generation {self._population.generation}, samples: {self._tot_sample_nums}")

        if self._generation == 0:
            self._init_population()

        while self._continue():
            self._generation += 1
            print(f"\nGeneration {self._generation} | Samples: {self._tot_sample_nums}")
            
            self._flash_reflection()
            # Sinh offspring
            sampler_threads = [
                Thread(target=self._evolutionary_step)
                for _ in range(self._num_samplers)
            ]

            for t in sampler_threads:
                t.start()
            for t in sampler_threads:
                t.join()
            
            if self._use_harmony_search:
                self._harmony_search()

            # Survival selection (sẽ dùng Pareto + diversity trong population)
            self._population.survival_selection([], elite_individuals=self._elite_individuals)

            # Thống kê
            stats = self._population.get_statistics()
            best = self._population.best_individual
            best_score = best.score[0] if best and isinstance(best.score, list) else None

            print(f"  Valid: {stats['valid_count']}/{len(self._population)}")
            print(f"  Best score: {best_score:.4f} | Eval time: {best.score[1]:.4f}s" if best_score else "No valid yet")

            if self._profiler and hasattr(self._profiler, 'register_population'):
                self._profiler.register_population(self._population)

        print("\nHSEvo finished.")
        print(f"Total samples: {self._tot_sample_nums}")
        print(f"Final generation: {self._generation}")

        best = self._population.best_individual
        if best:
            print(f"Best individual score: {best.score[0]:.4f} | time: {best.score[1]:.4f}s")
            print(f"Code:\n{str(best)}")

        if self._profiler:
            self._profiler.finish()

        return best