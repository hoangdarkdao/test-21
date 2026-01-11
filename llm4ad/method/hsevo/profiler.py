from __future__ import annotations
import json
import os
from threading import Lock
from typing import List, Dict, Any
import numpy as np
import traceback

from .population import HSEvoPopulation
from ...base import Function
from ...tools.profiler import ProfilerBase

class HSEvoProfiler(ProfilerBase):
    def __init__(self,
                 log_dir: str | None = None,
                 num_objs=1,
                 result_folder=None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 **kwargs):
        super().__init__(log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         num_objs=num_objs,
                         result_folder=result_folder,
                         **kwargs)

        self._cur_gen = 0
        self._pop_lock = Lock()
        self._sample_lock = Lock()
        self._best_score = -float('inf') # Mặc định cho so sánh đơn mục tiêu

        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            self._reflection_dir = os.path.join(self._log_dir, 'reflection')
            os.makedirs(self._ckpt_dir, exist_ok=True)
            os.makedirs(self._reflection_dir, exist_ok=True)

    def _update_best(self, content: Dict[str, Any]):
        """Ghi đè để tránh lỗi khi so sánh List score (đa mục tiêu)"""
        try:
            score = content.get('score')
            if score is None:
                return

            # Nếu là đa mục tiêu, lấy mục tiêu đầu tiên (thường là score chính) để so sánh
            current_val = score[0] if isinstance(score, (list, tuple, np.ndarray)) else score
            
            if isinstance(current_val, (int, float)) and current_val > self._best_score:
                self._best_score = current_val
                # Gọi logic gốc của cha nếu cần hoặc tự cập nhật file best.json
                path = os.path.join(self._log_dir, 'best_ever.json')
                with open(path, 'w') as f:
                    json.dump(content, f, indent=4)
        except Exception:
            pass # Đảm bảo update_best không bao giờ làm dừng chương trình

    def register_function(self, prompt: str, func: Function, program: str = "", operation_type: str = 'unknown'):
        """Khớp với cách gọi: self._profiler.register_function(prompt, func, program=str(program))"""
        if not self._log_dir or func is None:
            return

        with self._sample_lock:
            self._num_samples += 1
            sample_order = self._num_samples
            
            score = func.score
            # Chuyển đổi score sang định dạng JSON serializable
            if score is not None:
                if np.isinf(np.array(score) if hasattr(score, '__iter__') else [score]).any():
                    score = None
                elif isinstance(score, (np.ndarray, np.generic)):
                    score = score.tolist()
                elif isinstance(score, tuple):
                    score = list(score)

            content = {
                'sample_order': sample_order,
                'operation': operation_type,
                'algorithm': getattr(func, 'algorithm', ''),
                'score': score,
                'function': str(func),
                'program': program,
                'entire_code': getattr(func, 'entire_code', str(func)),
                'prompt': prompt
            }

            # Lưu vào file sample (mỗi 300 mẫu một file)
            lower_bound = ((sample_order - 1) // 300) * 300
            upper_bound = lower_bound + 300
            filename = f'samples_{lower_bound + 1}~{upper_bound}.json'
            path = os.path.join(self._samples_json_dir, filename)

            try:
                data = []
                if os.path.exists(path):
                    with open(path, 'r') as fp:
                        try:
                            data = json.load(fp)
                        except json.JSONDecodeError:
                            data = []
                
                data.append(content)
                with open(path, 'w') as fp:
                    json.dump(data, fp, indent=4)

                self._update_best(content)
            except Exception:
                traceback.print_exc()

    def register_population(self, pop: HSEvoPopulation):
        if not self._log_dir:
            return

        with self._pop_lock:
            # Cho phép ghi đè log nếu thế hệ bằng nhau để cập nhật thông tin mới nhất
            funcs_json: List[Dict] = []
            for f in pop:
                score = f.score
                if score is not None and np.isinf(np.array(score) if hasattr(score, '__iter__') else [score]).any():
                    score = None
                
                funcs_json.append({
                    'algorithm': getattr(f, 'algorithm', ''),
                    'function': str(f),
                    'score': score.tolist() if isinstance(score, np.ndarray) else score,
                    'entire_code': getattr(f, 'entire_code', str(f))
                })

            path = os.path.join(self._ckpt_dir, f'pop_gen_{pop.generation}.json')
            with open(path, 'w') as fp:
                json.dump(funcs_json, fp, indent=4)
            self._cur_gen = pop.generation

    # ===================================================================
    # Reflection logging
    # ===================================================================
    def log_reflection(self, reflection_type: str, content: str, success: bool = True):
        """Log flash hoặc comprehensive reflection."""
        if not self._log_dir:
            return

        data = {
            'type': reflection_type,
            'content': content,
            'success': success,
            'generation': self._cur_gen,
            'sample_order': self._num_samples
        }

        filename = f'{reflection_type}_reflections.jsonl'
        path = os.path.join(self._reflection_dir, filename)

        with open(path, 'a') as fp:
            json.dump(data, fp)
            fp.write('\n')

    # ===================================================================
    # Harmony search logging (nếu có stats từ optimizer)
    # ===================================================================
    def log_harmony_search(self, iteration: int, best_parameters: Dict[str, float], score: float):
        if not self._log_dir:
            return

        data = {
            'iteration': iteration,
            'parameters': best_parameters,
            'score': score,
            'sample_order': self._num_samples
        }

        path = os.path.join(self._log_dir, 'harmony_search.jsonl')
        with open(path, 'a') as fp:
            json.dump(data, fp)
            fp.write('\n')

    # ===================================================================
    # Crossover / Mutation logging (tùy chọn, có thể log parent scores)
    # ===================================================================
    def log_crossover(self, parent1_score: Any, parent2_score: Any, child_score: Any):
        if not self._log_dir:
            return
        # Có thể mở rộng sau nếu cần

    def log_mutation(self, parent_score: Any, child_score: Any):
        if not self._log_dir:
            return
        # Có thể mở rộng sau nếu cần