from __future__ import annotations

import json
import os
import numpy as np
from threading import Lock
from typing import List, Dict, Optional

from .population import Population
from ...base import Function
from ...tools.profiler import ProfilerBase


class ReEvoProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: Optional[str] = None,
                 num_objs=2,
                 result_folder=None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        super().__init__(log_dir=log_dir,
                         num_objs=num_objs,
                         result_folder=result_folder,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)

        self._cur_gen = 0
        self._pop_lock = Lock()

        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Population logging (same semantics as MOMCTSProfiler)
    # ------------------------------------------------------------------
    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()

            if self._num_samples == 0 or pop.generation == self._cur_gen:
                return

            funcs = pop.population  # type: List[Function]
            funcs_json = []

            for f in funcs:
                f_score = f.score
                if f_score is not None:
                    if np.isinf(np.array(f_score)).any():
                        f_score = None

                funcs_json.append({
                    'algorithm': f.algorithm,
                    'function': str(f),
                    'score': f_score
                })

            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file, indent=4)

            self._cur_gen += 1

        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    # ------------------------------------------------------------------
    # JSON writer (fully aligned with MOMCTSProfiler)
    # ------------------------------------------------------------------
    def _write_json(self,
                    prompt: str,
                    function: Function,
                    program='',
                    *,
                    record_type='history',
                    record_sep=300,
                    **kwargs):
        """
        Write function data to a JSON file.
        """
        assert record_type in ['history', 'best']

        if not self._log_dir:
            return

        sample_order = self._num_samples

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        content = {
            'prompt': prompt,
            'sample_order': sample_order,
            'algorithm': function.algorithm,
            'function': str(function),
            'score': convert(function.score),
            'program': program,
        }

        if 'op' in kwargs:
            content['operation'] = kwargs['op']

        if record_type == 'history':
            lower = ((sample_order - 1) // record_sep) * record_sep
            upper = lower + record_sep
            filename = f'samples_{lower + 1}~{upper}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
