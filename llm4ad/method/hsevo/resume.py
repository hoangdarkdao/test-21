from __future__ import annotations

import json
import os
from tqdm.auto import tqdm
import copy

from .profiler import HSEvoProfiler
from .population import HSEvoPopulation
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_latest_pop_json(log_path: str):
    path = os.path.join(log_path, 'population')
    if not os.path.exists(path):
        return None, 0
    files = [f for f in os.listdir(path) if f.startswith('pop_gen_') and f.endswith('.json')]
    if not files:
        return None, 0
    orders = [int(f.split('_')[2].split('.')[0]) for f in files]
    max_gen = max(orders)
    return os.path.join(path, f'pop_gen_{max_gen}.json'), max_gen


def _get_all_samples_and_scores(log_path: str):
    samples_dir = os.path.join(log_path, 'samples')
    if not os.path.exists(samples_dir):
        return [], [], 0

    files = sorted(os.listdir(samples_dir), key=lambda x: int(x.split('_')[1].split('~')[0]))
    all_funcs = []
    all_scores = []
    max_order = 0

    for file in files:
        filepath = os.path.join(samples_dir, file)
        with open(filepath, 'r') as f:
            data = json.load(f)
        for item in data:
            all_funcs.append(item['function'])
            all_scores.append(item['score'] if item['score'] is not None else float('-inf'))
            max_order = max(max_order, item['sample_order'])

    return all_funcs, all_scores, max_order


def _resume_pop(log_path: str, pop_size: int) -> HSEvoPopulation:
    path, max_gen = _get_latest_pop_json(log_path)
    if path is None:
        print("No population checkpoint found for resume.")
        return HSEvoPopulation(pop_size=pop_size)

    print(f'RESUME HSEvo: Latest generation: {max_gen}')
    with open(path, 'r') as f:
        data = json.load(f)

    pop = HSEvoPopulation(pop_size=pop_size)
    for d in data:
        func_str = d['function']
        func = tfpc.text_to_function(func_str)
        if func is None:
            continue
        func.score = d['score']
        func.algorithm = d.get('algorithm', '')
        if hasattr(func, 'entire_code'):
            func.entire_code = d.get('entire_code', func_str)
        pop.register_function(func)

    # Force generation
    pop._generation = max_gen + 1  # vì trong run() sẽ gọi next_generation() sau survival
    return pop


def _resume_text2func(func_str: str, score: Any, template_func: Function) -> Function:
    func = tfpc.text_to_function(func_str)
    if func is None:
        temp = copy.deepcopy(template_func)
        temp.body = '    pass'
        temp.score = None
        return temp
    func.score = score
    return func


def _resume_profiler(log_path: str, profiler: HSEvoProfiler, template_func: Function):
    funcs, scores, max_sample_order = _get_all_samples_and_scores(log_path)
    if max_sample_order == 0:
        return

    print(f'RESUME HSEvo: Restoring {max_sample_order} samples into profiler...')
    profiler._num_samples = max_sample_order  # bypass counter

    # Không cần register từng cái vì profiler chỉ log, không lưu state nội tại
    # Nhưng để wandb/metrics đúng, có thể gọi register_function với resume_mode nếu cần
    # Ở đây ta chỉ cần set _num_samples là đủ


def resume_hsevo(hsevo) -> None:
    """
    Resume HSEvo từ log_dir của profiler.
    Gọi hàm này trước khi chạy hsevo.run() nếu resume_mode=True.
    """
    if not hsevo._resume_mode:
        return

    profiler = hsevo._profiler
    if not profiler or not profiler._log_dir:
        print("Resume failed: No log_dir in profiler.")
        return

    log_path = profiler._log_dir

    # Resume population
    pop = _resume_pop(log_path, hsevo._pop_size)
    hsevo._population = pop

    # Resume sample counter
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    hsevo._tot_sample_nums = sample_max_order

    # Resume profiler counter
    _resume_profiler(log_path, profiler, hsevo._function_to_evolve)

    # Resume generation counter (dựa trên population)
    latest_gen, _ = _get_latest_pop_json(log_path)
    if latest_gen is not None:
        hsevo._generation = latest_gen + 1

    print(f"HSEvo resume complete: generation ~{hsevo._generation}, samples = {hsevo._tot_sample_nums}")