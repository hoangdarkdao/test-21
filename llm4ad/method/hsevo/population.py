from __future__ import annotations

import copy
import random
import traceback
from typing import List, Optional, Tuple
import numpy as np

from ...base import Function


class HSEvoPopulation:
    """
    Population management cho HSEvo, được chỉnh lại để phù hợp với phong cách và cơ chế
    của MEoH trong project hiện tại:
    - Thread-safe với Lock
    - Register với non-dominated sorting + syntax match cho multi-objective (nếu cần)
    - Hỗ trợ duplicate checking và replacement tốt hơn
    - Giữ elite toàn cục (best ever)
    - Survival selection dựa trên Pareto + diversity (giống MEoH)
    - Tích hợp tốt với accumulative mode trong HSEvo
    """

    def __init__(self, pop_size: int = 10):
        self._pop_size = pop_size
        self._population: List[Function] = []
        self._next_gen_pop: List[Function] = []  # accumulative trong generation
        self._elitist: List[Function] = []       # archive các non-dominated best-ever
        self._generation = 0
        self._best_individual: Optional[Function] = None
        self._best_score = float('-inf')  # cho single-objective tracking (nếu evaluation là scalar)

        # Thread safety giống MEoH
        from threading import Lock
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._population)

    def __getitem__(self, index: int) -> Function:
        return self._population[index]

    def __iter__(self):
        return iter(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_individual(self) -> Optional[Function]:
        return self._best_individual

    @property
    def best_score(self) -> float:
        return self._best_score

    # ===================================================================
    # Accumulative registration (dùng trong HSEvo khi add_individual_temporary)
    # ===================================================================
    def add_individual_temporary(self, function: Function) -> None:
        """Thêm tạm vào next_gen_pop (accumulative trong generation)."""
        if function is None or function.score is None:
            return

        with self._lock:
            self._next_gen_pop.append(function)

            # Update global best (single-objective)
            if isinstance(function.score, (int, float)) and function.score > self._best_score:
                self._best_score = function.score
                self._best_individual = copy.deepcopy(function)

    # ===================================================================
    # Final registration (dùng khi init hoặc khi không dùng accumulative)
    # ===================================================================
    def register_function(self, function: Function) -> None:
        """Register chính thức – gọi survival selection ngay nếu cần."""
        if function is None or function.score is None:
            print("[REGISTER] Skipped: function or score is None")
            return

        # Bảo vệ: kiểm tra score có convert được sang np.array không
        try:
            np.array(function.score, dtype=float)  # thử convert trước
        except Exception as e:
            print(f"[REGISTER ERROR] Score không hợp lệ: {function.score} | {type(e).__name__}: {e}")
            return

        with self._lock:
            try:
                # Thêm vào next_gen_pop nếu không duplicate
                if not self.has_duplicate_function(function):
                    self._next_gen_pop.append(function)
                    print(f"[REGISTER] Added to next_gen_pop | current len: {len(self._next_gen_pop)}")
                else:
                    print("[REGISTER] Skipped: duplicate detected")

                # Trigger survival nếu đủ điều kiện (giống MEoH)
                if (len(self._next_gen_pop) >= self._pop_size or
                    (len(self._next_gen_pop) >= self._pop_size // 4 and self._generation == 0)):
                    
                    print(f"[REGISTER] Triggering survival selection... (next_gen: {len(self._next_gen_pop)})")
                    self._perform_survival_selection()
                    print(f"[REGISTER] Survival done | population size now: {len(self._population)}")

                # Update global best - hỗ trợ cả scalar và multi-objective
                current_score = function.score
                if isinstance(current_score, (list, tuple)):
                    score_value = max(current_score)  # hoặc min, hoặc weighted tùy bài toán
                else:
                    score_value = current_score

                if score_value > self._best_score:
                    self._best_score = score_value
                    self._best_individual = copy.deepcopy(function)
                    print(f"[REGISTER] New best updated: {score_value}")

            except Exception as e:
                print(f"[REGISTER CRASH] Trong quá trình register/survival: {type(e).__name__}")
                traceback.print_exc()
                # Không return ở đây để không bỏ qua hoàn toàn, nhưng có thể rollback nếu cần

    def _perform_survival_selection(self) -> None:
        """Thực hiện survival selection – kết hợp Pareto front + diversity (syntax match)."""
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        pop = self._population + self._next_gen_pop
        pop_elitist = pop + self._elitist

        # Lấy scores (hỗ trợ cả scalar và vector)
        objs = []
        for ind in pop_elitist:
            score = ind.score
            if isinstance(score, (int, float)):
                score = [score]  # convert scalar → vector để uniform
            objs.append(score if score is not None else [float('inf')])
        objs_array = np.array(objs)

        # Non-dominated sorting
        nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
        self._elitist = [pop_elitist[i] for i in nondom_idx]

        # Diversity selection cho population hiện tại (giống MEoH)
        crt_pop_size = len(pop)
        if crt_pop_size > self._pop_size:
            dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
            for i in range(crt_pop_size):
                for j in range(i + 1, crt_pop_size):
                    score_i = np.array(pop[i].score) if isinstance(pop[i].score, (list, tuple)) else np.array([pop[i].score])
                    score_j = np.array(pop[j].score) if isinstance(pop[j].score, (list, tuple)) else np.array([pop[j].score])

                    if (score_i <= score_j).all() and not (score_i == score_j).all():
                        dominated_counts[i, j] = -self._syntax_match(pop[i], pop[j])
                    elif (score_j <= score_i).all() and not (score_i == score_j).all():
                        dominated_counts[j, i] = -self._syntax_match(pop[j], pop[i])

            dominated_counts_sum = dominated_counts.sum(axis=1)  # higher = less dominated → better
            indices = np.argsort(-dominated_counts_sum)[:self._pop_size]  # descending
            self._population = [pop[i] for i in indices]
        else:
            self._population = pop[:self._pop_size]

        # Reset next_gen_pop và tăng generation
        self._next_gen_pop = []
        self._generation += 1

    def _syntax_match(self, f1: Function, f2: Function) -> float:
        """Tính similarity bằng codebleu syntax match (0-1, higher = similar hơn)."""
        try:
            from codebleu.syntax_match import calc_syntax_match
            return calc_syntax_match([f1.entire_code or str(f1)], [f2.entire_code or str(f2)], 'python')
        except:
            return 0.0

    # ===================================================================
    # Duplicate checking (giống MEoH)
    # ===================================================================
    def has_duplicate_function(self, func: Function) -> bool:
        """Kiểm tra duplicate và thay thế nếu mới tốt hơn."""
        if func.score is None:
            return True

        # Check trong population hiện tại
        for i, f in enumerate(self._population):
            if str(f) == str(func):
                if self._better(func, f):
                    self._population[i] = func
                    return True

        # Check trong next_gen_pop
        for i, f in enumerate(self._next_gen_pop):
            if str(f) == str(func):
                if self._better(func, f):
                    self._next_gen_pop[i] = func
                    return True

        # Check trong elitist archive
        for i, f in enumerate(self._elitist):
            if str(f) == str(func):
                if self._better(func, f):
                    self._elitist[i] = func
                    return True

        return False

    def _better(self, new: Function, old: Function) -> bool:
        """So sánh 2 individuals (hỗ trợ cả scalar và multi-objective)."""
        new_score = new.score if isinstance(new.score, (list, tuple)) else [new.score]
        old_score = old.score if isinstance(old.score, (list, tuple)) else [old.score]

        new_arr = np.array(new_score)
        old_arr = np.array(old_score)

        # Pareto better: new dominates old
        if (new_arr <= old_arr).all() and (new_arr < old_arr).any():
            return True
        return False

    # ===================================================================
    # Selection methods (giống HSEvo cũ nhưng dùng population hiện tại)
    # ===================================================================
    def get_valid_individuals(self) -> List[Function]:
        valid = [ind for ind in self._population if ind.score is not None and not np.isinf(np.array(ind.score if isinstance(ind.score, (list, tuple)) else [ind.score])).any()]
        return valid

    def get_sorted_individuals(self, reverse: bool = True) -> List[Function]:
        valid = self.get_valid_individuals()
        return sorted(valid, key=lambda x: x.score if isinstance(x.score, (int, float)) else max(x.score), reverse=reverse)

    def selection(self, tournament_size: int = 3) -> Optional[Function]:
        """Tournament selection trên population hiện tại."""
        valid = self.get_valid_individuals()
        if not valid:
            return random.choice(self._population) if self._population else None
        if len(valid) <= tournament_size:
            return max(valid, key=lambda x: x.score if isinstance(x.score, (int, float)) else max(x.score))
        tournament = random.sample(valid, tournament_size)
        return max(tournament, key=lambda x: x.score if isinstance(x.score, (int, float)) else max(x.score))

    def select_for_crossover(self, selection_num: int = 2) -> List[Tuple[Function, Function]]:
        """Chọn các cặp parent từ top individuals."""
        valid = self.get_valid_individuals()
        if len(valid) < 2:
            return []

        sorted_inds = self.get_sorted_individuals()[:max(selection_num * 2, len(valid))]
        pairs = []
        for i in range(0, len(sorted_inds) - 1, 2):
            if i + 1 < len(sorted_inds):
                pairs.append((sorted_inds[i], sorted_inds[i + 1]))
        return pairs

    def get_elite(self) -> Optional[Function]:
        """Lấy best hiện tại (scalar hoặc max của vector)."""
        valid = self.get_valid_individuals()
        if not valid:
            return None
        return max(valid, key=lambda x: x.score if isinstance(x.score, (int, float)) else max(x.score))

    def get_diversity_sample(self, sample_size: int = 5) -> List[Function]:
        """Lấy sample đa dạng theo rank."""
        sorted_inds = self.get_sorted_individuals()
        if len(sorted_inds) <= sample_size:
            return sorted_inds
        indices = np.linspace(0, len(sorted_inds) - 1, sample_size, dtype=int)
        return [sorted_inds[i] for i in indices]

    # ===================================================================
    # Survival selection gọi từ HSEvo.run() (khi kết thúc generation)
    # ===================================================================
    def survival_selection(self, offspring: List[Function], elite_individuals: int = 1) -> None:
        """
        Gọi cuối mỗi generation trong HSEvo.
        Vì HSEvo dùng accumulative → offspring thường rỗng.
        """
        with self._lock:
            # Thêm offspring nếu có (backward compatibility)
            for ind in offspring:
                if ind and ind.score is not None:
                    if not self.has_duplicate_function(ind):
                        self._next_gen_pop.append(ind)

            # Thực hiện survival
            if self._next_gen_pop:  # chỉ khi có cá thể mới
                self._perform_survival_selection()

    def next_generation(self) -> None:
        """Tăng generation (được gọi trong HSEvo.run())."""
        with self._lock:
            self._generation += 1

    def get_statistics(self) -> dict:
        valid = self.get_valid_individuals()
        if not valid:
            return {
                'size': len(self._population),
                'valid_count': 0,
                'best_score': None,
                'generation': self._generation
            }

        scores = [ind.score if isinstance(ind.score, (int, float)) else max(ind.score) for ind in valid]
        return {
            'size': len(self._population),
            'valid_count': len(valid),
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'generation': self._generation
        }

    def clear(self) -> None:
        with self._lock:
            self._population.clear()
            self._next_gen_pop.clear()
            self._elitist.clear()
            self._generation = 0
            self._best_individual = None
            self._best_score = float('-inf')