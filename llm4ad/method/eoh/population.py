from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._elitist = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elitist(self):
        return self._elitist

    @property
    def generation(self):
        return self._generation

    def survival(self):
        pop = self._population + self._next_gen_pop
        pop = sorted(pop, key=lambda f: f.score)
        self._population = pop[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if func.score is None:
            return

        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size or (len(self._next_gen_pop) >= self._pop_size//5 and self._generation == 0):
                print("Perform suvrival")
                self.survival()
            else: 
                print("Do not surviva")
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self) -> Function:
        funcs = [f for f in self._population if not math.isinf(f.score)]
        
        if len(funcs) == 0:
            print("⚠️ Warning: No valid functions in population, returning random from all")
            funcs = [f for f in self._population]
            if len(funcs) == 0:
                return None
            return funcs[0]
        
        func = sorted(funcs, key=lambda f: f.score)
        p = [1 / (r + 1 + len(func)) for r in range(len(func))]  # Avoid division by zero
        p = np.array(p)
        p = p / np.sum(p)
        return np.random.choice(func, p=p)