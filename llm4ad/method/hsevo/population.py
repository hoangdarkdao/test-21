from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple
import numpy as np

from ...base import Function


class HSEvoPopulation:
    """Population management for HSEvo algorithm."""
    
    def __init__(self, pop_size: int = 5):
        """Initialize HSEvo population.
        
        Args:
            pop_size: Maximum population size
        """
        self._pop_size = pop_size
        self._individuals: List[Function] = []
        self._generation = 0
        self._best_individual: Optional[Function] = None
        self._best_score = float('-inf')
        
    def __len__(self) -> int:
        """Get current population size."""
        return len(self._individuals)
    
    def __getitem__(self, index: int) -> Function:
        """Get individual by index."""
        return self._individuals[index]
    
    def __iter__(self):
        """Iterate over population."""
        return iter(self._individuals)
    
    @property
    def generation(self) -> int:
        """Get current generation number."""
        return self._generation
    
    @property
    def best_individual(self) -> Optional[Function]:
        """Get best individual in population."""
        return self._best_individual
    
    @property
    def best_score(self) -> float:
        """Get best score in population."""
        return self._best_score
    
    def add_individual_temporary(self, function: Function) -> None:
        """Add individual to population temporarily (accumulative during generation).
        
        Args:
            function: Function to add to population
        """
        if function is None:
            return
            
        # Add to population without trimming (accumulative)
        self._individuals.append(function)
        
        # Update best individual if needed
        if function.score is not None and function.score > self._best_score:
            self._best_score = function.score
            self._best_individual = copy.deepcopy(function)
    
    def register_function(self, function: Function) -> None:
        """Register a new function to the population with immediate size control.
        
        Args:
            function: Function to add to population
        """
        if function is None:
            return
            
        # Add to population
        self._individuals.append(function)
        
        # Update best individual if needed
        if function.score is not None and function.score > self._best_score:
            self._best_score = function.score
            self._best_individual = copy.deepcopy(function)
        
        # Maintain population size (for backward compatibility)
        if len(self._individuals) > self._pop_size:
            self._individuals = self._individuals[-self._pop_size:]
    
    def get_valid_individuals(self) -> List[Function]:
        """Get all individuals with valid scores.
        
        Returns:
            List of valid individuals
        """
        return [ind for ind in self._individuals if ind.score is not None and not np.isinf(ind.score)]
    
    def get_sorted_individuals(self, reverse: bool = True) -> List[Function]:
        """Get individuals sorted by score.
        
        Args:
            reverse: If True, sort in descending order (best first)
            
        Returns:
            List of sorted individuals
        """
        valid_individuals = self.get_valid_individuals()
        return sorted(valid_individuals, key=lambda x: x.score, reverse=reverse)
    
    def selection(self, tournament_size: int = 2) -> Function:
        """Select an individual using tournament selection.
        
        Args:
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        valid_individuals = self.get_valid_individuals()
        
        if len(valid_individuals) == 0:
            # Return a random individual if no valid ones
            return random.choice(self._individuals) if self._individuals else None
        
        if len(valid_individuals) <= tournament_size:
            # Return best individual if population is small
            return max(valid_individuals, key=lambda x: x.score)
        
        # Tournament selection
        tournament = random.sample(valid_individuals, tournament_size)
        return max(tournament, key=lambda x: x.score)
    
    def select_parents(self, num_parents: int = 2) -> List[Function]:
        """Select multiple parents for crossover.
        
        Args:
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        parents = []
        for _ in range(num_parents):
            parent = self.selection()
            if parent is not None:
                parents.append(parent)
        
        return parents
    
    def select_for_crossover(self, selection_num: int = 2) -> List[Tuple[Function, Function]]:
        """Select pairs of parents for crossover operations.
        
        Args:
            selection_num: Number of individuals to select for crossover
            
        Returns:
            List of parent pairs
        """
        valid_individuals = self.get_valid_individuals()
        
        if len(valid_individuals) < 2:
            return []
        
        pairs = []
        selected_count = min(selection_num, len(valid_individuals))
        
        # Select top individuals for crossover
        sorted_individuals = self.get_sorted_individuals()[:selected_count]
        
        # Create pairs
        for i in range(0, len(sorted_individuals) - 1, 2):
            if i + 1 < len(sorted_individuals):
                pairs.append((sorted_individuals[i], sorted_individuals[i + 1]))
        
        return pairs
    
    def get_elite(self) -> Optional[Function]:
        """Get the elite individual (best performer).
        
        Returns:
            Elite individual or None if no valid individuals
        """
        valid_individuals = self.get_valid_individuals()
        
        if not valid_individuals:
            return None
        
        return max(valid_individuals, key=lambda x: x.score)
    
    def get_diversity_sample(self, sample_size: int = 5) -> List[Function]:
        """Get a diverse sample of individuals for analysis.
        
        Args:
            sample_size: Number of individuals to sample
            
        Returns:
            List of diverse individuals
        """
        valid_individuals = self.get_valid_individuals()
        
        if len(valid_individuals) <= sample_size:
            return valid_individuals
        
        # Use score-based sampling to get diverse representation
        sorted_individuals = self.get_sorted_individuals()
        
        if sample_size == 1:
            return [sorted_individuals[0]]
        
        # Sample across the performance spectrum
        indices = np.linspace(0, len(sorted_individuals) - 1, sample_size, dtype=int)
        return [sorted_individuals[i] for i in indices]
    
    def clear(self) -> None:
        """Clear the population."""
        self._individuals.clear()
        self._generation = 0
        self._best_individual = None
        self._best_score = float('-inf')
    
    def next_generation(self) -> None:
        """Advance to next generation."""
        self._generation += 1
    
    def get_statistics(self) -> dict:
        """Get population statistics.
        
        Returns:
            Dictionary containing population statistics
        """
        valid_individuals = self.get_valid_individuals()
        
        if not valid_individuals:
            return {
                'size': len(self._individuals),
                'valid_count': 0,
                'best_score': None,
                'worst_score': None,
                'mean_score': None,
                'std_score': None,
                'generation': self._generation
            }
        
        scores = [ind.score for ind in valid_individuals]
        
        return {
            'size': len(self._individuals),
            'valid_count': len(valid_individuals),
            'best_score': max(scores),
            'worst_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'generation': self._generation
        }
    
    def survival_selection(self, offspring: List[Function], elite_individuals: int = 1) -> None:
        """Perform survival selection with elite preservation and random selection.
        
        Args:
            offspring: List of offspring to consider for survival
            elite_individuals: Number of elite individuals to preserve (best performers)
        """
        # Note: Since we now use accumulative mode, self._individuals already contains
        # all individuals from the generation (parents + offspring from all operations)
        # The offspring parameter is kept for backward compatibility but may be empty
        
        # If offspring is provided, add them to the current population
        if offspring:
            for ind in offspring:
                if ind is not None:
                    self._individuals.append(ind)
        
        # Filter valid individuals
        valid_individuals = [ind for ind in self._individuals if ind.score is not None and not np.isinf(ind.score)]
        
        if len(valid_individuals) <= self._pop_size:
            # If we have fewer or equal valid individuals, keep them all
            self._individuals = valid_individuals
            
            # Fill with invalid individuals if needed
            if len(self._individuals) < self._pop_size:
                invalid_individuals = [ind for ind in self._individuals if ind not in valid_individuals]
                needed = self._pop_size - len(self._individuals)
                if invalid_individuals:
                    self._individuals.extend(invalid_individuals[:needed])
        else:
            # Elite + Random selection strategy
            # Sort valid individuals by score (best first)
            valid_individuals.sort(key=lambda x: x.score, reverse=True)
            
            # Use specified number of elite individuals, but ensure it doesn't exceed population or available individuals
            num_elite = min(elite_individuals, self._pop_size, len(valid_individuals))
            num_random = self._pop_size - num_elite
            
            # Keep elite individuals (best performers)
            selected = valid_individuals[:num_elite]
            
            # Random selection from remaining individuals
            if num_random > 0 and len(valid_individuals) > num_elite:
                remaining = valid_individuals[num_elite:]
                if len(remaining) <= num_random:
                    # Take all remaining if not enough
                    selected.extend(remaining)
                else:
                    # Random selection from remaining
                    selected.extend(random.sample(remaining, num_random))
            
            self._individuals = selected
        
        # Update best individual
        valid_in_pop = [ind for ind in self._individuals if ind.score is not None and not np.isinf(ind.score)]
        if valid_in_pop:
            current_best = max(valid_in_pop, key=lambda x: x.score)
            if current_best.score > self._best_score:
                self._best_score = current_best.score
                self._best_individual = copy.deepcopy(current_best) 