from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any

from .population import HSEvoPopulation
from ...base import Function
from ...tools.profiler import ProfilerBase

class HSEvoProfiler(ProfilerBase):
    """HSEvo profiler for logging and monitoring."""
    
    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 evaluation_name='Problem',
                 method_name='HSEvo',
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """Initialize HSEvo profiler.
        
        Args:
            log_dir: Directory for logging
            evaluation_name: Name of the evaluation problem
            method_name: Name of the method
            initial_num_samples: Initial sample count
            log_style: Logging style ('simple' or 'complex')
            create_random_path: Whether to create random path
            **kwargs: Additional arguments
        """
        super().__init__(
            log_dir=log_dir,
            evaluation_name=evaluation_name,
            method_name=method_name,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )
        
        # HSEvo specific logging
        self._harmony_search_logs = []
        self._reflection_logs = []
        self._crossover_logs = []
        self._mutation_logs = []
        self._generation_stats = []
        
    def register_function(self, function: Function, *, resume_mode=False):
        """Register a function with HSEvo-specific logging.
        
        Args:
            function: Function to register
            resume_mode: Whether in resume mode
        """
        super().register_function(function, resume_mode=resume_mode)
        
        # Log function details
        if hasattr(function, 'algorithm') and function.algorithm:
            self._log_algorithm_thought(function)
    
    def register_population(self, population: HSEvoPopulation):
        """Register population statistics.
        
        Args:
            population: Current population
        """
        stats = population.get_statistics()
        self._generation_stats.append(stats)
        
        # Log population statistics
        if self._log_dir:
            self._write_population_stats(stats)
    
    def log_harmony_search(self, iteration: int, parameters: Dict, best_score: float):
        """Log harmony search iteration.
        
        Args:
            iteration: Harmony search iteration
            parameters: Current parameters
            best_score: Best score found
        """
        log_entry = {
            'iteration': iteration,
            'parameters': parameters,
            'best_score': best_score,
            'sample_order': self._num_samples
        }
        self._harmony_search_logs.append(log_entry)
        
        if self._log_dir:
            self._write_harmony_search_log(log_entry)
    
    def log_reflection(self, reflection_type: str, content: str, successful: bool = True):
        """Log reflection content.
        
        Args:
            reflection_type: Type of reflection ('flash' or 'comprehensive')
            content: Reflection content
            successful: Whether reflection was successful
        """
        log_entry = {
            'type': reflection_type,
            'content': content,
            'successful': successful,
            'sample_order': self._num_samples,
            'generation': len(self._generation_stats)
        }
        self._reflection_logs.append(log_entry)
        
        if self._log_dir:
            self._write_reflection_log(log_entry)
    
    def log_crossover(self, parent1_score: float, parent2_score: float, 
                     offspring_score: Optional[float]):
        """Log crossover operation.
        
        Args:
            parent1_score: Score of first parent
            parent2_score: Score of second parent
            offspring_score: Score of offspring (None if invalid)
        """
        log_entry = {
            'parent1_score': parent1_score,
            'parent2_score': parent2_score,
            'offspring_score': offspring_score,
            'successful': offspring_score is not None,
            'sample_order': self._num_samples,
            'generation': len(self._generation_stats)
        }
        self._crossover_logs.append(log_entry)
        
        if self._log_dir:
            self._write_crossover_log(log_entry)
    
    def log_mutation(self, elite_score: float, offspring_score: Optional[float]):
        """Log mutation operation.
        
        Args:
            elite_score: Score of elite individual
            offspring_score: Score of offspring (None if invalid)
        """
        log_entry = {
            'elite_score': elite_score,
            'offspring_score': offspring_score,
            'successful': offspring_score is not None,
            'sample_order': self._num_samples,
            'generation': len(self._generation_stats)
        }
        self._mutation_logs.append(log_entry)
        
        if self._log_dir:
            self._write_mutation_log(log_entry)
    
    def _log_algorithm_thought(self, function: Function):
        """Log algorithm thought/description.
        
        Args:
            function: Function with algorithm description
        """
        if not self._log_dir:
            return
            
        log_entry = {
            'sample_order': self._num_samples,
            'algorithm': function.algorithm,
            'score': function.score,
            'generation': len(self._generation_stats)
        }
        
        algorithms_file = os.path.join(self._log_dir, 'algorithms.jsonl')
        with open(algorithms_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _write_population_stats(self, stats: Dict):
        """Write population statistics to file.
        
        Args:
            stats: Population statistics
        """
        stats_file = os.path.join(self._log_dir, 'population_stats.jsonl')
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
    
    def _write_harmony_search_log(self, log_entry: Dict):
        """Write harmony search log entry.
        
        Args:
            log_entry: Log entry to write
        """
        hs_file = os.path.join(self._log_dir, 'harmony_search.jsonl')
        with open(hs_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _write_reflection_log(self, log_entry: Dict):
        """Write reflection log entry.
        
        Args:
            log_entry: Log entry to write
        """
        reflection_file = os.path.join(self._log_dir, 'reflections.jsonl')
        with open(reflection_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _write_crossover_log(self, log_entry: Dict):
        """Write crossover log entry.
        
        Args:
            log_entry: Log entry to write
        """
        crossover_file = os.path.join(self._log_dir, 'crossover.jsonl')
        with open(crossover_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _write_mutation_log(self, log_entry: Dict):
        """Write mutation log entry.
        
        Args:
            log_entry: Log entry to write
        """
        mutation_file = os.path.join(self._log_dir, 'mutation.jsonl')
        with open(mutation_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': self._num_samples,
            'successful_samples': self._evaluate_success_program_num,
            'failed_samples': self._evaluate_failed_program_num,
            'best_score': self._cur_best_program_score,
            'generations': len(self._generation_stats),
            'harmony_search_iterations': len(self._harmony_search_logs),
            'reflections': len(self._reflection_logs),
            'crossovers': len(self._crossover_logs),
            'mutations': len(self._mutation_logs)
        }
        
        # Add success rates
        total_crossovers = len(self._crossover_logs)
        successful_crossovers = sum(1 for log in self._crossover_logs if log['successful'])
        stats['crossover_success_rate'] = successful_crossovers / total_crossovers if total_crossovers > 0 else 0
        
        total_mutations = len(self._mutation_logs)
        successful_mutations = sum(1 for log in self._mutation_logs if log['successful'])
        stats['mutation_success_rate'] = successful_mutations / total_mutations if total_mutations > 0 else 0
        
        return stats
    
    def finish(self):
        """Finish profiling and write final statistics."""
        if self._log_dir:
            # Write final statistics
            final_stats = self.get_statistics()
            stats_file = os.path.join(self._log_dir, 'final_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
        
        super().finish()


