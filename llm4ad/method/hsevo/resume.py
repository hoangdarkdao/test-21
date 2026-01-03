from __future__ import annotations

import json
import os
import glob
from typing import Dict, List, Optional, Tuple

from .population import HSEvoPopulation
from ...base import Function, TextFunctionProgramConverter


class HSEvoResume:
    """Resume functionality for HSEvo experiments."""
    
    def __init__(self, log_dir: str):
        """Initialize resume functionality.
        
        Args:
            log_dir: Directory containing experiment logs
        """
        self._log_dir = log_dir
        self._population_restored = False
        self._reflection_restored = False
        
    def can_resume(self) -> bool:
        """Check if experiment can be resumed.
        
        Returns:
            True if resume is possible
        """
        # Check for required files
        required_files = [
            'samples_*.json',
            'population_stats.jsonl'
        ]
        
        for pattern in required_files:
            if not glob.glob(os.path.join(self._log_dir, pattern)):
                return False
        
        return True
    
    def restore_population(self, population: HSEvoPopulation, template_program: str) -> int:
        """Restore population from saved logs.
        
        Args:
            population: Population to restore
            template_program: Template program for creating functions
            
        Returns:
            Number of samples restored
        """
        if self._population_restored:
            return 0
        
        # Find all sample files
        sample_files = sorted(glob.glob(os.path.join(self._log_dir, 'samples_*.json')))
        
        total_samples = 0
        restored_functions = []
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r') as f:
                    samples = json.load(f)
                
                for sample in samples:
                    function_str = sample.get('function', '')
                    score = sample.get('score')
                    sample_order = sample.get('sample_order', 0)
                    
                    if function_str and score is not None:
                        # Convert string back to Function
                        function = TextFunctionProgramConverter.text_to_function(function_str)
                        if function is not None:
                            function.score = score
                            # Try to restore algorithm description from algorithms.jsonl
                            algorithm = self._restore_algorithm_description(sample_order)
                            if algorithm:
                                function.algorithm = algorithm
                            
                            restored_functions.append(function)
                            total_samples = max(total_samples, sample_order)
                
            except Exception as e:
                print(f"Warning: Could not restore from {sample_file}: {e}")
                continue
        
        # Add restored functions to population
        for function in restored_functions:
            population.register_function(function)
        
        self._population_restored = True
        return total_samples
    
    def restore_reflections(self) -> Tuple[List[str], List[str], str]:
        """Restore reflection history.
        
        Returns:
            Tuple of (good_experiences, bad_experiences, comprehensive_memory)
        """
        if self._reflection_restored:
            return [], [], ""
        
        good_experiences = []
        bad_experiences = []
        comprehensive_memory = ""
        
        # Restore from reflections.jsonl
        reflections_file = os.path.join(self._log_dir, 'reflections.jsonl')
        if os.path.exists(reflections_file):
            try:
                with open(reflections_file, 'r') as f:
                    for line in f:
                        reflection = json.loads(line.strip())
                        
                        if reflection['type'] == 'flash' and reflection['successful']:
                            content = reflection['content']
                            # Parse flash reflection to extract experience
                            experience = self._extract_experience_from_flash(content)
                            if experience:
                                # Determine if it was good or bad based on success
                                # This is a heuristic - could be improved with more context
                                if 'improve' in experience.lower() or 'better' in experience.lower():
                                    good_experiences.append(experience)
                                else:
                                    bad_experiences.append(experience)
                        
                        elif reflection['type'] == 'comprehensive':
                            comprehensive_memory = reflection['content']
                
            except Exception as e:
                print(f"Warning: Could not restore reflections: {e}")
        
        self._reflection_restored = True
        return good_experiences, bad_experiences, comprehensive_memory
    
    def restore_harmony_search_state(self) -> Dict:
        """Restore harmony search state.
        
        Returns:
            Dictionary containing harmony search state
        """
        hs_state = {
            'iterations': 0,
            'best_parameters': {},
            'best_score': float('-inf'),
            'parameter_history': []
        }
        
        hs_file = os.path.join(self._log_dir, 'harmony_search.jsonl')
        if os.path.exists(hs_file):
            try:
                with open(hs_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        hs_state['iterations'] = max(hs_state['iterations'], entry['iteration'])
                        hs_state['parameter_history'].append(entry)
                        
                        if entry['best_score'] > hs_state['best_score']:
                            hs_state['best_score'] = entry['best_score']
                            hs_state['best_parameters'] = entry['parameters']
                
            except Exception as e:
                print(f"Warning: Could not restore harmony search state: {e}")
        
        return hs_state
    
    def restore_experiment_statistics(self) -> Dict:
        """Restore experiment statistics.
        
        Returns:
            Dictionary containing experiment statistics
        """
        stats = {
            'generations': 0,
            'crossovers': 0,
            'mutations': 0,
            'successful_crossovers': 0,
            'successful_mutations': 0,
            'population_history': []
        }
        
        # Restore population statistics
        pop_stats_file = os.path.join(self._log_dir, 'population_stats.jsonl')
        if os.path.exists(pop_stats_file):
            try:
                with open(pop_stats_file, 'r') as f:
                    for line in f:
                        pop_stat = json.loads(line.strip())
                        stats['population_history'].append(pop_stat)
                        stats['generations'] = max(stats['generations'], pop_stat.get('generation', 0))
                
            except Exception as e:
                print(f"Warning: Could not restore population statistics: {e}")
        
        # Restore crossover statistics
        crossover_file = os.path.join(self._log_dir, 'crossover.jsonl')
        if os.path.exists(crossover_file):
            try:
                with open(crossover_file, 'r') as f:
                    for line in f:
                        crossover = json.loads(line.strip())
                        stats['crossovers'] += 1
                        if crossover.get('successful', False):
                            stats['successful_crossovers'] += 1
                
            except Exception as e:
                print(f"Warning: Could not restore crossover statistics: {e}")
        
        # Restore mutation statistics
        mutation_file = os.path.join(self._log_dir, 'mutation.jsonl')
        if os.path.exists(mutation_file):
            try:
                with open(mutation_file, 'r') as f:
                    for line in f:
                        mutation = json.loads(line.strip())
                        stats['mutations'] += 1
                        if mutation.get('successful', False):
                            stats['successful_mutations'] += 1
                
            except Exception as e:
                print(f"Warning: Could not restore mutation statistics: {e}")
        
        return stats
    
    def _restore_algorithm_description(self, sample_order: int) -> Optional[str]:
        """Restore algorithm description for a sample.
        
        Args:
            sample_order: Sample order to find
            
        Returns:
            Algorithm description or None if not found
        """
        algorithms_file = os.path.join(self._log_dir, 'algorithms.jsonl')
        if not os.path.exists(algorithms_file):
            return None
        
        try:
            with open(algorithms_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get('sample_order') == sample_order:
                        return entry.get('algorithm', '')
        except Exception:
            pass
        
        return None
    
    def _extract_experience_from_flash(self, flash_content: str) -> Optional[str]:
        """Extract experience from flash reflection content.
        
        Args:
            flash_content: Flash reflection content
            
        Returns:
            Extracted experience or None
        """
        try:
            # Look for experience section in flash reflection
            lines = flash_content.split('\n')
            in_experience = False
            experience_lines = []
            
            for line in lines:
                if '**Experience:**' in line or 'Experience:' in line:
                    in_experience = True
                    continue
                elif in_experience and line.strip():
                    if line.startswith('**') or line.startswith('##'):
                        break
                    experience_lines.append(line.strip())
            
            if experience_lines:
                return ' '.join(experience_lines)
            
        except Exception:
            pass
        
        return None
    
    def get_resume_info(self) -> Dict:
        """Get comprehensive resume information.
        
        Returns:
            Dictionary containing all resume information
        """
        if not self.can_resume():
            return {'can_resume': False}
        
        # Get basic file information
        sample_files = glob.glob(os.path.join(self._log_dir, 'samples_*.json'))
        total_samples = 0
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r') as f:
                    samples = json.load(f)
                    for sample in samples:
                        total_samples = max(total_samples, sample.get('sample_order', 0))
            except Exception:
                continue
        
        stats = self.restore_experiment_statistics()
        hs_state = self.restore_harmony_search_state()
        
        return {
            'can_resume': True,
            'log_dir': self._log_dir,
            'total_samples': total_samples,
            'generations': stats['generations'],
            'crossovers': stats['crossovers'],
            'mutations': stats['mutations'],
            'harmony_search_iterations': hs_state['iterations'],
            'best_harmony_score': hs_state['best_score'],
            'sample_files': len(sample_files)
        } 