from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation
from llm4ad.task.optimization.bi_kp import BIKPEvaluation
from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation
from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation

from llm4ad.tools.llm.llm_api_codestral import MistralApi

from llm4ad.method.momcts import MOMCTS_AHD, MOMCTSProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
from llm4ad.method.hsevo import HSEvo, HSEvoProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
from llm4ad.method.mpage import MPaGEProfiler, MPaGE
from llm4ad.method.moead import MOEAD, MOEADProfiler
import os
from dotenv import load_dotenv

load_dotenv()

algorithm_map = {
    'momcts': (MOMCTS_AHD, MOMCTSProfiler),
    'meoh': (MEoH, MEoHProfiler),
    'eoh': (EoH, EoHProfiler),
    'reevo': (ReEvo, ReEvoProfiler),
    'hsevo': (HSEvo, HSEvoProfiler),
    'nsga2': (NSGA2, NSGA2Profiler),
    'mpage': (MPaGE, MPaGEProfiler),
    'moead': (MOEAD, MOEADProfiler)
}

task_map = {
    "tsp_semo": BITSPEvaluation(),
    "bi_kp": BIKPEvaluation(),
    "bi_cvrp": BICVRPEvaluation(),
    "tri_tsp": TRITSPEvaluation()
}

# Change variable here
ALGORITHM_NAME = 'reevo'  # Could also be 'MEoH' or 'NSGA2'
PROBLEM_NAME = "tsp_semo" # Could also be "tsp_semo, bi_kp, bi_cvrp"
exact_log_dir_name = "nhv_runtime_50/v1" # must be unique here
api_key = os.getenv('API_KEY1') # change APIKEY1, APIKEY2, APIKEY3

if __name__ == '__main__':
    
    log_dir = f'logs/{ALGORITHM_NAME}/{PROBLEM_NAME}'
    MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]
    TaskClass = task_map[PROBLEM_NAME]
    
    llm = MistralApi(
        keys=api_key,
        model='codestral-latest',
        timeout=60
    )
    
    task = TaskClass 
    method = MethodClass(
        llm=llm,
        llm_cluster=llm,
        profiler=ProfilerClass(log_dir=log_dir, log_style='complex', result_folder = exact_log_dir_name),
        evaluation=task,
        max_sample_nums=305, # max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
        max_generations=31,
        pop_size=10, # 20
        num_samplers=4,
        num_evaluators=4,
        selection_num=2     
        )
    method.run()