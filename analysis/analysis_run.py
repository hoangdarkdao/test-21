# analysis/run_analysis.py

from analysis.IGD import plot_igd, build_reference_pf
from analysis.HV import calculate_hv_progression
from plot_pareto_front import compare_pareto_from_algorithms
from utils import read_json


def run_analysis(
    metric,
    problem,
    working_dir=None,
    igd_ylim=None,          # 👈 NEW
):
    """
    Run analysis for a given metric and problem.

    metric: "igd" | "hv" | "pareto"
    problem: key in analysis_problem.json
    working_dir: optional experiment subfolder
    igd_ylim: tuple (ymin, ymax) or None
    """
    config = read_json("analysis/analysis_problem.json")
    algorithms = config[problem]

    if metric == "hv":
        calculate_hv_progression(
            algorithms,
            batch_size=10,
            visualize=True,
            max_samples=300,
            print_detail=True,
        )

    elif metric == "igd":
        all_paths = [p for paths in algorithms.values() for p in paths]
        ref_pf = build_reference_pf(all_paths)

        plot_igd(
            algorithms,
            ref_pf,
            max_eval=300,
            step=10,
            ylim=igd_ylim,    # 👈 PASS THROUGH
        )

    elif metric == "pareto":
        compare_pareto_from_algorithms(
            algorithms,
            show_global=True,
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    run_analysis(
        metric="hv",
        problem="bi_kp",
        working_dir="nhv_runtime_20",
        igd_ylim=(0.0, 15.0)
    )
