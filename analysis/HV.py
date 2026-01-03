import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from utils import read_score_from_path
from plot_style import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def calculate_hv_progression(
    algorithms,
    batch_size=10,
    visualize=True,
    max_samples=300,
    print_detail=True,
):
    """
    Calculate and optionally visualize Hypervolume (HV) progression.

    Keeps original logic:
    - zero_to_one normalization
    - fixed ideal / nadir
    - batch-wise prefix HV
    """

    # -----------------------------
    # Step 1: collect all data
    # -----------------------------
    all_F_global = []
    for files in algorithms.values():
        for file_path in files:
            F = read_score_from_path(file_path)
            if len(F) > 0:
                all_F_global.append(F[:max_samples])

    if not all_F_global:
        raise ValueError("No valid data found in any file.")

    all_F_global = np.vstack(all_F_global)

    # ⚠️ Keep your original bounds
    z_ideal = [-1.5, 0]
    z_nadir = [0, 20]
    ref_point = [1.1, 1.1]

    if print_detail:
        print(f"\n🌍 Global Ideal: {z_ideal}, Nadir: {z_nadir}")

    if visualize:
        plt.figure(figsize=(7, 5))

    # -----------------------------
    # Step 2: per-algorithm HV
    # -----------------------------
    for algo, files in algorithms.items():
        hv_runs = []

        for file_path in files:
            F = read_score_from_path(file_path)
            if len(F) == 0:
                continue

            F = np.asarray(F[:max_samples], dtype=float)

            metric = Hypervolume(
                ref_point=ref_point,
                norm_ref_point=False,
                zero_to_one=True,
                ideal=z_ideal,
                nadir=z_nadir,
            )

            hv_values = []
            for end in range(batch_size, len(F) + 1, batch_size):
                hv_values.append(metric(F[:end]))

            if len(hv_values) == 0:
                continue

            # pad to expected length (KEEPING YOUR LOGIC)
            expected_len = max_samples // batch_size
            if len(hv_values) < expected_len:
                hv_values += [hv_values[-1]] * (expected_len - len(hv_values))

            hv_runs.append(hv_values)

        if not hv_runs:
            continue

        # -----------------------------
        # Step 3: align & aggregate
        # -----------------------------
        max_len = max(len(run) for run in hv_runs)
        hv_array = np.full((len(hv_runs), max_len), np.nan)

        for i, run in enumerate(hv_runs):
            hv_array[i, :len(run)] = run

        mean_hv = np.nanmean(hv_array, axis=0)
        std_hv = np.nanstd(hv_array, axis=0)
        batches = np.arange(1, max_len + 1) * batch_size

        if print_detail:
            print(f"✅ Final HV ({algo}): {mean_hv[-1]:.4f} ± {std_hv[-1]:.4f}")

        # -----------------------------
        # Step 4: plot
        # -----------------------------
        if visualize:
            style = PAPER_STYLES.get(algo, {}).copy()

            plt.plot(
                batches,
                mean_hv,
                label=algo,
                linewidth=1.8,
                **style,          # ✅ unpack style dict
            )

            plt.fill_between(
                batches,
                mean_hv - std_hv,
                mean_hv + std_hv,
                alpha=STD_ALPHA,
                color=style.get("color", None),  # ✅ extract only color here
            )

    # -----------------------------
    # Step 5: finalize plot
    # -----------------------------
    if visualize:
        plt.xlabel("Function evaluations", fontsize=16)
        plt.title(r"HV $\uparrow$", fontsize=18)    # ← TITLE to

        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        leg = plt.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),   # ✅ INSIDE lower-right
            fontsize=14,                   # big, paper-readable
            frameon=True,
            framealpha=0.95,
            handlelength=3.0,
            handletextpad=0.8,
            borderpad=0.8,
            markerscale=1.5,
        )

        
        ax = plt.gca()

        # Force y-axis to start at 0.0
        ax.set_ylim(bottom=0.5)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        # Major ticks every 0.2
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # Format tick labels nicely
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # make legend lines thick (important)
        for line in leg.get_lines():
            line.set_linewidth(2.5)

        # optional: crisp frame
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.8)

        

        plt.tight_layout()
        plt.show()
