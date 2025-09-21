#!/usr/bin/env python3
"""Render LaTeX tables for the STM whitepaper appendices."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTE_DIR = REPO_ROOT / "docs" / "note"
OUTPUT_DIR = REPO_ROOT / "docs" / "whitepaper"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_planbench_scorecard() -> None:
    df = pd.read_csv(NOTE_DIR / "planbench_scorecard.csv")
    df["lead_coverage_pct"] = df["lead_coverage"] * 100
    columns = [
        "domain",
        "n_traces",
        "plan_accuracy",
        "lead_mean",
        "lead_coverage_pct",
        "lead_max",
        "twin_rate@0.3",
        "twin_rate@0.4",
        "twin_rate@0.5",
        "aligned_mean",
        "aligned_min",
        "aligned_max",
        "lead_perm_pval",
    ]
    df = df[columns]
    with (OUTPUT_DIR / "planbench_scorecard.tex").open("w", encoding="utf-8") as fh:
        fh.write("\\toprule\n")
        fh.write(
            "Domain & $n$ & Accuracy & Lead Mean & Guardrail (\%) & Lead Max & Twin@0.3 & Twin@0.4 & Twin@0.5 & Aligned Mean & Aligned Min & Aligned Max & Permutation $p$\\\\\n"
        )
        fh.write("\\midrule\n")
        for _, row in df.iterrows():
            fh.write(
                "{domain} & {n} & {accuracy:.2f} & {lead_mean:.2f} & {coverage:.1f} & {lead_max:.0f} & {twin03:.2f} & {twin04:.2f} & {twin05:.2f} & {aligned_mean:.2f} & {aligned_min:.0f} & {aligned_max:.0f} & {pval:.2f}\\\\\n".format(
                    domain=row["domain"],
                    n=int(row["n_traces"]),
                    accuracy=row["plan_accuracy"],
                    lead_mean=row["lead_mean"],
                    coverage=row["lead_coverage_pct"],
                    lead_max=row["lead_max"],
                    twin03=row["twin_rate@0.3"],
                    twin04=row["twin_rate@0.4"],
                    twin05=row["twin_rate@0.5"],
                    aligned_mean=row["aligned_mean"],
                    aligned_min=row["aligned_min"],
                    aligned_max=row["aligned_max"],
                    pval=row["lead_perm_pval"],
                )
            )
        fh.write("\\bottomrule\n")


def write_guardrail_table() -> None:
    df = pd.read_csv(NOTE_DIR / "appendix_guardrail_sensitivity.csv")
    with (OUTPUT_DIR / "guardrail_sensitivity.tex").open("w", encoding="utf-8") as fh:
        fh.write("\\toprule\n")
        fh.write("Domain & Target & Observed & Lead Density & Notes\\\\\n")
        fh.write("\\midrule\n")
        for _, row in df.iterrows():
            fh.write(
                "{domain} & {target:.2f} & {observed:.3f} & {lead_density:.3f} & {notes}\\\\\n".format(
                    domain=row["domain"],
                    target=row["target_guardrail"],
                    observed=row["actual_coverage"],
                    lead_density=row["lead_density"],
                    notes=row["notes"],
                )
            )
        fh.write("\\bottomrule\n")


def write_tau_table() -> None:
    df = pd.read_csv(NOTE_DIR / "appendix_tau_sweep.csv")
    with (OUTPUT_DIR / "tau_sweep.tex").open("w", encoding="utf-8") as fh:
        fh.write("\\toprule\n")
        fh.write("Cohort & $\\tau$ & Twin Rate & Notes\\\\\n")
        fh.write("\\midrule\n")
        for _, row in df.iterrows():
            fh.write(
                "{cohort} & {tau:.2f} & {twin:.2f} & {notes}\\\\\n".format(
                    cohort=row["cohort"],
                    tau=row["tau"],
                    twin=row["twin_rate"],
                    notes=row["notes"],
                )
            )
        fh.write("\\bottomrule\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_planbench_scorecard()
    write_guardrail_table()
    write_tau_table()
    print("LaTeX tables written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
