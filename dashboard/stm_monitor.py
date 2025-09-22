"""Streamlit dashboard for STM guardrail demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

try:  # pragma: no cover - optional dependency for interactive demo
    import streamlit as st
except ImportError:  # pragma: no cover - handled at runtime
    st = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERMUTATION = REPO_ROOT / "analysis" / "router_config_logistics_invalid_5pct.permutation.json"


def load_permutation_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Permutation summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class STMDashboard:
    """Render live metrics from STM permutation summaries."""

    def __init__(self, *, permutation_path: Path = DEFAULT_PERMUTATION) -> None:
        if st is None:
            raise RuntimeError("Streamlit is required for STMDashboard; install streamlit>=1.33")
        self.permutation_path = permutation_path
        self.summary = load_permutation_summary(permutation_path)

    @property
    def records(self) -> Sequence[Dict[str, Any]]:
        return self.summary.get("records", [])

    def render(self) -> None:  # pragma: no cover - UI code
        st.set_page_config(page_title="STM Guardrail Monitor", layout="wide")
        st.title("STM Planning Guardrail Monitor")

        col_left, col_right = st.columns((2, 1))
        with col_left:
            self.show_live_trace()
        with col_right:
            self.show_roi_metrics()

        st.markdown("---")
        self.show_intervention_ui()

    def show_live_trace(self) -> None:  # pragma: no cover - UI code
        st.subheader("Trace Outcomes")
        table = _records_table(self.records)
        st.dataframe(table, hide_index=True)
        chart_data = {
            "lead_steps": [row["lead"] for row in table],
            "coverage": [row["coverage"] for row in table],
        }
        st.line_chart(chart_data)

    def show_intervention_ui(self) -> None:  # pragma: no cover - UI code
        st.subheader("Alert Drill-down")
        trace_names = [record.get("trace", f"trace_{idx}") for idx, record in enumerate(self.records)]
        if not trace_names:
            st.info("No traces available in the permutation summary.")
            return
        selected = st.selectbox("Select trace", trace_names)
        record = next((rec for rec in self.records if rec.get("trace") == selected), None)
        if not record:
            st.warning("Trace not found in summary.")
            return

        lead = record.get("lead")
        st.metric("Lead steps", value=lead if lead is not None else "n/a")
        st.write("### Suggested Interventions")
        st.markdown(
            "- Review resource allocations prior to the failure step.\n"
            "- Compare with high-recall twin windows from Logistics corpus.\n"
            "- Capture operator decision for ROI attribution."
        )

    def show_roi_metrics(self) -> None:  # pragma: no cover - UI code
        st.subheader("ROI Snapshot")
        leads = [row.get("lead") for row in self.records if isinstance(row.get("lead"), (int, float))]
        coverage = [row.get("coverage") for row in self.records if isinstance(row.get("coverage"), (int, float))]
        alerts = sum(int(row.get("alerts", 0) or 0) for row in self.records)
        total_windows = sum(int(row.get("window_count", 0) or 0) for row in self.records)

        avg_lead = mean(leads) if leads else 0.0
        avg_coverage = mean(coverage) if coverage else 0.0
        coverage_pct = round(avg_coverage * 100.0, 2)
        lead_steps = round(avg_lead, 2)

        st.metric("Average lead", f"{lead_steps} steps")
        st.metric("Average coverage", f"{coverage_pct}%")
        st.metric("Alerts evaluated", alerts)
        if total_windows:
            roi_estimate = alerts / total_windows * lead_steps * 1000
            st.metric("Estimated ROI", f"${roi_estimate:,.0f}")
        else:
            st.metric("Estimated ROI", "n/a")


def _records_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for record in records:
        lead = record.get("lead")
        coverage = record.get("coverage")
        table.append(
            {
                "trace": record.get("trace", "unknown"),
                "lead": lead if lead is not None else 0,
                "coverage": coverage if coverage is not None else 0.0,
                "alerts": record.get("alerts", 0),
                "p_value": record.get("p_value", 1.0),
            }
        )
    return table


def main() -> None:  # pragma: no cover - CLI entry
    if st is None:
        raise SystemExit("Install streamlit>=1.33 to run the dashboard.")
    dashboard = STMDashboard()
    dashboard.render()


if __name__ == "__main__":  # pragma: no cover
    main()

