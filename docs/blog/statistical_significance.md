# Why Statistical Significance Still Matters for Guardrails

Synthetic guardrail benchmarks are convenient, but they can mask how brittle a
system is when exposed to real-world operating conditions. Our latest STM
experiments on PlanBench++ hit the 5% alert budget with perfect precision, yet
permutation $p$-values stubbornly hovered near random chance. The lesson: dense
calibration grids and polished metrics are not enough—you need field data.

To close the gap we are:

- Partnering with operators who can share anonymised failure traces from
  robotics, logistics, and orchestration systems.
- Injecting causal features (irreversibility, commitment ratios, divergence
  rates) directly into the STM foreground windows to expose early signs of
  cascade failures.
- Shipping a Streamlit dashboard that visualises lead time, alert coverage, and
  ROI so stakeholders can witness the guardrail in action.

We invite researchers and practitioners to contribute real datasets or baseline
comparisons. If you're wrestling with high-stakes planning failures—especially
those with long lead times—let's collaborate on an evaluation that moves beyond
synthetic traces. Drop us a note at alex@sepdynamics.com.

