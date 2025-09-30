
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure sep_text_manifold package is importable when tests run directly
ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC = ROOT / 'score' / 'src'
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from score.scripts import logistics_guardrail_demo


def test_generate_demo_with_twin_payload(tmp_path):
    output_root = tmp_path / 'demo'
    twin_state = ROOT / 'score' / 'output' / 'planbench_demo_full' / 'gold' / 'states' / 'logistics_valid_01_state.json'
    assert twin_state.exists(), 'Expected sample twin state to exist for demo'

    args = SimpleNamespace(
        output_root=output_root,
        window_bytes=256,
        stride=128,
        verbose=False,
        twin_state=twin_state,
        twin_top_k=2,
        twin_max_distance=0.3,
        twin_match_signature=True,
        twin_keyword_limit=4,
    )

    summary = logistics_guardrail_demo.generate_demo(args)

    timeline_path = output_root / 'timeline.json'
    assert timeline_path.exists()
    timeline = json.loads(timeline_path.read_text())

    signal_summary = timeline['signal_summary']
    assert signal_summary['lead_time'] == 3
    assert summary['lead_time'] == 3
    assert summary['first_alert'] == 5
    assert summary['first_failure'] == 8

    twin_payload = timeline.get('twin', {})
    suggestions = twin_payload.get('suggestions', [])
    assert suggestions, 'Twin suggestions should be present when twin_state supplied'
    top = suggestions[0]
    assert 'keywords' in top and top['keywords']
    assert summary['twin_suggestions'], 'Summary should surface twin suggestions'

    dashboard_path = output_root / 'dashboard.html'
    assert dashboard_path.exists()
