import json
from pathlib import Path

from sep_text_manifold.pipeline import analyse_directory
from sep_text_manifold import cli


def _build_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text("Alpha beta alpha", encoding="utf-8")
    (corpus_dir / "doc2.md").write_text("Gamma beta delta", encoding="utf-8")
    return corpus_dir


def test_analyse_directory_basic(tmp_path):
    corpus = _build_corpus(tmp_path)
    result = analyse_directory(str(corpus), window_bytes=64, stride=32)

    assert Path(result.settings.directory) == corpus.resolve()
    assert len(result.files) == 2
    assert result.token_count == 6
    assert "alpha" in result.string_scores

    alpha_profile = result.string_scores["alpha"]
    assert alpha_profile["occurrences"] == 2
    assert 0.0 <= alpha_profile.get("patternability", 0.0) <= 1.0

    dilution = result.dilution_summary
    assert 0.0 <= dilution["path_mean"] <= 1.0
    assert 0.0 <= dilution["signal_mean"] <= 1.0
    assert 0.0 <= dilution["semantic_dilution"] <= 1.0

    summary = result.summary(top=2)
    assert summary["string_count"] >= 4
    assert len(summary["top_patternable_strings"]) == 2
    assert summary["theme_count"] >= 1
    assert "dilution" in summary


def test_cli_ingest_and_summary(tmp_path, monkeypatch):
    corpus = _build_corpus(tmp_path)
    state_path = tmp_path / "stm_state.json"
    monkeypatch.chdir(tmp_path)

    cli.main([
        "ingest",
        str(corpus),
        "--window-bytes",
        "64",
        "--stride",
        "32",
        "--output",
        str(state_path),
        "--summary-top",
        "2",
        "--min-token-len",
        "3",
        "--alpha-ratio",
        "0.4",
        "--drop-numeric",
        "--min-occ",
        "1",
        "--cap-tokens-per-win",
        "40",
        "--graph-min-pmi",
        "0.0",
        "--theme-min-size",
        "1",
        "--store-signals",
    ])

    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "string_scores" in state
    assert state["summary"]["string_count"] == len(state["string_scores"])
    assert "dilution_summary" in state

    cli.main([
        "summary",
        "--input",
        str(state_path),
        "--top",
        "1",
    ])

    cli.main([
        "strings",
        "--input",
        str(state_path),
        "--filter",
        "coh>=0.0",
        "--top",
        "1",
    ])

    cli.main([
        "strings",
        "--input",
        str(state_path),
        "--filter",
        "coh>=P0,ent<=P100",
        "--top",
        "1",
    ])

    cli.main([
        "dilution",
        "--input",
        str(state_path),
        "--top",
        "2",
    ])
