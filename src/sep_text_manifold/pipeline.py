from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .ingest import ingest_directory
from .manifold import build_manifold
from .scoring import connector_score, patternability_score
from .strings import StringOccurrence, aggregate_string_metrics, extract_strings
from .themes import build_theme_graph, compute_graph_metrics, detect_themes


@dataclass
class AnalysisSettings:
    directory: str
    window_bytes: int
    stride: int
    extensions: Optional[List[str]]


@dataclass
class FileDigest:
    file_id: str
    path: str
    byte_count: int
    char_count: int
    token_count: int


@dataclass
class AnalysisResult:
    settings: AnalysisSettings
    files: List[FileDigest]
    occurrences: List[StringOccurrence]
    signals: List[Dict[str, Any]]
    string_profiles: Dict[str, Dict[str, Any]]
    string_scores: Dict[str, Dict[str, Any]]
    graph_metrics: Dict[str, Dict[str, float]]
    themes: List[List[str]]

    @property
    def corpus_size_bytes(self) -> int:
        return sum(f.byte_count for f in self.files)

    @property
    def token_count(self) -> int:
        return sum(f.token_count for f in self.files)

    def to_state(
        self,
        *,
        include_signals: bool = False,
        include_occurrences: bool = False,
        include_profiles: bool = False,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "settings": asdict(self.settings),
            "files": [asdict(f) for f in self.files],
            "corpus_size_bytes": self.corpus_size_bytes,
            "token_count": self.token_count,
            "string_scores": self.string_scores,
            "themes": self.themes,
            "graph_metrics": self.graph_metrics,
        }
        if include_signals:
            state["signals"] = self.signals
        if include_occurrences:
            state["occurrences"] = [asdict(o) for o in self.occurrences]
        if include_profiles:
            state["string_profiles"] = self.string_profiles
        return state

    def summary(self, *, top: int = 10) -> Dict[str, Any]:
        return compute_summary(
            self.string_scores,
            signals=self.signals,
            themes=self.themes,
            corpus_size_bytes=self.corpus_size_bytes,
            token_count=self.token_count,
            file_count=len(self.files),
            top=top,
        )


def analyse_directory(
    directory: str,
    *,
    window_bytes: int = 2048,
    stride: int = 1024,
    extensions: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> AnalysisResult:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory} is not a directory")
    tokenised_occurrences: List[StringOccurrence] = []
    files: List[FileDigest] = []
    corpus_bytes = bytearray()
    current_offset = 0
    ext_list: Optional[List[str]] = list(extensions) if extensions is not None else None
    for file_id, path, text in ingest_directory(str(root), extensions=ext_list):
        occs = extract_strings(text, file_id)
        for occ in occs:
            occ.byte_start += current_offset
            occ.byte_end += current_offset
        tokenised_occurrences.extend(occs)
        data_bytes = text.encode("utf-8")
        corpus_bytes.extend(data_bytes)
        corpus_bytes.append(0)
        if verbose:
            print(
                f"[stm] processed {file_id} ({len(data_bytes)} bytes, {len(occs)} tokens)",
                flush=True,
            )
        files.append(
            FileDigest(
                file_id=file_id,
                path=path,
                byte_count=len(data_bytes),
                char_count=len(text),
                token_count=len(occs),
            )
        )
        current_offset += len(data_bytes) + 1
    if verbose:
        print(f"[stm] building manifold ({len(corpus_bytes)} bytes)", flush=True)
    signals = build_manifold(bytes(corpus_bytes), window_bytes=window_bytes, stride=stride)
    if verbose:
        print(f"[stm] manifold windows: {len(signals)}", flush=True)
        print("[stm] aggregating string metrics", flush=True)
    string_profiles = aggregate_string_metrics(
        tokenised_occurrences,
        signals,
        window_bytes=window_bytes,
        stride=stride,
    )
    if verbose:
        print(f"[stm] strings with profiles: {len(string_profiles)}", flush=True)
    string_scores: Dict[str, Dict[str, Any]] = {}
    for s, profile in string_profiles.items():
        metrics = profile.get("metrics", {})
        p_score = patternability_score(
            metrics.get("coherence", 0.0),
            metrics.get("stability", 0.0),
            metrics.get("entropy", 0.0),
            metrics.get("rupture", 0.0),
        )
        entry: Dict[str, Any] = {
            "metrics": metrics,
            "occurrences": profile.get("occurrences", 0),
            "window_ids": profile.get("window_ids", []),
            "patternability": p_score,
        }
        for field, value in metrics.items():
            entry[field] = value
        string_scores[s] = entry
    if verbose:
        print("[stm] computing theme graph", flush=True)
    graph = build_theme_graph(
        {s: prof.get("window_ids", []) for s, prof in string_profiles.items()},
        max_members_per_window=80,
    )
    graph_metrics = compute_graph_metrics(graph)
    themes = [sorted(list(t)) for t in detect_themes(graph)]
    if verbose:
        print(f"[stm] themes detected: {len(themes)}", flush=True)
    for s, entry in string_scores.items():
        gm = graph_metrics.get(s, {})
        c_score = connector_score(
            gm.get("betweenness", 0.0),
            gm.get("bridging", 0.0),
            0.0,
            gm.get("theme_entropy_neighbors", 0.0),
            gm.get("redundant_degree", 0.0),
        )
        entry["connector"] = c_score
        entry["graph_metrics"] = gm
    settings = AnalysisSettings(
        directory=str(root.resolve()),
        window_bytes=window_bytes,
        stride=stride,
        extensions=ext_list,
    )
    return AnalysisResult(
        settings=settings,
        files=files,
        occurrences=tokenised_occurrences,
        signals=signals,
        string_profiles=string_profiles,
        string_scores=string_scores,
        graph_metrics=graph_metrics,
        themes=themes,
    )


def compute_summary(
    string_scores: Dict[str, Dict[str, Any]],
    *,
    signals: Optional[List[Dict[str, Any]]] = None,
    themes: Optional[List[Iterable[str]]] = None,
    corpus_size_bytes: Optional[int] = None,
    token_count: Optional[int] = None,
    file_count: Optional[int] = None,
    top: int = 10,
) -> Dict[str, Any]:
    if top <= 0:
        top = 10
    strings_sorted = sorted(
        ((s, data) for s, data in string_scores.items()),
        key=lambda item: item[1].get("patternability", 0.0),
        reverse=True,
    )
    connectors_sorted = sorted(
        ((s, data) for s, data in string_scores.items()),
        key=lambda item: item[1].get("connector", 0.0),
        reverse=True,
    )
    top_patterns = [
        {
            "string": s,
            "patternability": data.get("patternability", 0.0),
            "occurrences": data.get("occurrences", 0),
        }
        for s, data in strings_sorted[:top]
    ]
    top_connectors = [
        {
            "string": s,
            "connector": data.get("connector", 0.0),
            "occurrences": data.get("occurrences", 0),
        }
        for s, data in connectors_sorted[:top]
    ]
    window_count = len(signals) if signals is not None else None
    avg_window_metrics: Dict[str, float] = {}
    if signals:
        totals: Dict[str, float] = {}
        count = 0
        for sig in signals:
            metrics = sig.get("metrics", {})
            if not metrics:
                continue
            count += 1
            for field, value in metrics.items():
                totals[field] = totals.get(field, 0.0) + float(value)
        if count:
            avg_window_metrics = {name: value / count for name, value in totals.items()}
    summary: Dict[str, Any] = {
        "string_count": len(string_scores),
        "top_patternable_strings": top_patterns,
        "top_connectors": top_connectors,
        "theme_count": len(themes or []),
    }
    if corpus_size_bytes is not None:
        summary["corpus_size_bytes"] = corpus_size_bytes
    if token_count is not None:
        summary["token_count"] = token_count
    if file_count is not None:
        summary["file_count"] = file_count
    if window_count is not None:
        summary["window_count"] = window_count
    if avg_window_metrics:
        summary["mean_window_metrics"] = avg_window_metrics
    return summary
