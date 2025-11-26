# main_fuzzy.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
import pm4py

from src.utils.paths import Paths


def discover_fuzzy(log) -> Tuple[Any, Dict[str, Any]]:
    """
    Fuzzy Miner model felfedezése; több fallback-kel, hogy különböző PM4Py verziókon is fusson.
    Visszatér: (model, params_dict)
    """
    # 1) Újabb wrapper
    try:
        model, params = pm4py.discover_fuzzy_model(log)
        return model, (params or {})
    except Exception:
        pass

    # 2) Algoritmus API
    try:
        from pm4py.algo.discovery.fuzzy import algorithm as fuzzy_alg
        params = {}
        model = fuzzy_alg.apply(log, parameters=params)
        return model, params
    except Exception:
        pass

    # 3) Régi factory
    try:
        from pm4py.algo.discovery.fuzzy import factory as fuzzy_factory
        model = fuzzy_factory.apply(log)
        return model, {}
    except Exception as e:
        raise RuntimeError(f"Fuzzy Miner discovery not available in this PM4Py build: {e}")


def export_fuzzy_tables(model: Any, out_dir: Path) -> None:
    """
    Csomópont/él significance (és ha elérhető: correlation) export CSV-be.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Nodes ---
    nodes_csv = out_dir / "fuzzy_nodes.csv"
    rows = []
    try:
        nodes_obj = getattr(model, "nodes", None) or getattr(model, "graph", None)
        if nodes_obj and hasattr(nodes_obj, "items"):
            for k, v in nodes_obj.items():
                if isinstance(v, dict):
                    sig = v.get("significance", None)
                    corr = v.get("correlation", None)
                else:
                    sig = getattr(v, "significance", None)
                    corr = getattr(v, "correlation", None)
                rows.append((str(k), sig, corr))
        else:
            node_sig = getattr(model, "node_significance", {})
            if isinstance(node_sig, dict):
                for k, sig in node_sig.items():
                    rows.append((str(k), sig, None))
    finally:
        with nodes_csv.open("w", encoding="utf-8") as f:
            f.write("node,significance,correlation\n")
            for node, sig, corr in rows:
                f.write(f"{node},{'' if sig is None else sig},{'' if corr is None else corr}\n")

    # --- Edges ---
    edges_csv = out_dir / "fuzzy_edges.csv"
    rows = []
    try:
        edges_obj = getattr(model, "edges", None)
        if edges_obj and hasattr(edges_obj, "items"):
            for k, v in edges_obj.items():
                if isinstance(k, tuple) and len(k) == 2:
                    src, dst = k
                elif isinstance(k, str) and "->" in k:
                    src, dst = k.split("->", 1)
                else:
                    src, dst = str(k), ""
                if isinstance(v, dict):
                    sig = v.get("significance", None)
                    corr = v.get("correlation", None)
                else:
                    sig = getattr(v, "significance", None)
                    corr = getattr(v, "correlation", None)
                rows.append((str(src), str(dst), sig, corr))

        if not rows:
            graph = getattr(model, "graph", None)
            if isinstance(graph, dict):
                for src, outs in graph.items():
                    if isinstance(outs, dict):
                        for dst, w in outs.items():
                            if isinstance(w, dict):
                                sig = w.get("significance", None)
                                corr = w.get("correlation", None)
                            else:
                                sig, corr = w, None
                            rows.append((str(src), str(dst), sig, corr))
    finally:
        with edges_csv.open("w", encoding="utf-8") as f:
            f.write("source,target,significance,correlation\n")
            for s, t, sig, corr in rows:
                f.write(f"{s},{t},{'' if sig is None else sig},{'' if corr is None else corr}\n")


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    ap = argparse.ArgumentParser(description="Fuzzy Miner on an XES log (CSV táblák, vizualizáció nélkül).")
    ap.add_argument("--xes", type=str, default=None,
                    help="XES path (alapértelmezett: data/xes/event_log_remaining_ALL.xes)")
    ap.add_argument("--out", type=str, default=None,
                    help="Kimeneti mappa (alapértelmezett: figures/fuzzy)")
    args = ap.parse_args()

    xes_path = Path(args.xes) if args.xes else (p.xes / "event_log_remaining_ALL.xes")
    if not xes_path.exists():
        raise SystemExit(f"XES nem található: {xes_path}. Előbb generáld le (python main_pm4py.py).")

    log = pm4py.read_xes(str(xes_path))
    print(f"📥 Log: {xes_path} | traces={len(log)} | events={sum(len(tr) for tr in log)}")

    model, params = discover_fuzzy(log)

    out_dir = Path(args.out) if args.out else (p.figures / "fuzzy")
    out_dir.mkdir(parents=True, exist_ok=True)

    # paraméterek mentése (ha vannak)
    with (out_dir / "fuzzy_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    export_fuzzy_tables(model, out_dir)

    print("✅ Fuzzy Miner kész.")
    print(f"💾 CSV-k: {out_dir}")


if __name__ == "__main__":
    main()
