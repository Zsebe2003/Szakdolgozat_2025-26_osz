# main_inductive.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pm4py

from src.utils.paths import Paths

# PM4Py 2.7+ kompatibilis metrikák
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_eval
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_eval

from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

# --- Petri export helper (PNML + PNG) ---
from pathlib import Path
import pm4py

def save_petri_artifacts(net, im, fm, out_dir: Path, basename: str):
    """
    Ment: PNML (mindig) + PNG (ha van Graphviz).
    out_dir/basename.pnml
    out_dir/basename.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pnml_path = out_dir / f"{basename}.pnml"
    png_path = out_dir / f"{basename}.png"

    # PNML (robosztus, több PM4Py variánst próbálunk)
    try:
        # újabb, kényelmi wrapper
        pm4py.write_pnml(net, im, fm, str(pnml_path))
    except Exception:
        try:
            # klasszikus exporter
            from pm4py.objects.petri.exporter import pnml as pnml_exporter
            pnml_exporter.export_net(net, im, fm, str(pnml_path))
        except Exception as e:
            print(f"⚠️ PNML export failed: {e}")

    # PNG (ha van Graphviz)
    try:
        gviz = pm4py.visualization.petri_net.visualizer.apply(net, im, fm)
        pm4py.visualization.petri_net.visualizer.save(gviz, str(png_path))
    except Exception as e:
        print(f"⚠️ PNG render failed (Graphviz?): {e}")

    print(f"💾 Petri saved → {pnml_path}")
    if png_path.exists():
        print(f"🖼  Petri image → {png_path}")



def compute_metrics(log, net, im, fm) -> dict:
    # Fitness (token-based)
    try:
        fit = fitness_eval.apply(log, net, im, fm, variant=fitness_eval.Variants.TOKEN_BASED)
        fitness = float(fit.get("log_fitness", fit.get("average_trace_fitness", 0.0)))
    except Exception:
        # Fallback: token replay list átlag
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        rep = token_replay.apply(log, net, im, fm)
        vals = []
        for r in rep:
            if isinstance(r, dict):
                if "trace_fitness" in r and isinstance(r["trace_fitness"], dict):
                    vals.append(r["trace_fitness"].get("fitness", 0.0))
                elif "fitness" in r:
                    vals.append(float(r["fitness"]))
        fitness = float(sum(vals) / len(vals)) if vals else 0.0

    # Precision (ETConformance-token)
    try:
        precision = float(precision_eval.apply(log, net, im, fm, variant=precision_eval.Variants.ETCONFORMANCE_TOKEN))
    except Exception:
        precision = 0.0

    # Generalization
    try:
        generalization = float(generalization_eval.apply(log, net, im, fm))
    except Exception:
        # Fallback: aktivált tranzíciók aránya
        from pm4py.algo.conformance.tokenreplay import algorithm as token_rep
        rep = token_rep.apply(log, net, im, fm)
        activated = set()
        for tr in rep:
            if isinstance(tr, dict) and "activated_transitions" in tr:
                activated.update(tr["activated_transitions"])
        all_tr = {t.label for t in net.transitions if t.label}
        generalization = float(len(activated) / len(all_tr)) if all_tr else 0.5

    # Simplicity
    try:
        simplicity = float(simplicity_eval.apply(net))
    except Exception:
        num_arcs = len(net.arcs)
        denom = len(net.places) + len(net.transitions)
        simplicity = 1.0 / (1.0 + (num_arcs / denom)) if denom else 0.5

    # F-score (fitness & precision)
    f_score = (2 * fitness * precision / (fitness + precision)) if (fitness > 0 and precision > 0) else 0.0

    return {
        "fitness": fitness,
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity,
        "f_score": f_score,
        "model_info": {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
            "visible_transitions": len([t for t in net.transitions if t.label]),
            "silent_transitions": len([t for t in net.transitions if not t.label]),
        },
    }


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    ap = argparse.ArgumentParser(description="Inductive Miner on an XES log (metrics only, no viz).")
    ap.add_argument("--xes", type=str, default=None,
                    help="XES path (default: data/xes/event_log_remaining_ALL.xes)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir for metrics (default: figures/inductive)")
    args = ap.parse_args()

    xes_path = Path(args.xes) if args.xes else (p.xes / "event_log_remaining_ALL.xes")
    if not xes_path.exists():
        raise SystemExit(f"XES not found: {xes_path}. Generate it first (python main_pm4py.py).")

    log = pm4py.read_xes(str(xes_path))
    print(f"📥 Log: {xes_path} | traces={len(log)} | events={sum(len(tr) for tr in log)}")

    # Inductive Miner → process tree → Petri-net
    pt = inductive_miner.apply(log)                 # process tree
    net, im, fm = pt_converter.apply(pt)            # convert to Petri

    # Petri-háló export (PNML + PNG)
    out_dir = Path(args.out) if args.out else (p.figures / "inductive")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_petri_artifacts(net, im, fm, out_dir, basename="inductive_petri")

    # Metrics
    metrics = compute_metrics(log, net, im, fm)

    out_dir = Path(args.out) if args.out else (p.figures / "inductive")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON + CSV
    json_path = out_dir / "inductive_metrics.json"
    csv_path = out_dir / "inductive_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"xes_path": str(xes_path), **metrics}, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            if k == "model_info":
                continue
            f.write(f"{k},{v}\n")

    print(" Inductive Miner done.")
    print(f"   Fitness:        {metrics['fitness']:.4f}")
    print(f"   Precision:      {metrics['precision']:.4f}")
    print(f"   F-score:        {metrics['f_score']:.4f}")
    print(f"   Generalization: {metrics['generalization']:.4f}")
    print(f"   Simplicity:     {metrics['simplicity']:.4f}")
    print(f"   Model: places={metrics['model_info']['places']}, "
          f"transitions={metrics['model_info']['transitions']}, arcs={metrics['model_info']['arcs']}")
    print(f"💾 Saved: {json_path}\n💾 Saved: {csv_path}")


if __name__ == "__main__":
    main()

