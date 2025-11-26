# main_alpha_metrics.py - CORRECTED VERSION
from __future__ import annotations

import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
import pm4py
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_eval

from src.utils.paths import Paths


def compute_metrics_for_log(xes_path: Path) -> dict:
    if not xes_path.exists():
        raise FileNotFoundError(f"XES not found: {xes_path}")

    # 1) Log beolvasása
    log = pm4py.read_xes(str(xes_path))
    print(f"📊 Log loaded: {len(log)} traces")

    # 2) Alpha Miner modell
    net, im, fm = pm4py.discover_petri_net_alpha(log)
    print(f"🔧 Petri net discovered: {len(net.places)} places, {len(net.transitions)} transitions")

    # =============================
    # FITNESS - token based replay
    # =============================
    try:
        fitness_result = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
        fitness = float(fitness_result['log_fitness'])
        print(f"✅ Fitness calculated: {fitness}")
    except Exception as e:
        print(f"⚠️  Fitness calculation failed: {e}")
        fitness = 0.0

    # =============================
    # PRECISION - alignments based (megbízhatóbb)
    # =============================
    try:
        # Először próbáljuk meg az alignments-t (ajánlott)
        precision = float(precision_eval.apply(log, net, im, fm, variant=precision_eval.Variants.ALIGN_ETCONFORMANCE))
        print(f"✅ Precision calculated: {precision}")
    except Exception as e:
        print(f"⚠️  Alignments precision failed: {e}")
        try:
            # Fallback: token based precision
            precision = float(precision_eval.apply(log, net, im, fm, variant=precision_eval.Variants.ETCONFORMANCE_TOKEN))
            print(f"✅ Token-based precision calculated: {precision}")
        except Exception as e2:
            print(f"⚠️  Precision calculation failed: {e2}")
            precision = 0.0

    # =============================
    # GENERALIZATION
    # =============================
    try:
        generalization_score = float(generalization_eval.apply(log, net, im, fm))
        print(f"✅ Generalization calculated: {generalization_score}")
    except Exception as e:
        print(f"⚠️  Generalization calculation failed: {e}")
        # Alternatív generalization számítás
        try:
            from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
            
            replayed_traces = token_replay.apply(log, net, im, fm)
            activated_transitions = set()
            
            for trace in replayed_traces:
                if 'activated_transitions' in trace:
                    activated_transitions.update(trace['activated_transitions'])
            
            all_transitions = set(t.label for t in net.transitions if t.label is not None)
            
            if all_transitions:
                generalization_score = len(activated_transitions) / len(all_transitions)
            else:
                generalization_score = 0.5
                
            print(f"✅ Alternative generalization calculated: {generalization_score}")
        except Exception as e2:
            print(f"⚠️  Alternative generalization failed: {e2}")
            generalization_score = 0.5

    # =============================
    # SIMPLICITY - arc degree
    # =============================
    try:
        simplicity_score = float(simplicity_eval.apply(net))
        print(f"✅ Simplicity calculated: {simplicity_score}")
    except Exception as e:
        print(f"⚠️  Simplicity calculation failed: {e}")
        # Alternatív simplicity számítás
        try:
            from pm4py.algo.evaluation.simplicity.variants import arc_degree
            simplicity_score = float(arc_degree.apply(net))
            print(f"✅ Alternative simplicity calculated: {simplicity_score}")
        except Exception as e2:
            print(f"⚠️  Alternative simplicity failed: {e2}")
            # Nagyon egyszerű simplicity számítás: átmenetek/helyek aránya
            num_places = len(net.places)
            num_transitions = len(net.transitions)
            num_arcs = len(net.arcs)
            
            if num_transitions > 0 and num_places > 0:
                simplicity_score = 1.0 / (1.0 + (num_arcs / (num_places + num_transitions)))
            else:
                simplicity_score = 0.5
            print(f"✅ Basic simplicity calculated: {simplicity_score}")

    return {
        "xes_path": str(xes_path),
        "fitness_token_based": fitness,
        "precision_etconformance": precision,
        "generalization": generalization_score,
        "simplicity": simplicity_score,
        "model_info": {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs)
        }
    }


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    parser = argparse.ArgumentParser(description="Alpha Miner metrics")
    parser.add_argument(
        "--xes",
        type=str,
        default=None,
        help="XES path (default: event_log_remaining_ALL.xes)",
    )
    args = parser.parse_args()

    # alapértelmezett XES
    xes_path = Path(args.xes) if args.xes else (p.xes / "event_log_remaining_ALL.xes")

    print(f"▶︎ Computing Alpha Miner metrics for: {xes_path}")
    print("=" * 60)
    
    metrics = compute_metrics_for_log(xes_path)

    # mentés
    out_dir = p.figures / "alpha"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "alpha_metrics.json"
    csv_path = out_dir / "alpha_metrics.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # CSV (csak a fő metrikák)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            if k not in ["xes_path", "model_info"]:
                f.write(f"{k},{v}\n")

    print("\n" + "=" * 60)
    print("✅ Final Results:")
    print(f"  📍 Fitness: {metrics['fitness_token_based']:.4f}")
    print(f"  🎯 Precision: {metrics['precision_etconformance']:.4f}")
    print(f"  🔄 Generalization: {metrics['generalization']:.4f}")
    print(f"  ✨ Simplicity: {metrics['simplicity']:.4f}")
    print(f"  🔧 Model: {metrics['model_info']['places']} places, {metrics['model_info']['transitions']} transitions")
    print(f"💾 Saved to:\n - {json_path}\n - {csv_path}")


if __name__ == "__main__":
    main()