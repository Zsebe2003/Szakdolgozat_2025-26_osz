# main_heuristics.py
from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from dotenv import load_dotenv
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from src.utils.paths import Paths

# PM4Py 2.7+ compatible evaluators
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_eval
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_eval

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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



def discover_heuristics_petri(
    log: EventLog,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5,
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Discover Petri net using Heuristics Miner.
    
    Higher threshold values produce stricter, "cleaner" models.
    
    Args:
        log: Event log to mine from
        dependency_threshold: Threshold for dependency relations
        and_threshold: Threshold for AND/parallel relations  
        loop_two_threshold: Threshold for length-two loops
        
    Returns:
        Tuple of (Petri net, initial marking, final marking)
    """
    logger.info(f"Discovering Petri net with thresholds: dependency={dependency_threshold}, "
                f"and={and_threshold}, loop2={loop_two_threshold}")
    
    net, im, fm = pm4py.discover_petri_net_heuristics(
        log,
        dependency_threshold=dependency_threshold,
        and_threshold=and_threshold,
        loop_two_threshold=loop_two_threshold,
    )

    
    logger.info(f"Discovered Petri net: {len(net.places)} places, {len(net.transitions)} transitions")
    return net, im, fm


def compute_metrics_fixed(log: EventLog, net: PetriNet, im: Marking, fm: Marking) -> Dict[str, Any]:
    """
    Compute quality metrics for the discovered model with robust error handling.
    
    Args:
        log: Event log for evaluation
        net: Petri net
        im: Initial marking
        fm: Final marking
        
    Returns:
        Dictionary containing quality metrics
    """
    metrics = {}
    
    # 1. Fitness calculation - using PM4Py's high-level API
    try:
        fitness_dict = fitness_eval.apply(log, net, im, fm, variant=fitness_eval.Variants.TOKEN_BASED)
        metrics["fitness"] = float(fitness_dict["average_trace_fitness"])
        logger.info(f"Fitness calculated successfully: {metrics['fitness']:.4f}")
    except Exception as e:
        logger.warning(f"Standard fitness calculation failed: {e}. Using fallback...")
        try:
            # Fallback: manual token replay
            from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
            aligned_traces = token_replay.apply(log, net, im, fm)
            
            total_fitness = 0
            count = 0
            for trace in aligned_traces:
                if isinstance(trace, dict) and "trace_fitness" in trace:
                    total_fitness += trace["trace_fitness"]
                    count += 1
                elif hasattr(trace, 'get'):
                    # Try different possible keys
                    for key in ['fitness', 'trace_fitness', 'conformance']:
                        if key in trace:
                            total_fitness += float(trace[key])
                            count += 1
                            break
            
            metrics["fitness"] = total_fitness / count if count > 0 else 0.0
            logger.info(f"Fallback fitness calculation completed: {metrics['fitness']:.4f}")
        except Exception as e2:
            logger.error(f"All fitness calculations failed: {e2}")
            metrics["fitness"] = 0.0

    # 2. Precision calculation
    try:
        precision_dict = precision_eval.apply(log, net, im, fm, variant=precision_eval.Variants.ETCONFORMANCE_TOKEN)
        metrics["precision"] = float(precision_dict)
        logger.info(f"Precision calculated successfully: {metrics['precision']:.4f}")
    except Exception as e:
        logger.warning(f"Precision calculation failed: {e}")
        metrics["precision"] = 0.0

    # 3. Generalization
    try:
        generalization = float(generalization_eval.apply(log, net, im, fm))
        metrics["generalization"] = generalization
        logger.info(f"Generalization calculated successfully: {metrics['generalization']:.4f}")
    except Exception as e:
        logger.warning(f"Generalization calculation failed, using fallback: {e}")
        # Fallback: ratio of activated transitions
        try:
            from pm4py.algo.conformance.tokenreplay import algorithm as token_rep
            rep = token_rep.apply(log, net, im, fm)
            activated = set()
            for tr in rep:
                if hasattr(tr, 'get') and "activated_transitions" in tr:
                    activated.update(tr["activated_transitions"])
                elif hasattr(tr, 'activated_transitions'):
                    activated.update(tr.activated_transitions)
            
            all_tr = {t.label for t in net.transitions if t.label is not None}
            metrics["generalization"] = float(len(activated) / len(all_tr)) if all_tr else 0.5
        except Exception as e2:
            logger.error(f"Generalization fallback also failed: {e2}")
            metrics["generalization"] = 0.5

    # 4. Simplicity
    try:
        simplicity = float(simplicity_eval.apply(net))
        metrics["simplicity"] = simplicity
        logger.info(f"Simplicity calculated successfully: {metrics['simplicity']:.4f}")
    except Exception as e:
        logger.warning(f"Simplicity calculation failed, using fallback: {e}")
        # Structural fallback
        num_arcs = len(net.arcs)
        denom = (len(net.places) + len(net.transitions))
        metrics["simplicity"] = 1.0 / (1.0 + (num_arcs / denom)) if denom else 0.5

    # 5. Additional metrics
    metrics["model_info"] = {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
        "visible_transitions": len([t for t in net.transitions if t.label is not None]),
        "silent_transitions": len([t for t in net.transitions if t.label is None]),
    }
    
    # 6. Calculate F-score (harmonic mean of fitness and precision)
    if metrics["fitness"] > 0 and metrics["precision"] > 0:
        metrics["f_score"] = 2 * (metrics["fitness"] * metrics["precision"]) / (metrics["fitness"] + metrics["precision"])
    else:
        metrics["f_score"] = 0.0
    
    return metrics


def debug_token_replay_structure(log: EventLog, net: PetriNet, im: Marking, fm: Marking) -> None:
    """Debug function to understand the token replay result structure."""
    logger.info("Debugging token replay structure...")
    try:
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        results = token_replay.apply(log, net, im, fm)
        
        if results and len(results) > 0:
            first_result = results[0]
            logger.info(f"Type of first result: {type(first_result)}")
            logger.info(f"First result keys (if dict): {list(first_result.keys()) if isinstance(first_result, dict) else 'Not a dict'}")
            
            # Print first few results for inspection
            for i, result in enumerate(results[:3]):
                logger.info(f"Result {i}: {result}")
                
    except Exception as e:
        logger.error(f"Debug failed: {e}")


def save_visualizations(log: EventLog, net: PetriNet, im: Marking, fm: Marking, 
                       out_dir: Path, params: Dict[str, float]) -> None:
    """Save Petri net and Heuristics net visualizations."""
    # Petri net visualization
    try:
        gviz_pn = pm4py.visualization.petri_net.visualizer.apply(net, im, fm)
        pm4py.visualization.petri_net.visualizer.save(gviz_pn, str(out_dir / "heuristics_petri.png"))
        logger.info(f"Petri net visualization saved → {out_dir/'heuristics_petri.png'}")
    except Exception as e:
        logger.error(f"Petri net visualization failed (Graphviz required): {e}")

    # Heuristics net visualization
    try:
        hn = pm4py.discover_heuristics_net(
            log,
            dependency_threshold=params["dependency"],
            and_threshold=params["andthr"],
            loop_two_threshold=params["loop2"],
        )
        gviz_hn = pm4py.visualization.heuristics_net.visualizer.apply(hn)
        pm4py.visualization.heuristics_net.visualizer.save(gviz_hn, str(out_dir / "heuristics_net.png"))
        logger.info(f"Heuristics net visualization saved → {out_dir/'heuristics_net.png'}")
    except Exception as e:
        logger.error(f"Heuristics net visualization failed: {e}")


def main():
    """Main execution function."""
    load_dotenv()
    p = Paths()
    p.ensure()

    ap = argparse.ArgumentParser(
        description="Heuristics Miner on an XES log (+optional viz +metrics)."
    )
    ap.add_argument("--xes", type=str, default=None,
                    help="XES file path (default: data/xes/event_log_remaining_ALL.xes)")
    ap.add_argument("--viz", action="store_true",
                    help="If set, save Petri and heuristics net visualizations (requires Graphviz).")
    ap.add_argument("--dependency", type=float, default=0.5,
                    help="Heuristics Miner dependency_threshold (default 0.5)")
    ap.add_argument("--andthr", type=float, default=0.65,
                    help="Heuristics Miner and_threshold (default 0.65)")
    ap.add_argument("--loop2", type=float, default=0.5,
                    help="Heuristics Miner loop_two_threshold (default 0.5)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for results (default: figures/heuristics)")
    ap.add_argument("--debug", action="store_true",
                    help="Enable debug output for token replay structure")
    
    args = ap.parse_args()

    # Resolve paths
    xes_path = Path(args.xes) if args.xes else (p.xes / "event_log_remaining_ALL.xes")
    if not xes_path.exists():
        raise SystemExit(f"❌ XES file not found: {xes_path}. Generate it first (python main_pm4py.py).")

    out_dir = Path(args.output_dir) if args.output_dir else (p.figures / "heuristics")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load event log
        log = pm4py.read_xes(str(xes_path))
        logger.info(f"Loaded log: {xes_path} | traces={len(log)} | events={sum(len(trace) for trace in log)}")
        
        # Discover Petri net using Heuristics Miner
        net, im, fm = discover_heuristics_petri(
            log,
            dependency_threshold=args.dependency,
            and_threshold=args.andthr,
            loop_two_threshold=args.loop2,
        )

        # Debug token replay structure if requested
        if args.debug:
            debug_token_replay_structure(log, net, im, fm)

        # Generate visualizations if requested
        if args.viz:
            save_visualizations(log, net, im, fm, out_dir, vars(args))

        # Compute metrics with fixed calculation
        metrics = compute_metrics_fixed(log, net, im, fm)

        save_petri_artifacts(net, im, fm, out_dir, basename="heuristics_petri")


        # Save results
        results = {
            "xes_path": str(xes_path),
            "log_info": {
                "traces": len(log),
                "events": sum(len(trace) for trace in log)
            },
            "parameters": {
                "dependency_threshold": args.dependency,
                "and_threshold": args.andthr,
                "loop_two_threshold": args.loop2,
            },
            **metrics,
        }

        # Save JSON
        json_path = out_dir / "heuristics_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save CSV
        csv_path = out_dir / "heuristics_metrics.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for k, v in metrics.items():
                if k != "model_info":
                    f.write(f"{k},{v}\n")

        # Print summary
        logger.info("✅ Heuristics Miner completed successfully")
        logger.info(f"   Fitness:        {metrics['fitness']:.4f}")
        logger.info(f"   Precision:      {metrics['precision']:.4f}")
        logger.info(f"   F-score:        {metrics.get('f_score', 0):.4f}")
        logger.info(f"   Generalization: {metrics['generalization']:.4f}")
        logger.info(f"   Simplicity:     {metrics['simplicity']:.4f}")
        logger.info(f"   Model: places={metrics['model_info']['places']}, "
                   f"transitions={metrics['model_info']['transitions']}, arcs={metrics['model_info']['arcs']}")
        logger.info(f"💾 Results saved: {json_path}")
        logger.info(f"💾 Results saved: {csv_path}")

    except Exception as e:
        logger.error(f"❌ Heuristics Miner failed: {e}")
        raise


if __name__ == "__main__":
    main()