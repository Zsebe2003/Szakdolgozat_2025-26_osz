from pathlib import Path
import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_vis
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_vis
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.statistics.start_activities.log import get as start_act_get
from pm4py.statistics.end_activities.log import get as end_act_get
from .config import DEPENDENCY_THRESH, MIN_ACTIVITY_OCC, MIN_DFG_OCC, AND_MEASURE_THRESH


def run_heuristics_pipeline(event_log, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heuristics Miner
    hm_params = {
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: DEPENDENCY_THRESH,
        heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: MIN_ACTIVITY_OCC,
        heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: MIN_DFG_OCC,
        heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: AND_MEASURE_THRESH,
    }
    heu_net = heuristics_miner.apply_heu(event_log, parameters=hm_params)

    # Heuristics Net ábra
    hn_gviz = hn_vis.apply(heu_net, parameters={"format": "png"})
    hn_vis.save(hn_gviz, str(out_dir / "heuristics_net.png"))

    # DFG + ábra
    dfg = dfg_discovery.apply(event_log)
    start_acts = start_act_get.get_start_activities(event_log)
    end_acts = end_act_get.get_end_activities(event_log)
    dfg_gviz = dfg_vis.apply(dfg, log=event_log, variant=dfg_vis.Variants.FREQUENCY, parameters={
        "format": "png", "start_activities": start_acts, "end_activities": end_acts
    })
    dfg_vis.save(dfg_gviz, str(out_dir / "dfg_frequency.png"))

    # Petri háló ábra
    pn, im, fm = hn_converter.apply(heu_net)
    pm4py.save_vis_petri_net(pn, im, fm, str(out_dir / "petri_from_heuristics.png"))

    print("PM4Py Heuristics Miner kész.")
