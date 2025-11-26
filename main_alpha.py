from dotenv import load_dotenv
from pathlib import Path
import pm4py
from src.utils.paths import Paths


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    # --- 1) XES beolvasása ---
    xes_path = p.xes / "event_log_remaining_ALL.xes"
    if not xes_path.exists():
        raise SystemExit(f"Nincs XES fájl: {xes_path}\nElőbb futtasd: python main_pm4py.py")

    print(f"Beolvasás: {xes_path}")
    log = pm4py.read_xes(str(xes_path))

    # --- 2) Alpha Miner modell építése ---
    print("Alpha Miner futtatása...")
    net, im, fm = pm4py.discover_petri_net_alpha(log)

    # --- 3) Ábrák mentése ---
    out_dir = p.figures / "alpha"
    out_dir.mkdir(parents=True, exist_ok=True)

    gviz = pm4py.visualization.petri_net.visualizer.apply(net, im, fm)
    pm4py.visualization.petri_net.visualizer.save(
        gviz, str(out_dir / "alpha_miner_petri.png")
    )

    print(f"Alpha Miner Petri-háló elmentve ide: {out_dir / 'alpha_miner_petri.png'}")


if __name__ == "__main__":
    main()

