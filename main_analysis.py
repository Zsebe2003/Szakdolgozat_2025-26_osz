from dotenv import load_dotenv
from src.utils.paths import Paths
from src.analysis.plotting import (
    plot_timeparts_stacked_by_category,
    plot_timeparts_stacked_by_orai_otthoni,
    plot_monthly_bars_orai_otthoni,
    save_four_subplots_category,
    save_four_subplots_orai,
    save_four_subplots_3way,
)
from src.analysis.stats import print_basic_stats
import pandas as pd


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    df_remaining = pd.read_csv(p.processed / "df_remaining_export.csv")

    # Alap leírók
    print_basic_stats(df_remaining)

    # Ábrák mentése – kategóriák szerint
    save_four_subplots_category(df_remaining, out_dir=p.figures / "osszesitett")

    # Ábrák mentése – Órai vs Otthoni
    save_four_subplots_orai(df_remaining, out_dir=p.figures / "osszesitett")

    # Háromutas verzió: Otthoni / Órai – nem számonkérés / Órai – számonkérés
    save_four_subplots_3way(df_remaining, out_dir=p.figures / "osszesitett")

    # Időrészek szerinti halmozott ábrák
    plot_timeparts_stacked_by_category(df_remaining, out_dir=p.figures / "reszletes")
    plot_timeparts_stacked_by_orai_otthoni(df_remaining, out_dir=p.figures / "reszletes")
    
    # Havi bontás
    plot_monthly_bars_orai_otthoni(df_remaining, out_dir=p.figures)

    print("✅ Elemzési ábrák mentve a figures/ alá.")


if __name__ == "__main__":
    main()