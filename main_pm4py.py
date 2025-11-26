from dotenv import load_dotenv
from src.utils.paths import Paths
from src.pm4py_pipeline.eventlog import (
    export_xes,
    export_weekly_xes,
    weekly_counts_dataframe,
)
import pandas as pd


def main():
    load_dotenv()
    p = Paths()
    p.ensure()

    # A main_preprocess.py által előállított, már tisztított adat
    df = pd.read_csv(p.processed / "df_remaining_export.csv")

    # =====================================================================
    # 1) NAPI ESEMÉNYSZINTŰ XES – TELJES ADATBÁZIS
    # =====================================================================
    export_xes(df, p.xes / "event_log_remaining_ALL.xes")

    # =====================================================================
    # 2) NAPI XES – SZÁMONKÉRÉS NÉLKÜLI ESEMÉNYEK
    # =====================================================================
    if "Uj_oszlop" in df.columns:
        df_no_exam = df[df["Uj_oszlop"] != "Szamonkeres"].copy()
        export_xes(df_no_exam, p.xes / "event_log_remaining_NO_EXAM.xes")

    # =====================================================================
    # 3) HETI BONTÁSÚ XES + HOZZÁ TARTOZÓ CSV-K (időbeli dinamika)
    # =====================================================================

    # 3/a) Heti + kategória XES
    weekly_log_cat, weekly_ev_cat_df = export_weekly_xes(
        df_src=df,
        out_path=p.xes / "event_log_weekly_with_category.xes",
        include_category=True,
    )
    weekly_ev_cat_df.to_csv(
        p.processed / "weekly_events_with_category.csv",
        index=False,
        encoding="utf-8",
    )

    # 3/b) Tiszta heti XES (kategória nélkül)
    weekly_log, weekly_ev_df = export_weekly_xes(
        df_src=df,
        out_path=p.xes / "event_log_weekly.xes",
        include_category=False,
    )
    weekly_ev_df.to_csv(
        p.processed / "weekly_events.csv",
        index=False,
        encoding="utf-8",
    )

    # 3/c) Elemző tábla: user × év × hét × kategória → esemény darabszám
    wk_counts = weekly_counts_dataframe(df)
    wk_counts.to_csv(
        p.processed / "weekly_counts_user_week_category.csv",
        index=False,
        encoding="utf-8",
    )

        # --- 4) NAPI XES – LOOP-MENTES LOG ---
    # A main_preprocess.py által elmentett loop-mentes DF-ből (Idő_dt, Uj_oszlop, user_id)
    no_loops_csv = p.processed / "df_remaining_no_loops.csv"
    if no_loops_csv.exists():
        df_no_loops = pd.read_csv(no_loops_csv)
        # (opcionális, ha kell biztosan datetime)
        if "Idő_dt" in df_no_loops.columns:
            df_no_loops["Idő_dt"] = pd.to_datetime(df_no_loops["Idő_dt"], errors="coerce")
        export_xes(df_no_loops, p.xes / "event_log_remaining_NO_LOOPS.xes")
        print("✅ XES (loop-mentes) mentve: data/xes/event_log_remaining_NO_LOOPS.xes")
    else:
        print("⚠️ Nem található a loop-mentes CSV (data/processed/df_remaining_no_loops.csv). "
            "Futtasd előbb a main_preprocess.py-t.")


    # =====================================================================
    # 4) NAPI XES – CSAK ISMERT TANTERVI HÉT (Tantervi_hét ≠ "Ismeretlen")
    #    + concept:name = Tantervi_hét
    # =====================================================================

    # 4/a) Szűrés
    df_known_weeks = df[
        df["Tantervi_hét"].notna() & (df["Tantervi_hét"] != "Ismeretlen")
    ].copy()

    # 4/b) Tantervi hét legyen az esemény neve (concept:name)
    df_known_for_xes = df_known_weeks.copy()
    df_known_for_xes["Uj_oszlop"] = df_known_for_xes["Tantervi_hét"].astype(str)

    # 4/c) Export
    export_xes(
        df_known_for_xes,
        p.xes / "event_log_TANTERVIHET_KNOWN_ONLY.xes",
    )

    # =====================================================================

    print("XES fájlok és heti elemző CSV-k elkészültek.")


if __name__ == "__main__":
    main()
