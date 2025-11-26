from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple
import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

REQUIRED = {"user_id", "Uj_oszlop", "Idő_dt"}

def fix_xes_for_prom(path: Path) -> None:
    """
    Gyors header-fix, hogy az IEEE XES parser (pl. ProM) ne panaszkodjon:

      - xes.version értéke legyen decimális (pl. 1.0 az 1849-2016 helyett)
      - xes.features attribútum biztosan létezzen, de ne tartalmazzon 'nested-attributes'-t
        (→ xes.features="")
    """
    try:
        text = path.read_text(encoding="utf-8")

        # 1) xes.version javítása: bármi is volt, legyen "1.0"
        text = re.sub(r'xes\.version="[^"]*"', 'xes.version="1.0"', text)

        # 2) xes.features kezelése
        if 'xes.features="' in text:
            # ha van, csak az értékét írjuk át üresre
            text = re.sub(r'xes\.features="[^"]*"', 'xes.features=""', text)
        else:
            # ha nincs, beszúrjuk a <log ...> tagba
            # első <log ...> előfordulás után teszünk egy xes.features="" attribútumot
            text = text.replace("<log ", '<log xes.features="" ', 1)

        path.write_text(text, encoding="utf-8")
        print(f"✅ XES header javítva ProM-hoz: {path}")
    except Exception as e:
        print(f"⚠️ Nem sikerült a XES header javítása ({path}): {e}")

# --- NAPI ESEMÉNYSZINTŰ XES ---

def to_event_log(df_src: pd.DataFrame):
    missing = REQUIRED - set(df_src.columns)
    assert not missing, f"Hiányzó oszlop(ok): {missing}"

    df_pm = df_src[["user_id", "Uj_oszlop", "Idő_dt"]].copy()
    df_pm["time:timestamp"] = pd.to_datetime(df_pm["Idő_dt"], errors="coerce")
    df_pm = (
        df_pm.dropna(subset=["time:timestamp"])
             .rename(columns={
                 "user_id": "case:concept:name",
                 "Uj_oszlop": "concept:name",
             })[["case:concept:name", "concept:name", "time:timestamp"]]
    )

    df_pm["case:concept:name"] = df_pm["case:concept:name"].astype(str)
    df_pm["concept:name"] = df_pm["concept:name"].astype(str)
    df_pm = df_pm.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)

    params = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
    event_log = log_converter.apply(df_pm, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)
    return event_log


def export_xes(df_src: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    event_log = to_event_log(df_src)
    pm4py.write_xes(event_log, str(out_path))

    # <<< ÚJ: fejléc-fix ProM-hoz
    fix_xes_for_prom(out_path)

    print(f"XES mentve: {out_path} | Események: {len(df_src)} | Esetek: {df_src['user_id'].nunique()}")
    return event_log



# --- HETI BONTÁSÚ XES + DF ---

def _weekly_event_df(df_src: pd.DataFrame, include_category: bool = True) -> pd.DataFrame:
    """
    Heti bontású 'eseménylista' XES-hez és elemzéshez.
    - include_category=True:  concept:name = "W{week}-{year} | {Uj_oszlop}"
    - include_category=False: concept:name = "W{week}-{year}"
    1 esemény / (user_id, hét[, kategória]) a hétfő 00:00 időpontra.
    """
    need = {"user_id", "Uj_oszlop", "Idő_dt"}
    missing = need - set(df_src.columns)
    assert not missing, f"Hiányzó oszlop(ok) a heti exporthoz: {missing}"

    df = df_src.copy()
    df["Idő_dt"] = pd.to_datetime(df["Idő_dt"], errors="coerce")
    df = df.dropna(subset=["Idő_dt", "user_id"])

    iso = df["Idő_dt"].dt.isocalendar()
    df["_iso_year"] = iso.year.astype(int)
    df["_iso_week"] = iso.week.astype(int)
    df["_week_start"] = df["Idő_dt"].dt.to_period("W-MON").dt.start_time

    if include_category:
        df["_event_name"] = (
            "W" + df["_iso_week"].astype(str).str.zfill(2) + "-" + df["_iso_year"].astype(str)
            + " | " + df["Uj_oszlop"].astype(str)
        )
        group_cols = ["user_id", "_iso_year", "_iso_week", "Uj_oszlop"]
    else:
        df["_event_name"] = "W" + df["_iso_week"].astype(str).str.zfill(2) + "-" + df["_iso_year"].astype(str)
        group_cols = ["user_id", "_iso_year", "_iso_week"]

    ev = (
        df.groupby(group_cols, as_index=False)
          .agg({"_week_start": "min", "_event_name": "first"})
          .rename(columns={
              "_week_start": "time:timestamp",
              "user_id": "case:concept:name",
              "_event_name": "concept:name"
          })
    )

    ev["case:concept:name"] = ev["case:concept:name"].astype(str)
    ev["concept:name"] = ev["concept:name"].astype(str)
    ev["time:timestamp"] = pd.to_datetime(ev["time:timestamp"], errors="coerce")
    ev = ev.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    ev = dataframe_utils.convert_timestamp_columns_in_df(ev)
    return ev


def export_weekly_xes(
    df_src: pd.DataFrame,
    out_path: Path,
    include_category: bool = True
) -> tuple[object, pd.DataFrame]:
    """
    Heti bontású XES export + a használt esemény-DataFrame visszaadása.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_ev = _weekly_event_df(df_src, include_category=include_category)

    params = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
    event_log = log_converter.apply(df_ev, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)

    pm4py.write_xes(event_log, str(out_path))

    # <<< ÚJ: fejléc-fix ProM-hoz
    fix_xes_for_prom(out_path)

    print(f"XES (heti) mentve: {out_path} | Események: {len(df_ev)} | Esetek: {df_ev['case:concept:name'].nunique()}")
    return event_log, df_ev



def weekly_counts_dataframe(df_src: pd.DataFrame) -> pd.DataFrame:
    """
    Elemző dataframe heti bontásban:
      user_id × iso_year × iso_week × Uj_oszlop → esemeny_db
    """
    need = {"user_id", "Uj_oszlop", "Idő_dt"}
    missing = need - set(df_src.columns)
    assert not missing, f"Hiányzó oszlop(ok) a weekly_counts_dataframe-hoz: {missing}"

    df = df_src.copy()
    df["Idő_dt"] = pd.to_datetime(df["Idő_dt"], errors="coerce")
    df = df.dropna(subset=["Idő_dt", "user_id"])

    iso = df["Idő_dt"].dt.isocalendar()
    out = (
        df.assign(iso_year=iso.year.astype(int), iso_week=iso.week.astype(int))
          .groupby(["user_id", "iso_year", "iso_week", "Uj_oszlop"], as_index=False)
          .size()
          .rename(columns={"size": "esemeny_db"})
          .sort_values(["user_id", "iso_year", "iso_week", "Uj_oszlop"])
          .reset_index(drop=True)
    )
    return out
