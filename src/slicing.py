import pandas as pd
import numpy as np
from datetime import time


def split_users(df: pd.DataFrame, exclude_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["user_id"].isin(exclude_ids)
    return df[mask].copy(), df[~mask].copy()


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Idő_dt"] = pd.to_datetime(out["Idő_dt"], errors="coerce")
    iso_cal = out["Idő_dt"].dt.isocalendar()
    out["hónap"] = out["Idő_dt"].dt.month
    out["hét"] = iso_cal.week.astype("Int64")
    out["nap"] = out["Idő_dt"].dt.day
    out["óra"] = out["Idő_dt"].dt.hour
    return out


def label_orai_otthoni(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid_ts = out["Idő_dt"].notna()

    # tavaszi szünet kizárás – igény szerint módosítható
    break_start = pd.Timestamp("2025-04-21 00:00:00")
    break_end   = pd.Timestamp("2025-04-27 23:59:59")
    not_break = ~((out["Idő_dt"] >= break_start) & (out["Idő_dt"] <= break_end))

    is_wed = out["Idő_dt"].dt.dayofweek == 2  # H=0
    t = out["Idő_dt"].dt.time
    in_slot = (
        ((t >= time(8, 0)) & (t <= time(9, 30))) |
        ((t >= time(9, 50)) & (t <= time(11, 20))) |
        ((t >= time(11, 40)) & (t <= time(13, 10)))
    )
    orai_mask = valid_ts & not_break & is_wed & in_slot

    out["Munka_típus"] = orai_mask.map({True: "Órai", False: "Otthoni"})

    # IP keresztmetszet - csak IP alapján
    ip_is_campus = out.get("IP-cím", pd.Series([None]*len(out))).astype(str).str.startswith("146.110", na=False)
    out["lokacio"] = np.where(
        ip_is_campus,
        "Egyetemen",
        "Egyéb helyen"
    )

    # 3-utas jelölés
    out["Munka_3utas"] = np.select(
        [
            (out["lokacio"] == "Egyetemen") & (out["Uj_oszlop"] == "Szamonkeres") & (out["Munka_típus"] == "Órai"),   # Egyetemi lokáció + számonkérés
            (out["lokacio"] == "Egyetemen") & (out["Munka_típus"] == "Órai"),        # Egyetemi lokáció + órai munka
        ],
        [
            "Órai – számonkérés",
            "Órai – nem számonkérés", 
        ],
            default="Otthoni",
    )
    return out


def build_slices(df_remaining: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices: dict[str, pd.DataFrame] = {}

    # Számonkérés nélküli rész
    slices["df_remaining_no_exam"] = df_remaining[df_remaining["Uj_oszlop"] != "Szamonkeres"].copy()

    # Extra-hallgatók user_id listája
    extra_students = (
        df_remaining.loc[df_remaining["Uj_oszlop"] == "Extra", "user_id"]
        .dropna().astype("Int64").drop_duplicates().sort_values()
    )

    slices["df_extra_with_exam"] = df_remaining[df_remaining["user_id"].isin(extra_students)].copy()
    slices["df_nonextra_with_exam"] = df_remaining[~df_remaining["user_id"].isin(extra_students)].copy()

    slices["df_extra_no_exam"] = slices["df_extra_with_exam"][slices["df_extra_with_exam"]["Uj_oszlop"] != "Szamonkeres"].copy()
    slices["df_nonextra_no_exam"] = slices["df_nonextra_with_exam"][slices["df_nonextra_with_exam"]["Uj_oszlop"] != "Szamonkeres"].copy()

    slices["df_extra_with_exam_only"] = slices["df_extra_with_exam"][slices["df_extra_with_exam"]["Uj_oszlop"] == "Szamonkeres"].copy()

    return slices
