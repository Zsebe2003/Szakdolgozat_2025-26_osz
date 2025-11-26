# src/transformations.py
from __future__ import annotations
from datetime import time
import pandas as pd

def _compute_orai_mask(df: pd.DataFrame,
                       break_start=pd.Timestamp("2025-04-21 00:00:00"),
                       break_end=pd.Timestamp("2025-04-27 23:59:59")) -> pd.Series:
    """
    Ugyanaz a logika, mint amit leírtál: szerda, 3 idősáv, tavaszi szünet kizárva.
    Visszaad egy bool maszkot: True = Órai, False = nem órai.
    """
    out = df.copy()
    valid_ts = pd.to_datetime(out["Idő_dt"], errors="coerce").notna()
    out["Idő_dt"] = pd.to_datetime(out["Idő_dt"], errors="coerce")

    # tavaszi szünet kizárás
    not_break = ~((out["Idő_dt"] >= break_start) & (out["Idő_dt"] <= break_end))

    # szerda
    is_wed = out["Idő_dt"].dt.dayofweek == 2  # hétfő=0

    # idősávok
    t = out["Idő_dt"].dt.time
    in_slot = (
        ((t >= time(8, 0)) & (t <= time(9, 30))) |
        ((t >= time(9, 50)) & (t <= time(11, 20))) |
        ((t >= time(11, 40)) & (t <= time(13, 10)))
    )

    orai_mask = valid_ts & not_break & is_wed & in_slot
    return orai_mask


def reclassify_exam_to_admin_if_otthoni(df: pd.DataFrame,
                                        exam_label: str = "Szamonkeres",
                                        admin_label: str = "Admin") -> pd.DataFrame:
    """
    Azokat a sorokat, ahol Uj_oszlop == exam_label ÉS a fenti logika szerint 'Otthoni' lenne,
    átírja Uj_oszlop = admin_label értékre.

    Ha a DataFrame-ben már létezik 'Munka_típus' (Órai/Otthoni), azt használja.
    Ha nincs, a maszkot újraszámolja a fenti órarend-logikával.
    """
    out = df.copy()

    if "Munka_típus" in out.columns:
        otthoni_mask = out["Munka_típus"].astype(str) == "Otthoni"
    else:
        orai_mask = _compute_orai_mask(out)
        otthoni_mask = ~orai_mask

    target_mask = (out["Uj_oszlop"] == exam_label) & otthoni_mask
    n_aff = int(target_mask.sum())
    if n_aff > 0:
        out.loc[target_mask, "Uj_oszlop"] = admin_label
        print(f"Reclassify: {n_aff} sor átírva '{exam_label}' → '{admin_label}' (Otthoni).")
    else:
        print("Reclassify: nem volt átírható sor (feltétel nem teljesült).")

    return out
