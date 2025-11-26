from pathlib import Path
import argparse
from dotenv import load_dotenv
import os
import pandas as pd

from src.utils.paths import Paths
from src.data_loading import read_input_excel, parse_hu_datetime_series
from src.cleaning import (
    add_extracted_ids,
    apply_time_window,
    apply_exclusions,
)
from src.categories import build_mapping
from src.slicing import (
    split_users,
    add_time_parts,
    label_orai_otthoni,
    build_slices,
)
from src.weekly_plan import attach_week_plan_columns
from src.exports import save_csv, save_xlsx
from src.transformations import reclassify_exam_to_admin_if_otthoni


def main(input_path: str | None = None):
    load_dotenv()

    # --- Paths ---
    p = Paths()
    p.ensure()

    # --- Input ---
    src = input_path or os.getenv("INPUT_XLSX")
    if not src:
        raise SystemExit("Nincs megadva INPUT_XLSX sem argumentumban, sem .env fájlban.")
    src = Path(src)

    print(f"Beolvasás: {src}")
    df = read_input_excel(src)

    # --- Dátum + szűrés ---
    df["Idő_dt"] = parse_hu_datetime_series(df["Idő"])  # robusztus HU datetime

    start = pd.to_datetime(os.getenv("START_DATE", "2025-02-17 00:00:00"))
    end   = pd.to_datetime(os.getenv("END_DATE",   "2025-06-23 23:59:59"))
    df = apply_time_window(df, start, end)

    # --- ID-k + kizárások ---
    df = add_extracted_ids(df)
    df = apply_exclusions(df)

    # --- Kategóriák ---
    mapping = build_mapping()
    df["Uj_oszlop"] = df["Eseménykörnyezet"].map(mapping).fillna("Egyéb")

    # --- Felhasználó szeletek ---
    ids_exclude = [96499, 605, 125110, 70866, 60896, 124612, 576]
    df_masik_users, df_remaining = split_users(df, ids_exclude)

    # --- Időrészek (hónap/hét/nap/óra) ---
    df_remaining = add_time_parts(df_remaining)

    # --- Órai vs Otthoni címke + keresztmetszet ---
    df_remaining = label_orai_otthoni(df_remaining)

    # === ÚJ: Tantervi hét hozzárendelése ===
    df_remaining = attach_week_plan_columns(
        df_remaining,
        src_col="Eseménykörnyezet",      # innen olvassuk a mapping kulcsot
        label_col="Tantervi_hét",        # új oszlop (string)
        num_col="Tantervi_hét_szám"      # új oszlop (int/NA)
    )

    # opcionális: exportálunk egy katalógust is, hogy lásd a fedettséget
    catalog = (
        df_remaining[["Eseménykörnyezet", "Tantervi_hét", "Tantervi_hét_szám"]]
        .drop_duplicates()
        .sort_values(["Tantervi_hét_szám", "Tantervi_hét", "Eseménykörnyezet"], na_position="last")
    )
    save_csv(catalog, p.processed / "event_week_catalog.csv")

    # --- Szeletek előállítása ---
    slices = build_slices(df_remaining)

    # --- Órai vs Otthoni címke + keresztmetszet ---
    df_remaining = label_orai_otthoni(df_remaining)

    # --- Számonkérés → Admin, ha Otthoni lenne ---
    df_remaining = reclassify_exam_to_admin_if_otthoni(
        df_remaining,
        exam_label="Szamonkeres",   # ha más a pontos címke, itt módosítsd
        admin_label="Admin"
        )
    
    # Loop-mentes változat: egymást közvetlenül követő azonos Uj_oszlop értékek eldobása
    mask_no_loops = df_remaining["Uj_oszlop"].ne(
        df_remaining.groupby("user_id")["Uj_oszlop"].shift()
    )

    df_no_loops = (
        df_remaining.loc[mask_no_loops, ["Idő_dt", "Uj_oszlop", "user_id"]]
        .reset_index(drop=True)
    )

    # Külön elmentjük
    p = Paths()
    save_csv(df_no_loops, p.processed / "df_remaining_no_loops.csv")
    save_xlsx(df_no_loops, p.processed / "df_remaining_no_loops.xlsx")

    print("▶ Loop-mentes dataframe előállítva: df_remaining_no_loops.*")

    # --- Exportok ---
    save_csv(df_remaining, p.processed / "df_remaining_export.csv")
    try:
        save_xlsx(df_remaining, p.processed / "df_remaining_export.xlsx")
    except Exception as e:
        print("XLSX mentés nem sikerült, CSV mentés rendelkezésre áll.")
        print("Hiba:", e)

    # Extra riportok
    for name, dfx in slices.items():
        save_csv(dfx, p.processed / f"{name}.csv")

    df_known_weeks = df_remaining[df_remaining["Tantervi_hét"] != "Ismeretlen"].copy()


    print("✅ Előfeldolgozás kész. Kimenetek a data/processed mappában.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Bemeneti Excel útvonala")
    args = ap.parse_args()
    main(args.input)
