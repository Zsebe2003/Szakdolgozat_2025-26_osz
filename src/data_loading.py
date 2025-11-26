import pandas as pd
from pathlib import Path

HU_MONTHS = {
    "január": 1, "február": 2, "március": 3, "április": 4, "május": 5, "június": 6,
    "július": 7, "augusztus": 8, "szeptember": 9, "október": 10, "november": 11, "december": 12
}


def parse_hu_datetime_series(s: pd.Series) -> pd.Series:
    clean = s.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    parts = clean.str.extract(
        r"^(?P<ev>\d{4})\.\s*"
        r"(?P<honapnev>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)\s+"
        r"(?P<nap>\d{1,2})\.,\s*"
        r"(?P<ora>\d{1,2}):(?P<perc>\d{2}):(?P<mp>\d{2})$",
        expand=True,
    )
    parts["h"] = parts["honapnev"].str.lower().map(HU_MONTHS)
    iso = (
        parts["ev"].fillna("") + "-" +
        parts["h"].fillna("").astype(str).str.zfill(2) + "-" +
        parts["nap"].fillna("").astype(str).str.zfill(2) + " " +
        parts["ora"].fillna("").astype(str).str.zfill(2) + ":" +
        parts["perc"].fillna("").astype(str).str.zfill(2) + ":" +
        parts["mp"].fillna("").astype(str).str.zfill(2)
    )
    return pd.to_datetime(iso, errors="coerce", format="%Y-%m-%d %H:%M:%S")


def read_input_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)
