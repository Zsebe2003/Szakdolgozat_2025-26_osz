import pandas as pd

ID_PATTERNS = {
    "user_id":    r"user with id '(\d+)'",
    "attempt_id": r"attempt with id '(\d+)'",
    "quiz_id":    r"course module id '(\d+)'",
    "tetel_id":   r"Tétel (\d+) azonosítóval",
}


def add_extracted_ids(df: pd.DataFrame) -> pd.DataFrame:
    extracts = {name: df["Leírás"].str.extract(pat, expand=False).astype("Int64") for name, pat in ID_PATTERNS.items()}
    out = df.assign(**extracts)
    return out.drop(columns=[c for c in ["attempt_id", "quiz_id"] if c in out.columns])


def apply_time_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["Idő_dt"] >= start) & (df["Idő_dt"] <= end)].copy()


_EXCLUDE_PREFIX = {
    "Leírás": ["The user with id '-1'"],
    "Eseménykörnyezet": ["Címke:", "Más"],
}

_EXACT_EXCLUSIONS = {
    "Eseménykörnyezet": [
        "Címke: _Adatok_",
        "Címke: _Adatok_ (copy)",
        "Címke: _Adatok_ (copy) (copy)",
        "Címke: _EXTRA feladatok _",
        "Címke: _EXTRA feladatok _ (copy)",
        "Címke: _EXTRA feladatok _ (copy) (copy)",
        "Címke: _EXTRA feladatok _ (copy) (copy) (copy)",
        "Címke: _EXTRA feladatok _- Minta feladatsor Elmélet * A t...",
        "Címke: 1. HÁZI FELADAT A Scopus (https://www.scopus.com) ...",
        "Címke: 11:40-13:10 IDŐSÁV HALLGATÓI",
        "Címke: 2. BEADANDÓ FELADAT Készítse el a Jupyter notebook...",
        "Címke: 2025.JÚNIUS 12.",
        "Címke: 2025.JÚNIUS 4.",
        "Címke: 3. BEADANDÓ FELADAT (CSOPORTOS) Elkészültek a csop…",
        "Címke: 3. BEADANDÓ FELADAT (EGYÉNI TANRENDBEN TANULÓK SZÁ...",
        "Címke: 8:00-9:30 IDŐSÁV HALLGATÓI",
        "Címke: 9:50-11:20 IDŐSÁV HALLGATÓI",
        "Címke: A tárgy széles áttekintést ad a mesterséges intell...",
        "Címke: ELŐADÁSOK ANYAGA (copy)",
        "Címke: Érdeklődőknek",
        "Címke: Érdeklődőknek (copy)",
        "Címke: Érdeklődőknek (copy) (copy)",
        "Címke: Érdeklődőknek (copy) (copy) (copy)",
        "Címke: JELENLÉT",
        "Címke: KEDVEZMÉNYES TANULMÁNYI REND (KÜLFÖLDI RÉSZKÉPZÉSE...",
        "Címke: Label (copy)",
        "Címke: Label (copy) (copy)",
        "Címke: Label (copy) (copy) (copy)",
        "Címke: Label (copy) (copy) (copy) (copy)",
        "Címke: Label (copy) (copy) (copy) (copy) (copy)",
        "Címke: LOKÁLIS ÁLLOMÁNY ELÉRÉSE GOOGLE DRIVE-RÓL 1) Elérh...",
        "Címke: OKTATÓK ELÉRHETŐSÉGEI JAKOVÁC ANTAL (ELŐADÓ) Szoba...",
        "Címke: Programozási környezet kialakítása",
        "Kurzus: A mesterséges intelligencia alapjai (INSA011NMBB) Előadás (E01)",
        "Más",
    ]
}


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # prefix alapú kizárások
    for col, prefixes in _EXCLUDE_PREFIX.items():
        if col in out.columns:
            mask = False
            for px in prefixes:
                mask |= out[col].astype(str).str.startswith(px, na=False)
            out = out[~mask]
    # exact kizárások
    for col, values in _EXACT_EXCLUSIONS.items():
        if col in out.columns:
            out = out[~out[col].isin(values)]
    return out
