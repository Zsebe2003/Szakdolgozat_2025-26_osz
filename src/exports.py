import pandas as pd
from pathlib import Path


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"CSV mentve: {path}")


def save_xlsx(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pandas 2.x alatt az engine_kwargs az ajánlott mód a writer-nek átadni opciókat
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter", engine_kwargs={"options": {"strings_to_urls": False}}) as w:
            df.to_excel(w, index=False, sheet_name="df_remaining")
    except TypeError:
        # Fallback, ha az adott környezet nem ismeri az engine_kwargs-ot
        with pd.ExcelWriter(path, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="df_remaining")
            # XlsxWriter globális opció (kiterjesztett), nem minden verzióban érhető el
            try:
                w.book.strings_to_urls = False  # type: ignore[attr-defined]
            except Exception:
                pass
    print(f"XLSX mentve: {path}")
