import pandas as pd

def print_basic_stats(df: pd.DataFrame):
    print("\n== Alap statisztikák ==")
    
    # Sorok és oszlopok száma
    print(f"Sorok száma: {len(df)}")
    print(f"Oszlopok száma: {df.shape[1]}")   # <-- EZ ÍRJA KI HÁNY OSZLOP VAN
    print(f"Oszlopnevek: {list(df.columns)}")

    if "user_id" in df.columns:
        vc = df["user_id"].value_counts(dropna=True)
        print("\n== user_id statisztikák ==")
        print("Egyedi user_id-k:", vc.index.nunique())
        print("user_id gyakoriság leírók:\n", vc.describe())

    for c in ["Esemény neve", "Uj_oszlop", "Munka_típus", 
              "Munka_típus_keresztmetszet", "Munka_3utas"]:
        if c in df.columns:
            print(f"\n{c} – értékek száma:")
            print(df[c].value_counts(dropna=False).head(10))
