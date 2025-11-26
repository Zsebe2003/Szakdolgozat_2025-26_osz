import pandas as pd
from typing import Dict

# Eseménykörnyezet → tantervi hét (string) mapping
WEEK_PLAN: Dict[str, str] = {
    # Órai / előadás jellegű elemek
    "Fájl: Chatbot.txt": "11. hét",
    "Fájl: DeepSeek AGI": "1. hét",
    "Fájl: Előadás anyaga": "1. hét",
    "Fájl: Generatív AI alapjai - LSTM": "12. hét",
    "Fájl: Idősorok": "1. hét",
    "Fájl: Iris_Data.csv": "1. hét",
    "Fájl: Klasszifikáció": "2. hét",
    "Fájl: LLM": "1. hét",
    "Fájl: Modellek általánosítása, regresszió": "3. hét",
    "Fájl: Nem felügyelt gépi tanítás - Cluster": "4. hét",
    "Fájl: Nem felügyelt gépi tanítás - Dimenzió csökkentés": "5. hét",
    "Fájl: Neurális hálózatok": "7. hét",
    "Fájl: Pandas (ismétlés)": "1. hét",
    "Fájl: Természetes nyelvi feldolgozás alapjai": "8. hét",
    "Fájl: Természetes nyelvi feldolgozás II": "11. hét",

    "Mappa: Előadás anyagok": "Nem kapcsolódik héthez",
    "Oldal: Előadások és gyakorlatok hanganyagai": "Nem kapcsolódik héthez",

    # URL / Colab / segédanyagok
    "URL: A Kaggle egy site ahol számtalan gépi tanulási feladatot és hozzátartozó adatbázist érhetünk el.": "1. hét",
    "URL: Anaconda telepítés - Windows": "1. hét",
    "URL: Anaconda telepítése": "1. hét",
    "URL: Anaconda telepítése - MacOS": "1. hét",
    "URL: Bevezetés a Jupyter Notebook használatába": "1. hét",
    "URL: Chatbot - Colab": "11. hét",
    "URL: Csoportosítás  (Colab)": "4. hét",
    "URL: Dimenzió csökkentés (Colab)": "5. hét",
    "URL: Dimenzió csökkentés (Gyakoró feladat megoldása; Colab)": "5. hét",
    "URL: Hogyan működik a neurális hálózat": "7. hét",
    "URL: Iris_Data.csv (http://web.uni-corvinus.hu/~fszabina/data/Iris_Data.csv)": "1. hét",
    "URL: Klasszifikáció (Colab)": "2. hét",
    "URL: Klasszifikáció (Gyakorló feladatok megoldása; Colab)": "2. hét",
    "URL: Klasszifikáció adat Orange_Telecom_Churn_Data.csv": "2. hét",
    "URL: Magyar női utónév generátor- Colab": "12. hét",
    "URL: Modellek általánosítása; regresszió": "3. hét",
    "URL: Modellek általánosítása; regresszió (Gyakorló feladatok megoldás; Colab)": "3. hét",
    "URL: Neurális hálózatok - Colab": "7. hét",
    "URL: Neurális hálózatok - Colab (megoldás)": "7. hét",
    "URL: Név generátor- Colab": "12. hét",
    "URL: NLP - Colab": "8. hét",
    "URL: NLP - megoldás (Colab)": "8. hét",
    "URL: Pandas gakorlat megoldása (Colab)": "1. hét",
    "URL: Pandas gyakorlat (Colab)": "1. hét",
    "URL: Regresszió adat Ames_Housing_Sales.csv": "3. hét",
    "URL: Regresszió gyakorló feladat": "3. hét",
    "URL: Regresszió gyakorló feladat (megoldás)": "3. hét",
    "URL: Utasszám előrejelzés (LSTM)- Colab": "12. hét",
    "URL: ChatGPT + saját dokumentum - Colab": "11. hét",
    "URL: Csoportosítás (Gyakorló feladat megoldása; Colab)": "4. hét",
    "URL: EXTRA - Gyakorló feladatok": "2. hét",
    "URL: Gyakorló feladat (zh-ra készülés) - Felügyelt tanulás": "4. hét",
    "URL: Gyakorló feladat (zh-ra készülés) - Felügyelt tanulás (megoldás)": "4. hét",
    "URL: Gyakorló feladatok (ZH-ra készülés) - nem felügyelt gépi tanítás": "5. hét",
    "URL: Gyakorló feladatok (ZH-ra készülés) - nem felügyelt gépi tanítás (megoldás)": "5. hét",
    "URL: How neural network works": "7. hét",
    "URL: Keras": "12. hét",
    "URL: Legjobb regressziós modell kiválasztása": "3. hét",
    "URL: Magyar nyelvű szövegek előfeldolgozása": "8. hét",
    "URL: NLP online könyv": "8. hét",
    "URL: Playground - Neurális hálózatok": "7. hét",
    "URL: Practical Deep Learning": "7. hét",
    "URL: Tensorflow": "12. hét",
    "URL: Train/test vágás": "3. hét",
    "URL: Túltanulás/alultanulás": "3. hét",

    # Extra / minta ZH / egyéb
    "Fájl: ChatGPT+saját dokumentum": "11. hét",
    "Fájl: EXTRA felatok - Pandas": "1. hét",
    "Fájl: EXTRA felatok - Pandas feladat adatai": "1. hét",
    "Fájl: MintaZH - gyakolat (adatok)": "5. hét",
    "Fájl: MintaZH - gyakorlat (feladatsor)": "5. hét",
    "Fájl: Modell értékelés (Racskó Péter)": "3. hét",
    "Teszt: MintaZH - elmélet": "5. hét",
}

def attach_week_plan_columns(
    df: pd.DataFrame,
    src_col: str = "Eseménykörnyezet",
    label_col: str = "Tantervi_hét",
    num_col: str = "Tantervi_hét_szám"
) -> pd.DataFrame:
    """
    Hozzáad két oszlopot:
      - Tantervi_hét   (pl. '1. hét', 'Nem kapcsolódik héthez', 'Ismeretlen')
      - Tantervi_hét_szám (pl. 1..15 vagy NaN, ha nem kapcsolódik/ismeretlen)
    """
    out = df.copy()
    # szöveges címke
    out[label_col] = out[src_col].map(WEEK_PLAN).fillna("Ismeretlen")

    # numerikus hét kinyerése
    def _parse_week_num(x: str):
        x = str(x)
        if x.endswith("hét") or "hét" in x:
            # '12. hét' vagy '1. hét' formátum
            try:
                return int(x.split(".")[0])
            except Exception:
                return pd.NA
        return pd.NA

    out[num_col] = out[label_col].map(_parse_week_num)
    return out
