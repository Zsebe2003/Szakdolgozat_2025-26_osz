# Moodle Process Mining – modularizált Python projekt

## Gyors indítás

1) Python 3.10+ ajánlott. Telepítés:

```
pip install -r requirements.txt
```

2) Másold át a `.env.example` fájlt `.env` néven és állítsd be az `INPUT_XLSX` elérési utat.

3) Futtatás:

```
python main_preprocess.py      # tisztítás + szeletek + exportok
python main_analysis.py        # ábrák + statisztikák
python main_pm4py.py           # XES export + HM/DFG/Petri
```

## Kimenetek
- `data/processed/` – előállított CSV/XLSX szeletek  
- `data/xes/` – XES fájlok PM4Py-hez  
- `figures/` – ábrák (összesített és kategória-szintű)

## Környezeti változók

`.env` vagy `.env.example` fájlban:

```
INPUT_XLSX='xlsx elérhetősége'
START_DATE=2025-02-17 00:00:00
END_DATE=2025-06-23 23:59:59
```
