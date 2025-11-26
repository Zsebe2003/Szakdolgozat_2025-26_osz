import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
from pathlib import Path
import matplotlib.dates as mdates

# =============================================================================
# STYLUS BEÁLLÍTÁSOK
# =============================================================================

plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'figure.titlesize': 12,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (10, 6),
    'figure.autolayout': True
})

# SZÍNEK
CAT_COLORS_ALL = {
    "Orai": "#FFD700",
    "Hazi": "#FF8C00", 
    "Admin": "#2E8B57",
    "Szamonkeres": "#DC143C",
    "Extra": "#9370DB",
    "Egyéb": "#708090",
}

ORAI_COLORS = {
    "Órai": "#FFD700",
    "Otthoni": "#4169E1"
}

COLORS_3 = {
    "Otthoni": "#4169E1",
    "Órai – nem számonkérés": "#FFD700",
    "Órai – számonkérés": "#DC143C", 
    "Egyéb": "#708090"
}

# =============================================================================
# SEGÉDFÜGGVÉNYEK
# =============================================================================

def _ensure_out(out_dir: Path):
    """Kimeneti könyvtár biztosítása"""
    out_dir.mkdir(parents=True, exist_ok=True)

def _create_pretty_axis(ax, title, xlabel, ylabel, rotation=0):
    """Egységes tengely formázás"""
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    if rotation > 0:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _create_legend_plot(colors_dict, title, ax):
    """Általános legenda készítése"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Cím
    ax.text(0.5, 0.95, title, fontsize=12, fontweight='bold', 
            ha='center', va='top', transform=ax.transAxes)
    
    # Színes dobozok és címkék
    y_positions = np.linspace(0.8, 0.2, len(colors_dict))
    
    for i, (label, color) in enumerate(colors_dict.items()):
        y = y_positions[i]
        
        # Szín doboz
        ax.add_patch(plt.Rectangle((0.1, y-0.03), 0.1, 0.06, 
                                 facecolor=color, edgecolor='black', linewidth=0.5))
        
        # Címke
        ax.text(0.25, y, label, fontsize=10, ha='left', va='center', 
                transform=ax.transAxes)

# =============================================================================
# FŐ PLOTTING FUNKCIÓK - LEGENDÁVAL
# =============================================================================

def save_four_subplots_category(df: pd.DataFrame, out_dir: Path):
    """4 alminta kategóriák szerint - 4. helyén legenda"""
    _ensure_out(out_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    parts = ("hónap", "hét", "óra")  # Csak 3 tényleges ábra
    
    # Első 3 ábra - adatok
    for i, col in enumerate(parts):
        if col not in df.columns:
            axes[i].axis("off")
            axes[i].text(0.5, 0.5, f"Hiányzik: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            continue
            
        # Adatok előkészítése
        grouped = (df.dropna(subset=[col, "Uj_oszlop"])
                  .groupby([col, "Uj_oszlop"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
        
        if grouped.empty:
            axes[i].text(0.5, 0.5, f"Nincs adat: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{col.capitalize()} - Nincs adat", fontsize=11)
            continue
        
        # Plot készítése - LEGENDA NÉLKÜL
        colors = [CAT_COLORS_ALL.get(c, "#708090") for c in grouped.columns]
        grouped.plot(kind="bar", stacked=True, ax=axes[i], color=colors, width=0.8, legend=False)
        
        axes[i].set_title(f"Események {col} szerint", fontweight='bold', fontsize=11)
        axes[i].set_xlabel(col.capitalize())
        axes[i].set_ylabel("Események száma")
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        axes[i].tick_params(axis='x', rotation=45)
    
    # 4. ábra helyén legenda
    _create_legend_plot(CAT_COLORS_ALL, "Kategóriák Jelmagyarázata", axes[3])
    
    plt.tight_layout()
    out_path = out_dir / "osszegzo_kategoriak_legenda.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_four_subplots_orai(df: pd.DataFrame, out_dir: Path):
    """4 alminta órai vs otthoni szerint - 4. helyén legenda"""
    _ensure_out(out_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    parts = ("hónap", "hét", "óra")  # Csak 3 tényleges ábra
    
    # Első 3 ábra - adatok
    for i, col in enumerate(parts):
        if col not in df.columns:
            axes[i].axis("off")
            axes[i].text(0.5, 0.5, f"Hiányzik: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            continue
            
        # Adatok előkészítése
        grouped = (df.dropna(subset=[col, "Munka_típus"])
                  .groupby([col, "Munka_típus"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
        
        if grouped.empty:
            axes[i].text(0.5, 0.5, f"Nincs adat: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{col.capitalize()} - Nincs adat", fontsize=11)
            continue
        
        # Plot készítése - LEGENDA NÉLKÜL
        colors = [ORAI_COLORS.get(c, "#708090") for c in grouped.columns]
        grouped.plot(kind="bar", stacked=True, ax=axes[i], color=colors, width=0.8, legend=False)
        
        axes[i].set_title(f"Órai vs otthoni - {col}", fontweight='bold', fontsize=11)
        axes[i].set_xlabel(col.capitalize())
        axes[i].set_ylabel("Események száma")
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        axes[i].tick_params(axis='x', rotation=45)
    
    # 4. ábra helyén legenda
    _create_legend_plot(ORAI_COLORS, "Munka Típusa Jelmagyarázata", axes[3])
    
    plt.tight_layout()
    out_path = out_dir / "osszegzo_orai_otthoni_legenda.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_four_subplots_3way(df: pd.DataFrame, out_dir: Path):
    """4 alminta 3 kategóriás megjelenítéssel - 4. helyén legenda"""
    _ensure_out(out_dir)
    
    if "Munka_3utas" not in df.columns:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    parts = ("hónap", "hét", "óra")  # Csak 3 tényleges ábra
    
    # Első 3 ábra - adatok
    for i, col in enumerate(parts):
        if col not in df.columns:
            axes[i].axis("off")
            axes[i].text(0.5, 0.5, f"Hiányzik: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            continue
            
        # Adatok előkészítése
        grouped = (df.dropna(subset=[col, "Munka_3utas"])
                  .groupby([col, "Munka_3utas"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
        
        if grouped.empty:
            axes[i].text(0.5, 0.5, f"Nincs adat: {col}", 
                        ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{col.capitalize()} - Nincs adat", fontsize=11)
            continue
        
        # Rendezés a COLORS_3 sorrendje szerint
        order = [k for k in COLORS_3.keys() if k in grouped.columns]
        grouped = grouped.reindex(columns=order, fill_value=0)
        
        # Plot készítése - LEGENDA NÉLKÜL
        colors = [COLORS_3.get(c, "#708090") for c in grouped.columns]
        grouped.plot(kind="bar", stacked=True, ax=axes[i], color=colors, width=0.8, legend=False)
        
        axes[i].set_title(f"3 kategória - {col}", fontweight='bold', fontsize=11)
        axes[i].set_xlabel(col.capitalize())
        axes[i].set_ylabel("Események száma")
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        axes[i].tick_params(axis='x', rotation=45)
    
    # 4. ábra helyén legenda
    _create_legend_plot(COLORS_3, "3 Kategóriás Jelmagyarázata", axes[3])
    
    plt.tight_layout()
    out_path = out_dir / "osszegzo_3kategoria_legenda.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# EGYÉB PLOTTING FUNKCIÓK (LEGENDA ELTÁVOLÍTÁSÁVAL)
# =============================================================================

def plot_timeparts_stacked_by_category(df: pd.DataFrame, out_dir: Path, cols=("hónap","hét","óra")):
    """Időrészek szerinti halmozott oszlopdiagramok kategóriánként"""
    _ensure_out(out_dir)
    
    for col in cols:
        if col not in df.columns:
            continue
            
        grouped = (df.dropna(subset=[col, "Uj_oszlop"])
                  .groupby([col, "Uj_oszlop"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
        
        if grouped.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = [CAT_COLORS_ALL.get(c, "#708090") for c in grouped.columns]
        
        # LEGENDA NÉLKÜL
        grouped.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8, legend=False)
        
        title_map = {"hónap": "hónap", "hét": "hét", "óra": "óra"}
        _create_pretty_axis(ax, 
                           f"Események eloszlása {title_map[col]} szerint", 
                           title_map[col].capitalize(), 
                           "Események száma")
        
        plt.tight_layout()
        out_path = out_dir / f"esemenyek_{col}_kategoriak.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_timeparts_stacked_by_orai_otthoni(df: pd.DataFrame, out_dir: Path, cols=("hónap","hét","óra")):
    """Órai vs otthoni munka időrészek szerint"""
    _ensure_out(out_dir)
    
    for col in cols:
        if col not in df.columns:
            continue
            
        grouped = (df.dropna(subset=[col, "Munka_típus"])
                  .groupby([col, "Munka_típus"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
        
        if grouped.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = [ORAI_COLORS.get(c, "#708090") for c in grouped.columns]
        
        # LEGENDA NÉLKÜL
        grouped.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8, legend=False)
        
        title_map = {"hónap": "hónap", "hét": "hét", "óra": "óra"}
        _create_pretty_axis(ax, 
                           f"Órai vs otthoni munka {title_map[col]} szerint", 
                           title_map[col].capitalize(), 
                           "Események száma")
        
        plt.tight_layout()
        out_path = out_dir / f"orai_otthoni_{col}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_monthly_bars_orai_otthoni(df: pd.DataFrame, out_dir: Path):
    """Havi bontás - Órai vs Otthoni"""
    _ensure_out(out_dir)
    
    df_copy = df.copy()
    df_copy["Idő_dt"] = pd.to_datetime(df_copy["Idő_dt"], errors="coerce")
    df_copy = df_copy.dropna(subset=["Idő_dt"])
    
    if df_copy.empty:
        return
    
    monthly = (df_copy.groupby([df_copy["Idő_dt"].dt.strftime('%Y-%m'), "Munka_típus"])
               .size()
               .unstack(fill_value=0))
    
    start_date = df_copy["Idő_dt"].min().replace(day=1)
    end_date = df_copy["Idő_dt"].max().replace(day=1)
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    monthly = monthly.reindex(all_months, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [ORAI_COLORS.get(c, "#708090") for c in monthly.columns]
    
    # LEGENDA NÉLKÜL
    monthly.plot(kind="bar", ax=ax, color=colors, width=0.8, legend=False)
    
    _create_pretty_axis(ax, 
                       "Havi eseményszám - Órai vs Otthoni", 
                       "Hónap", 
                       "Események száma",
                       rotation=45)
    
    plt.tight_layout()
    
    out_path = out_dir / "havi_orai_otthoni.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()