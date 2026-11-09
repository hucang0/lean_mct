"""Shared matplotlib style for the MK paper figures (CPU only, no solver runs).
Palette: a colour-blind-safe categorical set
(blue, orange, aqua, yellow, magenta, green, violet, red), assigned in fixed order."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent          # this figure folder; data live in ../../data
_DATA = _HERE.parent.parent / "data"
NUM   = str(_DATA / "numerics")
SIM   = str(_DATA / "simulation")
STEP6 = str(_DATA / "brc")     # exact Gaussian replica curve vs. Mari-Kurchan Fig. 18 (SM Sec. S5 E)
RES   = str(_DATA / "restore")
OUT   = str(_HERE)

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", muted="#9a9890", grid="#e6e5e0")
# roles used throughout the paper
COL_BSELF = C["blue"]; COL_BCOLL = C["orange"]; COL_PY = C["aqua"]
COL_PHID  = C["ink2"]; COL_PERRON = C["violet"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

def panel_label(ax, s, x=-0.18, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    print("wrote", f"{OUT}/{name}.pdf")
