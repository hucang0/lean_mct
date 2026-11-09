"""Conceptual schematic, Letter Fig. 1: compact single-column vector diagram with minimal text.
(a) n-mode propagator diagram: 1 pairwise vertex vs. n relaxation channels -> 1/n dilution
(b) Real-space pair caging: Ghost pass-through (kappa=0) vs. Hard-core bond recoil (kappa!=0)
(c) 2D expansion plane: Multi-particle order N vs. Pair-exclusion order ell
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
from PIL import Image
from _style import OUT

PAL = dict(
    ink="#111111",
    ink_mid="#484848",
    ink_lt="#7E7E7E",
    cage_nb="#E2E2DF",
    cage_bd="#9E9E99",
    rule="#CCCCCC",
    bg_tint="#F5F5F3",
    blue="#1B5594",
    blue_lt="#D8E6F5",
    red="#B52F25",
    red_mid="#D94E34",
    red_lt="#F9DFDB",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42,
})


def draw_shaded_sphere(ax, x, y, r, facecolor, edgecolor, lw=1.15, ls="-", alpha=1.0,
                       hatch=None, specular=True, zorder=5):
    base = Circle((x, y), r, facecolor=facecolor, edgecolor="none", alpha=alpha, zorder=zorder)
    ax.add_patch(base)
    if specular:
        hl1 = Circle((x - 0.18 * r, y + 0.18 * r), 0.70 * r,
                     facecolor="white", edgecolor="none", alpha=0.22 * alpha, zorder=zorder + 0.1)
        hl2 = Circle((x - 0.30 * r, y + 0.30 * r), 0.23 * r,
                     facecolor="white", edgecolor="none", alpha=0.68 * alpha, zorder=zorder + 0.2)
        ax.add_patch(hl1)
        ax.add_patch(hl2)
    outline = Circle((x, y), r, facecolor="none", edgecolor=edgecolor,
                     linewidth=lw, linestyle=ls, hatch=hatch, zorder=zorder + 0.3)
    ax.add_patch(outline)


def draw_lens_intersection(ax, x1, y1, x2, y2, r, edgecolor, facecolor="none", hatch="xxxx", lw=1.0, zorder=6):
    d = np.hypot(x2 - x1, y2 - y1)
    if d >= 2 * r or d <= 0:
        return
    a = np.arccos(d / (2 * r))
    theta0 = np.arctan2(y2 - y1, x2 - x1)
    t1 = np.linspace(theta0 - a, theta0 + a, 60)
    t2 = np.linspace(theta0 + np.pi - a, theta0 + np.pi + a, 60)
    xs = np.concatenate([x1 + r * np.cos(t1), x2 + r * np.cos(t2)])
    ys = np.concatenate([y1 + r * np.sin(t1), y2 + r * np.sin(t2)])
    poly = Polygon(np.column_stack([xs, ys]), closed=True,
                   facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, lw=lw, zorder=zorder)
    ax.add_patch(poly)


def draw_cage_neighbors(ax, c1x, c2x, cy, r_nb=0.32, zorder=2):
    mid_x = 0.5 * (c1x + c2x)
    angles = np.deg2rad([28, 68, 112, 152, 208, 248, 292, 332])
    rx, ry = 2.22, 1.34
    for ang in angles:
        nx = mid_x + rx * np.cos(ang)
        ny = cy + ry * np.sin(ang)
        c = Circle((nx, ny), r_nb, facecolor=PAL["cage_nb"], edgecolor=PAL["cage_bd"],
                   lw=0.75, ls="-", alpha=0.85, zorder=zorder)
        ax.add_patch(c)


def make_cartoon():
    # Compact aspect ratio tailored for single-column (\columnwidth ~ 3.375 in)
    fig = plt.figure(figsize=(6.4, 2.35), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.90, 0.95], wspace=0.06,
                          left=0.01, right=0.99, bottom=0.02, top=0.94)

    # =========================================================================
    # PANEL (a): n-Mode Propagator Diagram (1/n Memory Dilution)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 8.8)
    ax_a.set_ylim(0.8, 8.6)
    ax_a.set_aspect("equal")
    ax_a.axis("off")

    ax_a.text(0.0, 8.35, "(a)", fontsize=14.5, fontweight="bold", color=PAL["ink"], va="bottom", ha="left")

    y_lines = [7.15, 5.10, 3.65, 1.85]
    x_left, x_right = 1.45, 6.45
    x_v = 3.95

    active_bg = mpatches.FancyBboxPatch(
        (1.90, 6.35), 4.10, 1.60, boxstyle="round,pad=0.06,rounding_size=0.18",
        facecolor=PAL["red_lt"], edgecolor=PAL["red"], lw=0.85, ls="-", alpha=0.45, zorder=1
    )
    ax_a.add_patch(active_bg)

    spec_bg = mpatches.FancyBboxPatch(
        (1.90, 1.35), 4.10, 4.25, boxstyle="round,pad=0.06,rounding_size=0.18",
        facecolor=PAL["bg_tint"], edgecolor=PAL["rule"], lw=0.85, ls="--", zorder=1
    )
    ax_a.add_patch(spec_bg)

    yy0 = y_lines[0]
    ax_a.plot([x_left, x_v], [yy0, yy0], color=PAL["ink"], lw=1.7, zorder=3)
    ax_a.annotate("", xy=(2.95, yy0), xytext=(2.65, yy0),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.4, mutation_scale=10), zorder=4)
    for dy_br in [0.42, -0.42]:
        ax_a.plot([x_v, x_right], [yy0, yy0 + dy_br], color=PAL["ink"], lw=1.6, zorder=3)
        xm_br = 5.30
        ym_br = yy0 + dy_br * (xm_br - x_v) / (x_right - x_v)
        ax_a.annotate("", xy=(xm_br + 0.15, ym_br + dy_br * 0.15 / (x_right - x_v)),
                      xytext=(xm_br - 0.15, ym_br - dy_br * 0.15 / (x_right - x_v)),
                      arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.3, mutation_scale=9.5), zorder=4)
    draw_shaded_sphere(ax_a, x_left - 0.32, yy0, 0.24, facecolor=PAL["red_mid"],
                       edgecolor=PAL["ink"], lw=1.0, zorder=5)
    ax_a.text(x_left - 0.65, yy0, r"$\mathbf{k}_1$", fontsize=12.5, color=PAL["ink"], ha="right", va="center")
    ax_a.add_patch(Circle((x_v, yy0), 0.21, facecolor=PAL["red"], edgecolor="white", lw=0.85, zorder=6))
    ax_a.text(x_v - 0.25, yy0 + 0.44, r"$\widetilde{W}$",
              fontsize=13.0, fontweight="bold", color=PAL["red"], ha="right", va="center")

    for yy, lbl in [(y_lines[1], r"$\mathbf{k}_2$"), (y_lines[2], r"$\mathbf{k}_3$"), (y_lines[3], r"$\mathbf{k}_n$")]:
        ax_a.plot([x_left, x_right], [yy, yy], color=PAL["blue"], lw=1.35, ls="--", zorder=3)
        for xm in [2.75, 5.30]:
            ax_a.annotate("", xy=(xm + 0.15, yy), xytext=(xm - 0.15, yy),
                          arrowprops=dict(arrowstyle="-|>", color=PAL["blue"], lw=1.2, mutation_scale=9), zorder=4)
        draw_shaded_sphere(ax_a, x_left - 0.32, yy, 0.22, facecolor=PAL["blue_lt"],
                           edgecolor=PAL["blue"], lw=0.95, specular=False, zorder=5)
        ax_a.text(x_left - 0.65, yy, lbl, fontsize=12.5, color=PAL["blue"], ha="right", va="center")
        ax_a.add_patch(Circle((x_v, yy), 0.12, facecolor="white", edgecolor=PAL["blue"], lw=1.0, zorder=5))

    for yd in [2.98, 2.75, 2.52]:
        ax_a.add_patch(Circle((x_v, yd), 0.055, facecolor=PAL["blue"], edgecolor="none", zorder=5))
        ax_a.add_patch(Circle((x_left - 0.32, yd), 0.05, facecolor=PAL["blue"], edgecolor="none", zorder=5))

    bx = 6.72
    ax_a.plot([bx - 0.18, bx, bx, bx - 0.18],
              [y_lines[3] - 0.15, y_lines[3] - 0.15, y_lines[0] + 0.15, y_lines[0] + 0.15],
              color=PAL["ink_mid"], lw=1.2, zorder=4)
    ax_a.plot([bx, bx + 0.14], [4.50, 4.50], color=PAL["ink_mid"], lw=1.2, zorder=4)
    ax_a.text(7.85, 4.50, r"$\propto 1/n$", fontsize=13.0, fontweight="bold", color=PAL["ink"], ha="center", va="center")

    # =========================================================================
    # PANEL (b): Real-Space Pair Caging: Ghost vs. Hard-Core
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0.2, 7.8)
    ax_b.set_ylim(0.8, 8.6)
    ax_b.set_aspect("equal")
    ax_b.axis("off")

    ax_b.text(0.2, 8.35, "(b)", fontsize=14.5, fontweight="bold", color=PAL["ink"], va="bottom", ha="left")

    c1x, c2x = 2.65, 4.35
    mid_x = 0.5 * (c1x + c2x)
    r_cage = 1.06

    # TOP HALF: Ghost pass-through
    cy_top = 6.65
    draw_cage_neighbors(ax_b, c1x, c2x, cy_top, r_nb=0.28, zorder=1)
    for cx in [c1x, c2x]:
        ax_b.add_patch(Circle((cx, cy_top), r_cage, facecolor=PAL["blue_lt"], alpha=0.32,
                              edgecolor=PAL["blue"], lw=0.85, ls=":", zorder=2))

    r_part = 0.72
    p1x_top, p2x_top = mid_x - 0.42, mid_x + 0.42
    draw_shaded_sphere(ax_b, p1x_top, cy_top, r_part, facecolor=PAL["blue_lt"],
                       edgecolor=PAL["blue"], lw=1.25, ls="--", alpha=0.68, specular=False, zorder=3)
    draw_shaded_sphere(ax_b, p2x_top, cy_top, r_part, facecolor=PAL["blue_lt"],
                       edgecolor=PAL["blue"], lw=1.25, ls="--", alpha=0.68, specular=False, zorder=3)
    draw_lens_intersection(ax_b, p1x_top, cy_top, p2x_top, cy_top, r_part,
                           edgecolor=PAL["ink"], facecolor="none", hatch="xxxx", lw=0.9, zorder=5)

    ax_b.annotate("", xy=(mid_x + 0.88, cy_top + 0.19), xytext=(mid_x - 0.88, cy_top + 0.19),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["blue"], lw=1.35, ls="--", mutation_scale=9.5), zorder=6)
    ax_b.annotate("", xy=(mid_x - 0.88, cy_top - 0.19), xytext=(mid_x + 0.88, cy_top - 0.19),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["blue"], lw=1.35, ls="--", mutation_scale=9.5), zorder=6)

    ax_b.text(5.95, cy_top + 0.34, r"$\boldsymbol{\kappa}=\mathbf{0}$",
              fontsize=12.2, fontweight="bold", color=PAL["blue"], ha="left", va="center")
    ax_b.text(5.95, cy_top - 0.38, r"$Z\equiv 1$",
              fontsize=12.0, color=PAL["blue"], ha="left", va="center")

    ax_b.plot([0.5, 7.3], [4.65, 4.65], color=PAL["rule"], lw=0.75)

    # BOTTOM HALF: Hard-core reflection
    cy_bot = 2.65
    draw_cage_neighbors(ax_b, c1x, c2x, cy_bot, r_nb=0.28, zorder=1)
    for cx in [c1x, c2x]:
        ax_b.add_patch(Circle((cx, cy_bot), r_cage, facecolor=PAL["bg_tint"], alpha=0.75,
                              edgecolor=PAL["ink_mid"], lw=0.85, ls=":", zorder=2))

    barrier = mpatches.Rectangle((mid_x - 0.10, cy_bot - 0.98), 0.20, 1.96,
                                 facecolor="white", edgecolor=PAL["ink"], hatch="////", lw=0.8, zorder=3)
    ax_b.add_patch(barrier)
    ax_b.plot([mid_x, mid_x], [cy_bot - 1.04, cy_bot + 1.04], color=PAL["ink"], lw=1.6, zorder=5)

    r_hc = 0.70
    s1x, s2x = mid_x - r_hc - 0.03, mid_x + r_hc + 0.03
    draw_shaded_sphere(ax_b, s1x, cy_bot, r_hc, facecolor=PAL["red_mid"],
                       edgecolor=PAL["ink"], lw=1.3, ls="-", alpha=0.95, specular=True, zorder=4)
    draw_shaded_sphere(ax_b, s2x, cy_bot, r_hc, facecolor=PAL["red_mid"],
                       edgecolor=PAL["ink"], lw=1.3, ls="-", alpha=0.95, specular=True, zorder=4)

    ax_b.annotate("", xy=(s1x - 1.02, cy_bot), xytext=(s1x - 0.12, cy_bot),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["red"], lw=2.1, mutation_scale=11), zorder=6)
    ax_b.annotate("", xy=(s2x + 1.02, cy_bot), xytext=(s2x + 0.12, cy_bot),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["red"], lw=2.1, mutation_scale=11), zorder=6)

    ax_b.text(5.95, cy_bot + 0.34, r"$\boldsymbol{\kappa}\neq\mathbf{0}$",
              fontsize=12.2, fontweight="bold", color=PAL["red"], ha="left", va="center")
    ax_b.text(5.95, cy_bot - 0.38, r"$1/Z\geq 1$",
              fontsize=12.0, color=PAL["red"], ha="left", va="center")

    # =========================================================================
    # PANEL (c): 2D Theory Space (N vs. \ell)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0.2, 8.2)
    ax_c.set_ylim(0.8, 8.6)
    ax_c.set_aspect("equal")
    ax_c.axis("off")

    ax_c.text(0.2, 8.35, "(c)", fontsize=14.5, fontweight="bold", color=PAL["ink"], va="bottom", ha="left")

    ox, oy = 1.65, 1.95
    W, H = 6.00, 6.00

    xs_N = [2.35, 3.55, 4.75, 5.95, 7.15]
    ys_n = [2.60, 3.55, 4.50, 5.45, 6.40, 7.35]
    for xg in xs_N:
        ax_c.plot([xg, xg], [oy, oy + H - 0.2], color=PAL["rule"], lw=0.45, zorder=0)
    for yg in ys_n:
        ax_c.plot([ox, ox + W - 0.2], [yg, yg], color=PAL["rule"], lw=0.45, zorder=0)

    ax_c.annotate("", xy=(ox + W + 0.15, oy), xytext=(ox, oy),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.35, mutation_scale=10), zorder=4)
    ax_c.annotate("", xy=(ox, oy + H + 0.15), xytext=(ox, oy),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.35, mutation_scale=10), zorder=4)

    ax_c.text(ox + 0.52 * W, 0.90, r"order $N$",
              fontsize=12.5, fontweight="bold", color=PAL["ink"], ha="center", va="center")
    ax_c.text(0.38, oy + 0.5 * H, r"order $\ell$",
              fontsize=12.5, fontweight="bold", color=PAL["ink"], ha="center", va="center", rotation=90)

    for xg, n_val in zip(xs_N, [2, 3, 4, 5, 6]):
        ax_c.plot([xg, xg], [oy, oy + 0.14], color=PAL["ink"], lw=0.8)
        ax_c.text(xg, oy - 0.40, f"${n_val}$", fontsize=11.5, color=PAL["ink"], ha="center", va="center")

    n_tick_labels = ["$1$", "$2$", "$3$", "$5$", "$10$", r"$\infty$"]
    for yg, n_lbl in zip(ys_n, n_tick_labels):
        ax_c.plot([ox, ox + 0.14], [yg, yg], color=PAL["ink"], lw=0.8)
        ax_c.text(ox - 0.20, yg, n_lbl, fontsize=11.0, color=PAL["ink"], ha="right", va="center")

    x0 = xs_N[0]
    y0 = ys_n[0]
    ax_c.plot([x0, x0], [y0, ys_n[-1]], color=PAL["red"], lw=2.3, zorder=4)
    ax_c.annotate("", xy=(x0, ys_n[-1] - 0.05), xytext=(x0, ys_n[-2]),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["red"], lw=2.3, mutation_scale=11), zorder=4)

    for yg in ys_n[1:]:
        is_top = (yg == ys_n[-1])
        rad = 0.22 if is_top else 0.16
        ax_c.add_patch(Circle((x0, yg), rad, facecolor=PAL["red"], edgecolor="white", lw=0.8, zorder=6))

    ax_c.text(x0 + 0.36, ys_n[-1], r"$\varphi_G=1.78$", fontsize=11.8,
              fontweight="bold", color=PAL["red"], ha="left", va="center")

    ax_c.plot([xs_N[0], xs_N[-1]], [y0, y0], color=PAL["blue"], lw=1.9, ls="--", zorder=3)
    ax_c.annotate("", xy=(xs_N[-1] + 0.65, y0), xytext=(xs_N[-1], y0),
                  arrowprops=dict(arrowstyle="-|>", color=PAL["blue"], lw=1.9, ls="--", mutation_scale=11), zorder=3)

    for xg in xs_N:
        sq = mpatches.Rectangle((xg - 0.17, y0 - 0.17), 0.34, 0.34,
                                facecolor="white", edgecolor=PAL["blue"], lw=1.6, zorder=6)
        ax_c.add_patch(sq)

    ax_c.text(xs_N[0] + 0.28, y0 + 0.35, "2.80", fontsize=11.2, color=PAL["blue"],
              fontweight="bold", ha="left", va="bottom")
    ax_c.text(xs_N[-1], y0 + 0.35, "4.49", fontsize=11.2, color=PAL["blue"],
              fontweight="bold", ha="center", va="bottom")

    tiff_path = f"{OUT}/fig0_cartoon.tiff"
    jpg_path = f"{OUT}/fig0_cartoon.jpg"
    fig.savefig(tiff_path, dpi=600, format="tiff")
    fig.savefig(f"{OUT}/fig0_cartoon.png", dpi=300)
    fig.savefig(f"{OUT}/fig0_cartoon.pdf")
    fig.savefig(f"{OUT}/fig0_cartoon_v2.png", dpi=300)
    fig.savefig(f"{OUT}/fig0_cartoon_v2.pdf")
    plt.close(fig)

    with Image.open(tiff_path) as im:
        rgb = im.convert("RGB")
        rgb.save(jpg_path, format="JPEG", quality=96, dpi=(600, 600))

    print("Saved:", tiff_path, "and converted to", jpg_path)


if __name__ == "__main__":
    make_cartoon()
