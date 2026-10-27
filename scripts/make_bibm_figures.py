"""Generate BIBM manuscript figures from audited result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "bibm2026_label_decodability_audit" / "figures"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "figure.dpi": 160,
    }
)


COLORS = {
    "c1": "#4D4D4D",
    "c2": "#5A8DBE",
    "c4": "#222222",
    "c5d": "#2C7040",
    "c6d": "#4F3D78",
    "geom": "#D98C30",
    "fixed_circle": "#4F9B62",
    "fixed_square": "#7B61A8",
    "fixed_ellipse": "#B45A5A",
    "gray": "#777777",
    "light_gray": "#E8E8E8",
    "ultrasam": "#B0B0B0",
    "ink": "#202020",
}


PRIMARY_LABELS = {
    "C1_full": "Full\n(C1)",
    "C2_lesion-only": "Lesion-only\n(C2)",
    "C4_lesion-removed-inpaint": "Lesion\nremoved\n(C4)",
    "C5d_dilated-mask-only": "Dilated\nmask\n(C5d)",
    "C6d_dilated-geometry-rendering": "Dilated\ngeometry\n(C6d)",
    "dilated_mask_geometry_features": "Measured\ngeometry",
}


POINT_STYLE = {
    "C1_full": (COLORS["c1"], "o"),
    "C2_lesion-only": (COLORS["c2"], "s"),
    "C4_lesion-removed-inpaint": (COLORS["c4"], "D"),
    "C5d_dilated-mask-only": (COLORS["c5d"], "^"),
    "C6d_dilated-geometry-rendering": (COLORS["c6d"], "P"),
    "dilated_mask_geometry_features": (COLORS["geom"], "X"),
}


def load_json(rel_path: str) -> dict[str, Any]:
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: mpl.figure.Figure, stem: str, output_path: Path | None = None) -> None:
    if output_path is None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def use_template_figure_labels() -> None:
    """Match the supplied conference template's 8 pt Times New Roman labels."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def asymmetric_errors(values: Iterable[float], cis: Iterable[tuple[float, float]]) -> np.ndarray:
    vals = np.asarray(list(values), dtype=float)
    ci_arr = np.asarray(list(cis), dtype=float)
    return np.vstack([vals - ci_arr[:, 0], ci_arr[:, 1] - vals])


def load_image(rel_path: str) -> np.ndarray:
    image = Image.open(ROOT / rel_path).convert("L")
    arr = np.asarray(image)
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        arr = arr.astype(float) / 255.0
    return arr


def add_box(ax: mpl.axes.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=0.9,
    )
    ax.add_patch(box)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=5.8)


def add_arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def make_fig1() -> None:
    sample = "bus_0001-l.png"
    panels = [
        ("Full\nimage", f"images/BUS-BRA/C1_full/{sample}"),
        ("Lesion\nremoved", f"images/BUS-BRA/C4_lesion-removed-inpaint/{sample}"),
        ("Dilated\nmask", f"images/BUS-BRA/C5d_dilated-mask-only/{sample}"),
        ("Geometry\nrendering", f"images/BUS-BRA/C6d_dilated-geometry-rendering/{sample}"),
        ("Fixed-circle\nedit", f"images/BUS-BRA/C10_fixed-circle-inpaint/{sample}"),
    ]

    fig = plt.figure(figsize=(7.05, 3.35))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 0.66], hspace=0.27, wspace=0.16)

    for idx, (title, rel_path) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(load_image(rel_path), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, pad=2, fontsize=6.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color("#BBBBBB")

    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.02,
        0.88,
        "Mask-conditioned lesion removal has three candidate signal paths",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    add_box(ax, (0.03, 0.61), (0.11, 0.16), "$X$\nimage", COLORS["c1"])
    add_box(ax, (0.03, 0.38), (0.11, 0.16), "$M$\nmask", COLORS["geom"])
    add_box(ax, (0.20, 0.49), (0.12, 0.16), "$T$\nedit", COLORS["ink"])
    add_box(ax, (0.82, 0.49), (0.13, 0.16), "$Z=T(X,M)$\nremoved", COLORS["ink"])
    add_arrow(ax, (0.14, 0.69), (0.20, 0.58), COLORS["gray"])
    add_arrow(ax, (0.14, 0.46), (0.20, 0.56), COLORS["gray"])
    add_arrow(ax, (0.32, 0.57), (0.82, 0.57), COLORS["gray"])

    source_boxes = [
        ((0.43, 0.69), "extra-lesion pixels", COLORS["c2"], (0.14, 0.69)),
        ((0.43, 0.50), "mask geometry", COLORS["c5d"], (0.14, 0.46)),
        ((0.43, 0.31), "edit effects", COLORS["c6d"], (0.32, 0.52)),
    ]
    for (xy, text, color, source) in source_boxes:
        add_box(ax, xy, (0.24, 0.10), text, color)
        add_arrow(ax, source, (xy[0], xy[1] + 0.05), color)
        add_arrow(ax, (xy[0] + 0.24, xy[1] + 0.05), (0.82, 0.57), color)

    ax.text(
        0.03,
        0.13,
        "High AUROC after lesion removal shows label decodability in the edited representation;\n"
        "it does not identify the source.",
        fontsize=6.4,
        color=COLORS["gray"],
        ha="left",
        va="center",
    )
    save_figure(fig, "fig1_interventions_schematic")


def primary_stats() -> dict[str, dict[str, float]]:
    primary = load_json("results/bibm_primary_endpoints.json")["settings"]["DINOv3/BUS-BRA"]
    dilated = load_json("results/bibm_dilated_mask_controls.json")["settings"]["DINOv3/BUS-BRA"]
    out: dict[str, dict[str, float]] = {}
    for condition in ("C1_full", "C2_lesion-only", "C4_lesion-removed-inpaint"):
        out[condition] = primary["conditions"][condition]
    for condition in ("C5d_dilated-mask-only", "C6d_dilated-geometry-rendering"):
        out[condition] = dilated["image_controls"][condition]
    out["dilated_mask_geometry_features"] = dilated["explicit_dilated_geometry"]
    return out


def sensitivity_stats() -> dict[str, dict[str, dict[str, float]]]:
    dilated = load_json("results/bibm_dilated_mask_controls.json")["settings"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in ("MedSAM2/BUS-BRA", "DINOv3/BUS-UCLM"):
        if setting not in dilated:
            continue
        entry = dilated[setting]
        out[setting] = {
            "C4_lesion-removed-inpaint": entry["c4_reference"],
            "C5d_dilated-mask-only": entry["image_controls"]["C5d_dilated-mask-only"],
            "C6d_dilated-geometry-rendering": entry["image_controls"]["C6d_dilated-geometry-rendering"],
            "dilated_mask_geometry_features": entry["explicit_dilated_geometry"],
        }
    return out


def draw_endpoint(ax: mpl.axes.Axes, x: float, stats: dict[str, float], condition: str, *, label: str | None = None) -> None:
    color, marker = POINT_STYLE[condition]
    val = stats["auroc"]
    ci = (stats["ci_lower"], stats["ci_upper"])
    ax.errorbar(
        [x],
        [val],
        yerr=np.array([[val - ci[0]], [ci[1] - val]]),
        fmt=marker,
        color=color,
        mfc=color,
        mec=color,
        markersize=4.8,
        elinewidth=0.8,
        capsize=2,
        label=label,
        zorder=2,
    )


def make_fig2(output_path: Path | None = None) -> None:
    use_template_figure_labels()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    conditions = [
        "C1_full",
        "C2_lesion-only",
        "C4_lesion-removed-inpaint",
        "C5d_dilated-mask-only",
        "C6d_dilated-geometry-rendering",
        "dilated_mask_geometry_features",
    ]
    stats = primary_stats()
    for idx, condition in enumerate(conditions):
        draw_endpoint(ax, idx, stats[condition], condition)
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([PRIMARY_LABELS[c] for c in conditions])
    ax.set_ylim(0.4, 1.01)
    ax.axhline(0.5, color=COLORS["light_gray"], linewidth=0.8)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.set_ylabel("Patient-level AUROC")
    ax.set_title("A. Primary intervention-aligned audit\nDINOv3 / BUS-BRA")

    ax = axes[1]
    sens = sensitivity_stats()
    setting_order = [s for s in ("MedSAM2/BUS-BRA", "DINOv3/BUS-UCLM") if s in sens]
    offset = {
        "C4_lesion-removed-inpaint": -0.18,
        "C5d_dilated-mask-only": -0.06,
        "C6d_dilated-geometry-rendering": 0.06,
        "dilated_mask_geometry_features": 0.18,
    }
    for base, setting in enumerate(setting_order):
        for condition in offset:
            draw_endpoint(
                ax,
                base + offset[condition],
                sens[setting][condition],
                condition,
                label=PRIMARY_LABELS[condition].replace("\n", " ") if base == 0 else None,
            )
    ax.set_xticks(np.arange(len(setting_order)))
    ax.set_xticklabels([s.replace("/", "\n") for s in setting_order])
    ax.set_ylim(0.4, 1.01)
    ax.axhline(0.5, color=COLORS["light_gray"], linewidth=0.8)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.set_title("B. Encoder/dataset sensitivity")
    ax.legend(frameon=False, loc="lower left", ncols=1)
    fig.suptitle("Mask-derived controls retain high label decodability; C5d exceeds C4", y=1.02, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig2_primary_endpoints", output_path)


def fixed_records() -> list[dict[str, Any]]:
    source_files = {
        "DINOv3/BUS-BRA": "results/fixed_shape_controls.json",
        "MedSAM2/BUS-BRA": "results/fixed_shape_controls_medsam2.json",
        "DINOv3/BUS-UCLM": "results/fixed_shape_controls_bus_uclm.json",
        "UltraSam/BUS-BRA": "results/fixed_shape_controls_ultrasam.json",
    }
    records: list[dict[str, Any]] = []
    for setting, file in source_files.items():
        data = load_json(file)
        for shape in ("circle", "square", "ellipse"):
            vals = data["fixed_shape_removal"][shape]["c4_minus_fixed"]
            records.append(
                {
                    "setting": setting,
                    "condition": shape.capitalize(),
                    "delta": -vals["delta_auroc"],
                    "ci": (-vals["ci_upper"], -vals["ci_lower"]),
                }
            )
    return records


def make_fig3(output_path: Path | None = None) -> None:
    use_template_figure_labels()
    recoverability = load_json("results/bibm_geometry_recoverability.json")
    fixed = fixed_records()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.15), gridspec_kw={"width_ratios": [1.12, 1.0]})

    ax = axes[0]
    conditions = [
        "C1_full",
        "C4_lesion-removed-inpaint",
        "C5d_dilated-mask-only",
        "C6d_dilated-geometry-rendering",
    ]
    condition_labels = ["Full", "C4", "C5d", "C6d"]
    targets = [
        ("sqrt_normalized_area", "area", COLORS["c5d"], "o", -0.18),
        ("normalized_perimeter", "perim.", COLORS["geom"], "s", -0.06),
        ("centroid_x", "centroid x", COLORS["c2"], "D", 0.06),
        ("centroid_y", "centroid y", COLORS["c6d"], "P", 0.18),
    ]
    x = np.arange(len(conditions))
    ax.axvspan(1.65, 3.35, color=COLORS["light_gray"], alpha=0.38, zorder=0)
    for target, label, color, marker, dx in targets:
        for c_idx, condition in enumerate(conditions):
            stats = recoverability["conditions"][condition]["targets"][target]["spearman"]
            val = float(stats["value"])
            ci = (float(stats["ci_lower"]), float(stats["ci_upper"]))
            alpha = 0.46 if condition in {"C5d_dilated-mask-only", "C6d_dilated-geometry-rendering"} else 1.0
            ax.errorbar(
                [x[c_idx] + dx],
                [val],
                yerr=np.array([[val - ci[0]], [ci[1] - val]], dtype=float),
                fmt=marker,
                color=color,
                mfc=color,
                mec=color,
                alpha=alpha,
                markersize=4.1,
                elinewidth=0.8,
                capsize=2,
                label=label if c_idx == 0 else None,
                zorder=3,
            )
    ax.axhline(0, color=COLORS["light_gray"], linewidth=0.8)
    ax.set_ylim(-0.15, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("A. Dilated-mask geometry recoverability")
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower left", ncols=2)
    ax.text(
        2.5,
        0.16,
        "positive\nprocedural\ncontrols",
        color=COLORS["gray"],
        fontsize=8,
        ha="center",
        va="center",
    )

    ax = axes[1]
    settings = ["DINOv3/BUS-BRA", "MedSAM2/BUS-BRA", "DINOv3/BUS-UCLM", "UltraSam/BUS-BRA"]
    setting_labels = ["DINOv3\nBUS-BRA", "MedSAM2\nBUS-BRA", "DINOv3\nBUS-UCLM", "UltraSam\nBUS-BRA"]
    x = np.arange(len(settings))
    shape_specs = [
        ("Circle", COLORS["fixed_circle"], "o", -0.14),
        ("Square", COLORS["fixed_square"], "s", 0.0),
        ("Ellipse", COLORS["fixed_ellipse"], "^", 0.14),
    ]
    for shape, color, marker, dx in shape_specs:
        vals = [next(r for r in fixed if r["setting"] == s and r["condition"] == shape)["delta"] for s in settings]
        cis = [next(r for r in fixed if r["setting"] == s and r["condition"] == shape)["ci"] for s in settings]
        for idx, (x_pos, val, ci) in enumerate(zip(x + dx, vals, cis)):
            edge = COLORS["ultrasam"] if settings[idx] == "UltraSam/BUS-BRA" else color
            face = "white" if settings[idx] == "UltraSam/BUS-BRA" else color
            ax.errorbar(
                [x_pos],
                [val],
                yerr=np.array([[val - ci[0]], [ci[1] - val]], dtype=float),
                fmt=marker,
                color=edge,
                mfc=face,
                mec=edge,
                markersize=4.1,
                elinewidth=0.8,
                capsize=2,
                label=shape if idx == 0 else None,
                zorder=2,
            )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels)
    ax.set_ylabel("ΔAUROC = fixed-shape − C4")
    ax.set_ylim(-0.34, 0.06)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.set_title("B. Geometry-and-coverage perturbation")
    ax.legend(frameon=False, loc="lower left", ncols=1)
    ax.text(
        0.0,
        0.045,
        "open: possible dataset-level exposure",
        fontsize=8,
        ha="left",
        va="top",
        color=COLORS["gray"],
    )
    fig.tight_layout()
    save_figure(fig, "fig3_geometry_controls_delta", output_path)


def main() -> None:
    make_fig1()
    make_fig2()
    make_fig3()
    print(f"Wrote manuscript figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
