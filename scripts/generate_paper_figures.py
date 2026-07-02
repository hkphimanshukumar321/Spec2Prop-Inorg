#!/usr/bin/env python3
"""
Generate all data-driven IEEE-style figures for the Spec2Prop-InorgBench paper.

Figures generated:
  fig2  – Dataset construction funnel
  fig5  – Chemical family distribution bar chart
  fig6  – Normalized confusion matrix (deployment model)
  fig7  – Feature importance / per-class F1 radar + reject-option curve
  fig8  – Property prediction grouped bar chart
  fig10 – Multimodal improvement comparison (bonus)

IEEE/journal style: serif fonts, tight layout, 300 dpi, PDF + PNG output.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS  = os.path.join(ROOT, 'results')
NUMBERS  = os.path.join(RESULTS, 'numbers.json')
OUT_DIR  = os.path.join(ROOT, 'Spec2Prop_ChemistryAfrica', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(NUMBERS, 'r') as f:
    NUM = json.load(f)

# Load deployment eval for confusion matrix / calibration / reject-option
DEPLOY_EVAL = os.path.join(RESULTS, 'hybrid', 'deployment', 'deployment_full_eval.json')
with open(DEPLOY_EVAL, 'r') as f:
    DEPLOY = json.load(f)

# Load confusion matrices
FAMILY_CM_FILE = os.path.join(RESULTS, 'family_classification', 'confusion_matrices.json')
with open(FAMILY_CM_FILE, 'r') as f:
    FAMILY_CM_DATA = json.load(f)

PROP_CM_FILE = os.path.join(RESULTS, 'property_prediction', 'confusion_matrices.json')
with open(PROP_CM_FILE, 'r') as f:
    PROP_CM_DATA = json.load(f)

# ── IEEE Style Setup ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# IEEE column width ~3.5 in, double col ~7.16 in
COL_W = 3.5
DBL_W = 7.16

# Color palette – professional muted
COLORS = {
    'primary':   '#2C3E50',
    'accent1':   '#2980B9',
    'accent2':   '#E74C3C',
    'accent3':   '#27AE60',
    'accent4':   '#F39C12',
    'accent5':   '#8E44AD',
    'accent6':   '#16A085',
    'light_bg':  '#ECF0F1',
    'grid':      '#BDC3C7',
}

# Categorical palette for 9 families
FAMILY_COLORS = [
    '#3498DB',  # Borate
    '#E67E22',  # Carbonate
    '#2ECC71',  # Halide
    '#95A5A6',  # Other/Rare
    '#E74C3C',  # Oxide
    '#9B59B6',  # Phosphate
    '#1ABC9C',  # Silicate
    '#F1C40F',  # Sulfate
    '#34495E',  # Sulfide
]

FAMILY_NAMES = ['Borate', 'Carbonate', 'Halide', 'Other/Rare', 'Oxide',
                'Phosphate', 'Silicate', 'Sulfate', 'Sulfide']


def save_fig(fig, name):
    """Save as both PDF and PNG."""
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, format=ext)
        print(f'  Saved: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Dataset Construction Funnel
# ══════════════════════════════════════════════════════════════════════════
def fig2_dataset_funnel():
    print('\n[Fig 2] Dataset Construction Funnel')
    fig, ax = plt.subplots(figsize=(COL_W, 3.8))

    stages = [
        ('Raw RRUFF + MLROD\nrecords', None, '#3498DB'),
        ('Usable Raman\nspectra', 4118, '#2980B9'),
        ('Clean inorganic\nsamples', 4027, '#27AE60'),
        ('Property-matched\nsamples', 1844, '#E67E22'),
        ('Raman–XRD linked\nsamples', 1462, '#E74C3C'),
    ]

    n = len(stages)
    y_positions = np.linspace(0.88, 0.08, n)
    max_width = 0.80
    min_width = 0.30
    widths = np.linspace(max_width, min_width, n)

    for i, (label, count, color) in enumerate(stages):
        y = y_positions[i]
        w = widths[i]
        x_left = 0.5 - w / 2
        h = 0.12

        box = FancyBboxPatch((x_left, y), w, h, boxstyle="round,pad=0.015",
                             facecolor=color, edgecolor='white', linewidth=1.5,
                             alpha=0.9, transform=ax.transAxes)
        ax.add_patch(box)

        # Text label
        txt = label if count is None else f'{label}\n(N = {count:,})'
        ax.text(0.5, y + h/2, txt, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=0.5, foreground='black')])

        # Downward arrow
        if i < n - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + h + 0.005),
                        xytext=(0.5, y - 0.005),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))

        # Filter labels on the right
        filters = [
            None,
            'Parse & interpolate\nto 2048 pts',
            'Remove organic,\nmixed, unknown',
            'Matbench formula\nmatching',
            'XRD pairing by\nRRUFF-ID',
        ]
        if filters[i]:
            ax.text(0.5 + w/2 + 0.05, y + h/2, filters[i],
                    ha='left', va='center', fontsize=6.5, color='#7F8C8D',
                    fontstyle='italic', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Dataset Construction Funnel', fontsize=10, fontweight='bold', pad=5)
    save_fig(fig, 'fig2_dataset_funnel')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Chemical Family Distribution Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def fig5_family_distribution():
    print('\n[Fig 5] Chemical Family Distribution')
    # Use deployment per-class support as the authoritative counts
    support = DEPLOY['per_class_report']
    counts = {}
    for cls_key in [f'class_{i}' for i in range(9)]:
        if cls_key in support:
            counts[cls_key] = int(support[cls_key]['support'])

    # Total test samples = 604, but we need the full dataset counts.
    # Test set is 15% of 4027 → ~604. Scale up to approximate full distribution.
    scale_factor = 4027 / 604
    full_counts = {FAMILY_NAMES[i]: int(counts[f'class_{i}'] * scale_factor)
                   for i in range(9)}

    # Sort by count descending
    sorted_families = sorted(full_counts.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_families]
    vals = [s[1] for s in sorted_families]
    colors = [FAMILY_COLORS[FAMILY_NAMES.index(n)] for n in names]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Approximate sample count', fontsize=9)
    ax.set_title('Chemical Family Distribution\n(Spec2Prop-InorgBench, N ≈ 4,027)', fontsize=9, fontweight='bold')

    # Add count labels on bars
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                f'{v}', va='center', ha='left', fontsize=7, color=COLORS['primary'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig(fig, 'fig5_family_distribution')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Normalized Confusion Matrix (Deployment LightGBM-DART)
# ══════════════════════════════════════════════════════════════════════════
def fig6_confusion_matrix():
    print('\n[Fig 6] Normalized Confusion Matrix')

    # Build confusion matrix from the per-class precision/recall/f1/support
    # We'll use the RamanFormer1D from family_classification as the main deep learning result
    # AND the deployment model.  The deployment model doesn't have a CM stored,
    # so let's use the RamanFormer1D family classification CM as the primary figure.
    cm_raw = np.array(FAMILY_CM_DATA['RamanFormer1D']['chemical_family_model'])

    # Normalize by row (true class)
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(DBL_W * 0.65, DBL_W * 0.55))

    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized frequency', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    n_classes = len(FAMILY_NAMES)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    short_names = ['BOR', 'CAR', 'HAL', 'OTH', 'OXI', 'PHO', 'SIL', 'SUL', 'SUD']
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(FAMILY_NAMES, fontsize=7)
    ax.set_xlabel('Predicted family', fontsize=9)
    ax.set_ylabel('True family', fontsize=9)
    ax.set_title('Normalized Confusion Matrix – RamanFormer1D\n(9-class Chemical Family Classification)',
                 fontsize=9, fontweight='bold')

    # Annotate cells
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            raw = cm_raw[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}\n({raw})', ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold' if i == j else 'normal')

    plt.tight_layout()
    save_fig(fig, 'fig6_confusion_matrix')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Two-panel — Per-class F1 + Reject-Option Curve
# ══════════════════════════════════════════════════════════════════════════
def fig7_feature_importance_reject():
    print('\n[Fig 7] Per-class Performance & Reject-Option Analysis')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 3.0))

    # --- Panel (a): Per-class F1 of deployment model ---
    per_class = DEPLOY['per_class_report']
    f1_vals = []
    class_names_ordered = []
    for i in range(9):
        key = f'class_{i}'
        if key in per_class:
            f1_vals.append(per_class[key]['f1-score'])
            class_names_ordered.append(FAMILY_NAMES[i])

    sorted_idx = np.argsort(f1_vals)[::-1]
    f1_sorted = [f1_vals[i] for i in sorted_idx]
    names_sorted = [class_names_ordered[i] for i in sorted_idx]
    colors_sorted = [FAMILY_COLORS[FAMILY_NAMES.index(n)] for n in names_sorted]

    bars = ax1.barh(range(len(f1_sorted)), [v*100 for v in f1_sorted],
                    color=colors_sorted, edgecolor='white', linewidth=0.5, height=0.7)
    ax1.set_yticks(range(len(names_sorted)))
    ax1.set_yticklabels(names_sorted, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel('F1-score (%)', fontsize=8)
    ax1.set_title('(a) Per-class F1 — Deployment Model', fontsize=9, fontweight='bold')
    ax1.set_xlim(0, 100)

    # Add value labels
    for bar, v in zip(bars, f1_sorted):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{v*100:.1f}%', va='center', fontsize=6.5, color=COLORS['primary'])

    # Macro-F1 line
    macro_f1 = DEPLOY['macro_f1'] * 100
    ax1.axvline(macro_f1, color=COLORS['accent2'], linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(macro_f1 + 1, len(f1_sorted) - 0.5, f'Macro-F1\n{macro_f1:.1f}%',
             fontsize=6.5, color=COLORS['accent2'], va='bottom')

    # --- Panel (b): Reject-option curve ---
    reject = DEPLOY['reject_option']
    # Add the no-threshold baseline
    thresholds = [0.0] + [r['threshold'] for r in reject]
    coverages  = [1.0] + [r['coverage'] for r in reject]
    accuracies = [DEPLOY['accuracy']] + [r['accuracy'] for r in reject]
    macro_f1s  = [DEPLOY['macro_f1']] + [r['macro_f1'] for r in reject]

    ax2.plot([c*100 for c in coverages], [a*100 for a in accuracies],
             'o-', color=COLORS['accent1'], label='Accuracy', markersize=5, linewidth=1.5)
    ax2.plot([c*100 for c in coverages], [f*100 for f in macro_f1s],
             's--', color=COLORS['accent3'], label='Macro-F1', markersize=5, linewidth=1.5)

    # Annotate thresholds
    for i, th in enumerate(thresholds):
        if th > 0:
            ax2.annotate(f'τ={th}', (coverages[i]*100, accuracies[i]*100),
                         textcoords="offset points", xytext=(5, 5), fontsize=6,
                         color=COLORS['accent1'])

    ax2.set_xlabel('Coverage (%)', fontsize=8)
    ax2.set_ylabel('Performance (%)', fontsize=8)
    ax2.set_title('(b) Reject-Option Analysis', fontsize=9, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.9)
    ax2.set_xlim(35, 105)
    ax2.set_ylim(50, 95)
    ax2.grid(True, alpha=0.3, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig7_feature_importance_shap')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Property Prediction Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def fig8_property_prediction():
    print('\n[Fig 8] Property Prediction Results')

    models = ['SimpleCNN1D', 'DescriptorMLP', 'FusionSpec2PropNet', 'Spec2PropLite']
    model_labels = ['SimpleCNN1D\n(Raman only)', 'DescriptorMLP\n(Descriptors)', 
                    'FusionSpec2PropNet\n(Raman+Desc)', 'Spec2PropLite\n(Lite+Desc)']
    tasks = ['band_gap_class', 'is_metal', 'formation_energy_class']
    task_labels = ['Band-gap class', 'Metal / Non-metal', 'Formation energy']
    task_colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]

    prop_test = NUM['results']['task2_property_prediction']['models']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DBL_W, 5.5), gridspec_kw={'height_ratios': [1, 1]})

    # --- Panel (a): Accuracy ---
    x = np.arange(len(models))
    width = 0.22
    offsets = [-width, 0, width]

    for k, (task, tlabel, color) in enumerate(zip(tasks, task_labels, task_colors)):
        accs = [prop_test[m]['test'][task]['accuracy'] * 100 for m in models]
        bars = ax1.bar(x + offsets[k], accs, width, label=tlabel, color=color,
                       edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=7)
    ax1.set_ylabel('Accuracy (%)', fontsize=8)
    ax1.set_title('(a) Test Accuracy by Task and Model', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7, ncol=3, loc='upper left', frameon=True, framealpha=0.9)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.4)

    # --- Panel (b): Macro-F1 ---
    for k, (task, tlabel, color) in enumerate(zip(tasks, task_labels, task_colors)):
        f1s = [prop_test[m]['test'][task]['macro_f1'] * 100 for m in models]
        bars = ax2.bar(x + offsets[k], f1s, width, label=tlabel, color=color,
                       edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=7)
    ax2.set_ylabel('Macro-F1 (%)', fontsize=8)
    ax2.set_title('(b) Test Macro-F1 by Task and Model', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7, ncol=3, loc='upper left', frameon=True, framealpha=0.9)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.4)

    plt.tight_layout()
    save_fig(fig, 'fig8_property_prediction')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10 (Bonus): Multimodal Raman+XRD Improvement
# ══════════════════════════════════════════════════════════════════════════
def fig10_multimodal_comparison():
    print('\n[Fig 10] Multimodal Raman+XRD Comparison')
    mm = NUM['results']['task3_multimodal_raman_xrd']['models']

    metrics = ['test_accuracy', 'test_macro_f1']
    metric_labels = ['Accuracy', 'Macro-F1']
    models_mm = ['SimpleCNN1D_raman_only_ablation', 'DualBranchRamanXRDNet']
    model_labels_mm = ['SimpleCNN1D\n(Raman only)', 'DualBranch\n(Raman+XRD)']
    model_colors = [COLORS['accent1'], COLORS['accent3']]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(metrics))
    width = 0.3

    for k, (model, mlabel, color) in enumerate(zip(models_mm, model_labels_mm, model_colors)):
        vals = [mm[model][m] * 100 for m in metrics]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=mlabel,
                      color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylabel('Score (%)', fontsize=8)
    ax.set_title('Multimodal Raman+XRD vs Raman-Only\n(Chemical Family Classification, N = 1,462)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.9)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3, linewidth=0.4)

    # Improvement annotation
    imp = NUM['results']['task3_multimodal_raman_xrd']['multimodal_improvement']
    ax.annotate(f'+{imp["accuracy_gain_relative_pct"]:.1f}% rel. gain',
                xy=(0 + 0.5*width, mm['DualBranchRamanXRDNet']['test_accuracy']*100),
                xytext=(0.6, 55), fontsize=7, color=COLORS['accent2'],
                arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=1))

    plt.tight_layout()
    save_fig(fig, 'fig10_multimodal_comparison')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: Family Classification Model Comparison (Baselines + DL + Hybrid)
# ══════════════════════════════════════════════════════════════════════════
def fig_family_model_comparison():
    print('\n[Fig Extra] Family Classification Model Comparison')

    # Collect all family classification results
    baselines = NUM['results']['task4_traditional_baselines_family_classification']['models']
    hybrid = NUM['results']['task5_hybrid_pipeline']

    models_data = []
    # Best traditional baselines
    for name, data in baselines.items():
        models_data.append((name, data['best_accuracy']*100, data['best_macro_f1']*100))

    # Deep learning
    dl = NUM['results']['task1_family_classification_9class']['models']
    for name, data in dl.items():
        models_data.append((name, data['test_accuracy']*100, data['test_macro_f1']*100))

    # Hybrid models
    stage2 = hybrid['stage2_hybrid_nn_plus_gbdt']
    models_data.append(('LightGBM-DART\n(PCA+Domain)', 
                         stage2['LightGBM_DART_PCA_Domain_only']['test_accuracy']*100,
                         stage2['LightGBM_DART_PCA_Domain_only']['test_macro_f1']*100))

    # Deployment model
    dep = hybrid['deployment_model']
    models_data.append(('Deployment\n(PCA+Dom+Proto)',
                         dep['test_accuracy']*100, dep['test_macro_f1']*100))

    # Sort by Macro-F1
    models_data.sort(key=lambda x: x[2])

    names = [m[0] for m in models_data]
    accs = [m[1] for m in models_data]
    f1s = [m[2] for m in models_data]

    fig, ax = plt.subplots(figsize=(DBL_W, 3.5))
    y = np.arange(len(names))
    height = 0.35

    bars1 = ax.barh(y - height/2, accs, height, label='Accuracy',
                     color=COLORS['accent1'], alpha=0.85, edgecolor='white')
    bars2 = ax.barh(y + height/2, f1s, height, label='Macro-F1',
                     color=COLORS['accent3'], alpha=0.85, edgecolor='white')

    for bar, v in zip(bars1, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=6, color=COLORS['accent1'])
    for bar, v in zip(bars2, f1s):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=6, color=COLORS['accent3'])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Score (%)', fontsize=8)
    ax.set_title('Family Classification: All Models Compared\n(9-class, Clean Inorganic Subset)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 80)
    ax.grid(axis='x', alpha=0.3, linewidth=0.4)

    plt.tight_layout()
    save_fig(fig, 'fig_family_model_comparison')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: Top-k Accuracy Curve (Deployment)
# ══════════════════════════════════════════════════════════════════════════
def fig_topk_accuracy():
    print('\n[Fig Extra] Top-k Accuracy Curve')
    topk = DEPLOY
    ks = [1, 2, 3, 5]
    vals = [topk['top_1_accuracy'], topk['top_2_accuracy'],
            topk['top_3_accuracy'], topk['top_5_accuracy']]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.plot(ks, [v*100 for v in vals], 'o-', color=COLORS['accent1'],
            markersize=8, linewidth=2, markerfacecolor=COLORS['accent2'],
            markeredgecolor='white', markeredgewidth=1.5)

    for k, v in zip(ks, vals):
        ax.text(k, v*100 + 1.5, f'{v*100:.1f}%', ha='center', fontsize=8,
                fontweight='bold', color=COLORS['primary'])

    ax.set_xticks(ks)
    ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_title('Top-k Accuracy — Deployment Model\n(LightGBM-DART, 9-class)',
                 fontsize=9, fontweight='bold')
    ax.set_ylim(55, 100)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight top-3
    ax.axhspan(85, 100, alpha=0.08, color=COLORS['accent3'])
    ax.text(4.3, 92, 'Top-3:\n87.4%', fontsize=7, color=COLORS['accent3'],
            fontweight='bold', va='center')

    plt.tight_layout()
    save_fig(fig, 'fig_topk_accuracy')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: Calibration Plot
# ══════════════════════════════════════════════════════════════════════════
def fig_calibration():
    print('\n[Fig Extra] Calibration Plot')
    cal = DEPLOY['calibration']
    confidences = [c['confidence'] for c in cal]
    accuracies  = [c['accuracy'] for c in cal]
    counts      = [c['count'] for c in cal]

    fig, ax = plt.subplots(figsize=(COL_W, 3.0))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect calibration')

    # Calibration curve
    ax.plot(confidences, accuracies, 'o-', color=COLORS['accent1'],
            markersize=6, linewidth=1.5, label='Deployment model')

    # Size of dots proportional to counts
    sizes = [c / max(counts) * 200 + 30 for c in counts]
    ax.scatter(confidences, accuracies, s=sizes, color=COLORS['accent1'],
               alpha=0.3, zorder=1)

    ax.set_xlabel('Mean predicted confidence', fontsize=8)
    ax.set_ylabel('Observed accuracy', fontsize=8)
    ax.set_title('Reliability Diagram — Deployment Model',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', frameon=True)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.4)

    plt.tight_layout()
    save_fig(fig, 'fig_calibration')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: Property Prediction Confusion Matrices (Best Model per task)
# ══════════════════════════════════════════════════════════════════════════
def fig_property_confusion_matrices():
    print('\n[Fig Extra] Property Prediction Confusion Matrices')

    tasks_info = [
        ('FusionSpec2PropNet', 'band_gap_class',
         ['Metal-like', 'Narrow\ngap', 'Semi-\nconductor', 'Wide-gap\ninsulator'],
         'Band-gap Classification'),
        ('Spec2PropLite', 'is_metal',
         ['Non-metal', 'Metal'],
         'Metallicity Classification'),
        ('FusionSpec2PropNet', 'formation_energy_class',
         ['Highly\nstable', 'Marginally\nstable', 'Stable', 'Unstable'],
         'Formation Energy Classification'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.8))

    for idx, (model, task, labels, title) in enumerate(tasks_info):
        ax = axes[idx]
        cm_raw = np.array(PROP_CM_DATA[model][task])
        cm_norm = cm_raw.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        n = len(labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha='right')
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_title(f'{title}\n({model})', fontsize=7, fontweight='bold')

        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

        if idx == 0:
            ax.set_ylabel('True label', fontsize=7)

    fig.suptitle('Property Prediction — Normalized Confusion Matrices (Best Models)',
                 fontsize=9, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_property_confusion_matrices')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Runtime Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════
def fig3_preprocessing_pipeline():
    print('\n[Fig 3] Runtime Preprocessing Pipeline')
    fig, ax = plt.subplots(figsize=(DBL_W, 1.8))

    steps = [
        'Raw file\nparsing',
        'Invalid row\nremoval',
        'Sort &\ndeduplicate',
        'Interpolate\n(2048 pts)',
        'Baseline\ncorrection',
        'Savitzky–Golay\nsmoothing',
        'Max\nnormalization',
        '2048-d\nvector',
    ]

    n = len(steps)
    step_colors = [
        '#3498DB', '#3498DB', '#3498DB',
        '#E67E22', '#E67E22', '#E67E22',
        '#27AE60', '#2C3E50'
    ]

    x_positions = np.linspace(0.06, 0.94, n)
    box_w = 0.09
    box_h = 0.55

    for i, (step, color) in enumerate(zip(steps, step_colors)):
        x = x_positions[i]
        box = FancyBboxPatch((x - box_w/2, 0.22), box_w, box_h,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='white', linewidth=1,
                             alpha=0.9, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, 0.5, step, ha='center', va='center', fontsize=6,
                fontweight='bold', color='white', transform=ax.transAxes)

        if i < n - 1:
            ax.annotate('', xy=(x_positions[i+1] - box_w/2 - 0.005, 0.5),
                        xytext=(x + box_w/2 + 0.005, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Runtime Raman/XRD Spectral Preprocessing Pipeline', fontsize=9, fontweight='bold', pad=8)
    save_fig(fig, 'fig3_runtime_preprocessing')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Feature Extraction + Model Architecture
# ══════════════════════════════════════════════════════════════════════════
def fig4_feature_architecture():
    print('\n[Fig 4] Feature Extraction & Model Architecture')
    fig, ax = plt.subplots(figsize=(DBL_W, 2.5))

    # Define block positions and sizes
    blocks = [
        # (x, y, w, h, label, color)
        (0.02, 0.30, 0.12, 0.40, '2048-pt\nRaman\nvector', '#3498DB'),
        (0.18, 0.55, 0.12, 0.30, 'PCA\n(32-d)', '#9B59B6'),
        (0.18, 0.15, 0.12, 0.30, 'Expert\nDescriptors\n(126-d)', '#E67E22'),
        (0.36, 0.30, 0.12, 0.40, 'Concatenate\n158-d\nvector', '#2ECC71'),
        (0.54, 0.30, 0.14, 0.40, 'LightGBM\n-DART\nClassifier', '#E74C3C'),
        (0.74, 0.55, 0.12, 0.25, 'Top-k\nfamily\nprediction', '#2C3E50'),
        (0.74, 0.20, 0.12, 0.25, 'Confidence\nscore', '#1ABC9C'),
        (0.90, 0.30, 0.08, 0.40, 'Screen\nOutput', '#F39C12'),
    ]

    for (x, y, w, h, label, color) in blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                             facecolor=color, edgecolor='white', linewidth=1.2,
                             alpha=0.9, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white',
                transform=ax.transAxes)

    # Arrows
    arrows = [
        (0.14, 0.70, 0.18, 0.70),  # Raman → PCA
        (0.14, 0.30, 0.18, 0.30),  # Raman → Expert
        (0.30, 0.70, 0.36, 0.55),  # PCA → Concat
        (0.30, 0.30, 0.36, 0.45),  # Expert → Concat
        (0.48, 0.50, 0.54, 0.50),  # Concat → LightGBM
        (0.68, 0.60, 0.74, 0.67),  # LightGBM → Top-k
        (0.68, 0.40, 0.74, 0.33),  # LightGBM → Confidence
        (0.86, 0.67, 0.90, 0.55),  # Top-k → Output
        (0.86, 0.33, 0.90, 0.45),  # Confidence → Output
    ]

    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Chemistry-Aware Feature Extraction & LightGBM-DART Architecture',
                 fontsize=9, fontweight='bold', pad=8)
    save_fig(fig, 'fig4_feature_model_architecture')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('Spec2Prop-InorgBench: Generating IEEE-style Paper Figures')
    print('='*60)
    print(f'Output directory: {OUT_DIR}')

    fig2_dataset_funnel()
    fig3_preprocessing_pipeline()
    fig4_feature_architecture()
    fig5_family_distribution()
    fig6_confusion_matrix()
    fig7_feature_importance_reject()
    fig8_property_prediction()
    fig10_multimodal_comparison()
    fig_family_model_comparison()
    fig_topk_accuracy()
    fig_calibration()
    fig_property_confusion_matrices()

    print('\n' + '='*60)
    print('All figures generated successfully!')
    print(f'Output: {OUT_DIR}')
    print('='*60)
