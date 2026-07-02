"""
Generate IEEE Transactions-quality figures for Spec2Prop-InorgBench.
All figures are vector PDF, 300 DPI, serif fonts, muted colorblind-safe palette.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Global IEEE style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.grid': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,  # TrueType for editable text
})

OUTDIR = 'figures'

# Muted colorblind-safe palette (Tol Bright variant)
PAL = {
    'blue':    '#4477AA',
    'cyan':    '#66CCEE',
    'green':   '#228833',
    'yellow':  '#CCBB44',
    'red':     '#EE6677',
    'purple':  '#AA3377',
    'grey':    '#BBBBBB',
    'orange':  '#EE8866',
    'teal':    '#44AA99',
}
FAMILY_COLORS = [PAL['blue'], PAL['orange'], PAL['green'], PAL['grey'],
                 PAL['red'], PAL['purple'], PAL['teal'], PAL['yellow'],
                 '#555555']

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Dataset construction funnel (waterfall / stepped bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_dataset_funnel():
    stages = [
        'Raw RRUFF records',
        'Usable Raman\nspectra',
        'Clean inorganic\nsamples',
        'Property-matched\nsamples',
        'Raman–XRD linked\nsamples',
    ]
    counts = [5200, 4118, 4027, 1844, 1462]
    operations = [
        '',
        'Parse & interpolate\nto 2048 pts',
        'Remove organic,\nmixed, unknown',
        'Matbench formula\nmatching',
        'XRD pairing by\nRRUFF-ID',
    ]
    colors = [PAL['grey'], PAL['blue'], PAL['green'], PAL['orange'], PAL['red']]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.barh(range(len(stages)), counts, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.65, zorder=3)

    for i, (bar, c, op) in enumerate(zip(bars, counts, operations)):
        ax.text(c + 60, i, f'N = {c:,}', va='center', ha='left',
                fontsize=7, fontweight='bold', color='#333333')
        if i > 0:
            ax.annotate('', xy=(counts[i], i - 0.35), xytext=(counts[i-1], i - 0.65),
                        arrowprops=dict(arrowstyle='->', color='#888888',
                                        lw=0.7, connectionstyle='arc3,rad=-0.15'))
            ax.text(max(counts[i], counts[i-1]) + 60, i - 0.5, op,
                    fontsize=5.5, va='center', ha='left', color='#666666',
                    fontstyle='italic')

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Number of samples', fontsize=8)
    ax.set_xlim(0, max(counts) * 1.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.savefig(f'{OUTDIR}/fig2_dataset_funnel.pdf')
    fig.savefig(f'{OUTDIR}/fig2_dataset_funnel.png')
    plt.close(fig)
    print('  ✓ fig2_dataset_funnel')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5a — Family distribution (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig5a_family_distribution():
    families = ['Silicate', 'Oxide', 'Carbonate', 'Phosphate',
                'Halide', 'Sulfate', 'Borate', 'Other/Rare', 'Sulfide']
    counts = np.array([1400, 626, 513, 446, 320, 266, 260, 133, 60])
    # Sort descending
    idx = np.argsort(counts)[::-1]
    families = [families[i] for i in idx]
    counts = counts[idx]
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y_pos = np.arange(len(families))
    bars = ax.barh(y_pos, counts, color=FAMILY_COLORS[:len(families)],
                   edgecolor='white', linewidth=0.4, height=0.7, zorder=3)

    for i, (c, p) in enumerate(zip(counts, pcts)):
        ax.text(c + 15, i, f'{c:,}  ({p:.1f}%)', va='center', ha='left',
                fontsize=6, color='#333333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Sample count', fontsize=8)
    ax.set_xlim(0, max(counts) * 1.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.savefig(f'{OUTDIR}/fig5a_family_distribution.pdf')
    fig.savefig(f'{OUTDIR}/fig5a_family_distribution.png')
    plt.close(fig)
    print('  ✓ fig5a_family_distribution')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5b — Confusion matrix (9×9 heatmap)
# ══════════════════════════════════════════════════════════════════════════════
def fig5b_confusion_matrix():
    labels = ['BOR', 'CAR', 'HAL', 'OTH', 'OXI', 'PHO', 'SIL', 'SUL', 'SUD']
    full_labels = ['Borate', 'Carbonate', 'Halide', 'Other/\nRare', 'Oxide',
                   'Phosphate', 'Silicate', 'Sulfate', 'Sulfide']
    # Normalized confusion matrix data from the paper's fig6
    cm = np.array([
        [0.65, 0.00, 0.05, 0.05, 0.00, 0.05, 0.00, 0.10, 0.10],
        [0.08, 0.69, 0.08, 0.03, 0.00, 0.05, 0.00, 0.08, 0.00],
        [0.15, 0.03, 0.28, 0.25, 0.03, 0.03, 0.00, 0.07, 0.17],
        [0.22, 0.00, 0.11, 0.67, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.19, 0.06, 0.48, 0.06, 0.02, 0.05, 0.00, 0.06, 0.06],
        [0.07, 0.03, 0.09, 0.06, 0.01, 0.70, 0.00, 0.03, 0.00],
        [0.37, 0.07, 0.19, 0.07, 0.01, 0.22, 0.00, 0.06, 0.01],
        [0.05, 0.01, 0.22, 0.05, 0.01, 0.34, 0.00, 0.30, 0.01],
        [0.10, 0.00, 0.06, 0.33, 0.02, 0.00, 0.00, 0.00, 0.48],
    ])

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    cmap = LinearSegmentedColormap.from_list('ieee_blues',
               ['#F7FBFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B'])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1,
                   aspect='equal')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = 'white' if val > 0.5 else '#222222'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5.5, color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(len(full_labels)))
    ax.set_xticklabels(full_labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(len(full_labels)))
    ax.set_yticklabels(full_labels, fontsize=6)
    ax.set_xlabel('Predicted family', fontsize=8)
    ax.set_ylabel('True family', fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Normalized frequency', fontsize=7)

    fig.savefig(f'{OUTDIR}/fig5b_confusion_matrix.pdf')
    fig.savefig(f'{OUTDIR}/fig5b_confusion_matrix.png')
    plt.close(fig)
    print('  ✓ fig5b_confusion_matrix')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6a — Per-class F1 (from fig7_feature_importance_shap / fig_family_model_comparison)
# ══════════════════════════════════════════════════════════════════════════════
def fig6a_perclass_f1():
    families = ['Borate', 'Silicate', 'Other/Rare', 'Halide', 'Oxide',
                'Carbonate', 'Phosphate', 'Sulfate', 'Sulfide']
    f1s = [73.6, 72.7, 70.3, 65.8, 64.4, 58.1, 48.5, 42.4, 37.0]
    macro_f1 = 59.2

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y_pos = np.arange(len(families))
    bars = ax.barh(y_pos, f1s, color=FAMILY_COLORS[:len(families)],
                   edgecolor='white', linewidth=0.4, height=0.7, zorder=3)

    for i, v in enumerate(f1s):
        ax.text(v + 0.8, i, f'{v:.1f}%', va='center', ha='left',
                fontsize=6, color='#333333')

    ax.axvline(x=macro_f1, color=PAL['red'], linestyle='--', linewidth=0.8,
               zorder=4, alpha=0.8)
    ax.text(macro_f1 + 0.5, len(families) - 0.5, f'Macro-F1\n{macro_f1:.1f}%',
            fontsize=6, color=PAL['red'], va='top', ha='left')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('F1-score (%)', fontsize=8)
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.savefig(f'{OUTDIR}/fig6a_perclass_f1.pdf')
    fig.savefig(f'{OUTDIR}/fig6a_perclass_f1.png')
    plt.close(fig)
    print('  ✓ fig6a_perclass_f1')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6b — Top-k accuracy (line + bar hybrid)
# ══════════════════════════════════════════════════════════════════════════════
def fig6b_topk_accuracy():
    ks = [1, 2, 3, 5]
    accs = [66.1, 81.1, 87.4, 95.5]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bar_colors = [PAL['blue'], PAL['cyan'], PAL['green'], PAL['teal']]
    bars = ax.bar(range(len(ks)), accs, color=bar_colors, edgecolor='white',
                  linewidth=0.5, width=0.6, zorder=3)

    # Connecting line
    ax.plot(range(len(ks)), accs, color='#333333', linewidth=1.0,
            marker='o', markersize=4, markerfacecolor='white',
            markeredgecolor='#333333', markeredgewidth=0.8, zorder=4)

    for i, (k, a) in enumerate(zip(ks, accs)):
        ax.text(i, a + 1.2, f'{a:.1f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='#333333')

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=7)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_ylim(55, 102)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Fill area under curve
    ax.fill_between(range(len(ks)), accs, alpha=0.08, color=PAL['blue'], zorder=2)

    fig.savefig(f'{OUTDIR}/fig6b_topk_accuracy.pdf')
    fig.savefig(f'{OUTDIR}/fig6b_topk_accuracy.png')
    plt.close(fig)
    print('  ✓ fig6b_topk_accuracy')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7a — Property prediction grouped bars
# ══════════════════════════════════════════════════════════════════════════════
def fig7a_property_prediction():
    tasks = ['Band-gap\nclass', 'Metal/\nNon-metal', 'Formation\nenergy']
    models = ['SimpleCNN1D', 'DescriptorMLP', 'FusionSpec2PropNet', 'Spec2PropLite']

    acc_data = np.array([
        [27.8, 70.0, 37.5],   # SimpleCNN1D
        [65.3, 80.5, 84.0],   # DescriptorMLP
        [66.1, 82.0, 86.3],   # FusionSpec2PropNet
        [61.7, 90.2, 80.9],   # Spec2PropLite
    ])
    f1_data = np.array([
        [27.4, 55.6, 26.2],
        [61.0, 68.0, 76.6],
        [63.2, 68.0, 80.7],
        [57.1, 77.1, 74.9],
    ])

    model_colors = [PAL['grey'], PAL['blue'], PAL['red'], PAL['orange']]
    x = np.arange(len(tasks))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=False)

    for i, (model, color) in enumerate(zip(models, model_colors)):
        bars1 = ax1.bar(x + offsets[i], acc_data[i], width * 0.9, color=color,
                        edgecolor='white', linewidth=0.3, label=model, zorder=3)
        bars2 = ax2.bar(x + offsets[i], f1_data[i], width * 0.9, color=color,
                        edgecolor='white', linewidth=0.3, label=model, zorder=3)
        for j in range(len(tasks)):
            ax1.text(x[j] + offsets[i], acc_data[i, j] + 1,
                     f'{acc_data[i, j]:.0f}', ha='center', va='bottom',
                     fontsize=4.5, color='#444444', rotation=90)
            ax2.text(x[j] + offsets[i], f1_data[i, j] + 1,
                     f'{f1_data[i, j]:.0f}', ha='center', va='bottom',
                     fontsize=4.5, color='#444444', rotation=90)

    for ax, ylabel, title in [(ax1, 'Accuracy (%)', '(a) Test accuracy'),
                               (ax2, 'Macro-F1 (%)', '(b) Test macro-F1')]:
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax2.legend(loc='upper left', fontsize=6, framealpha=0.9,
               edgecolor='#cccccc', ncol=2, columnspacing=0.8)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUTDIR}/fig7a_property_prediction.pdf')
    fig.savefig(f'{OUTDIR}/fig7a_property_prediction.png')
    plt.close(fig)
    print('  ✓ fig7a_property_prediction')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7b — Multimodal comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig7b_multimodal():
    metrics = ['Accuracy', 'Macro-F1']
    raman_only = [39.3, 28.3]
    dual_branch = [44.8, 36.6]
    gains = ['+5.5 pp', '+8.3 pp']

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars1 = ax.bar(x - width/2, raman_only, width, color=PAL['blue'],
                   edgecolor='white', linewidth=0.5, label='SimpleCNN1D (Raman only)',
                   zorder=3)
    bars2 = ax.bar(x + width/2, dual_branch, width, color=PAL['green'],
                   edgecolor='white', linewidth=0.5, label='DualBranch (Raman + XRD)',
                   zorder=3)

    for i in range(len(metrics)):
        ax.text(x[i] - width/2, raman_only[i] + 0.8, f'{raman_only[i]:.1f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.text(x[i] + width/2, dual_branch[i] + 0.8, f'{dual_branch[i]:.1f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold')
        # Gain annotation
        mid_y = max(raman_only[i], dual_branch[i]) + 5
        ax.annotate(gains[i], xy=(x[i], mid_y), fontsize=7,
                    ha='center', va='bottom', color=PAL['red'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel('Score (%)', fontsize=8)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Subtitle
    ax.set_title('Multimodal family classification (N = 1,462)', fontsize=9)

    fig.savefig(f'{OUTDIR}/fig7b_multimodal.pdf')
    fig.savefig(f'{OUTDIR}/fig7b_multimodal.png')
    plt.close(fig)
    print('  ✓ fig7b_multimodal')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Confidence calibration dual-panel
# ══════════════════════════════════════════════════════════════════════════════
def fig8_confidence_calibration():
    # (a) Reliability diagram
    pred_conf = np.array([0.27, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.75, 0.85, 0.95])
    obs_acc   = np.array([0.31, 0.32, 0.33, 0.39, 0.50, 0.52, 0.53, 0.63, 0.81, 0.94])

    # (b) Reject-option analysis
    coverage = np.array([100, 79.6, 70.0, 56.0, 43.1, 35.0])
    rej_acc  = np.array([66.1, 73.8, 78.0, 83.1, 89.2, 92.0])
    rej_f1   = np.array([59.2, 65.5, 71.0, 74.7, 81.5, 85.0])
    thresholds = ['none', '0.5', '0.6', '0.7', '0.8', '0.85']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # Panel (a) — Reliability diagram
    ax1.plot([0, 1], [0, 1], '--', color='#999999', linewidth=0.7, zorder=2)
    ax1.plot(pred_conf, obs_acc, '-o', color=PAL['blue'], linewidth=1.2,
             markersize=5, markerfacecolor='white', markeredgecolor=PAL['blue'],
             markeredgewidth=1.0, zorder=4)
    ax1.fill_between(pred_conf, pred_conf, obs_acc, alpha=0.12,
                     color=PAL['blue'], zorder=1)
    ax1.set_xlabel('Mean predicted confidence', fontsize=8)
    ax1.set_ylabel('Observed accuracy', fontsize=8)
    ax1.set_title('(a) Reliability diagram', fontsize=9, fontweight='bold')
    ax1.set_xlim(0.2, 1.0)
    ax1.set_ylim(0.2, 1.0)
    ax1.set_aspect('equal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b) — Reject-option
    ax2.plot(coverage, rej_acc, '-o', color=PAL['blue'], linewidth=1.2,
             markersize=5, markerfacecolor='white', markeredgecolor=PAL['blue'],
             markeredgewidth=1.0, zorder=4, label='Accuracy')
    ax2.plot(coverage, rej_f1, '--s', color=PAL['green'], linewidth=1.2,
             markersize=5, markerfacecolor='white', markeredgecolor=PAL['green'],
             markeredgewidth=1.0, zorder=4, label='Macro-F1')

    for i, t in enumerate(thresholds):
        if t != 'none' and i < len(coverage):
            ax2.annotate(f'τ={t}', (coverage[i], rej_acc[i]),
                         textcoords='offset points', xytext=(5, 6),
                         fontsize=5.5, color=PAL['blue'], fontstyle='italic')

    ax2.set_xlabel('Coverage (%)', fontsize=8)
    ax2.set_ylabel('Performance (%)', fontsize=8)
    ax2.set_title('(b) Reject-option analysis', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower left', framealpha=0.9, edgecolor='#cccccc')
    ax2.set_xlim(30, 105)
    ax2.set_ylim(50, 95)
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(f'{OUTDIR}/fig8_confidence_calibration.pdf')
    fig.savefig(f'{OUTDIR}/fig8_confidence_calibration.png')
    plt.close(fig)
    print('  ✓ fig8_confidence_calibration')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Radar chart (deployment model multi-metric profile)
# ══════════════════════════════════════════════════════════════════════════════
def fig9_radar_chart():
    metrics = ['Top-1 Acc', 'Macro-F1', 'Weighted-F1', 'Balanced\nAcc',
               'Top-3 Acc', 'Top-5 Acc']
    # Deployment model (222-d PCA+Domain+Proto)
    deploy_vals = [66.06, 59.21, 65.80, 58.74, 87.42, 95.53]
    # Ablation model (158-d PCA+Domain)
    ablation_vals = [67.22, 56.57, 63.0, 55.0, 82.0, 90.0]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    deploy_vals_c = deploy_vals + deploy_vals[:1]
    ablation_vals_c = ablation_vals + ablation_vals[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.2), subplot_kw=dict(polar=True))

    # Grid styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    # Draw grid circles
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=5, color='#888888')
    ax.yaxis.grid(True, color='#dddddd', linewidth=0.4)
    ax.xaxis.grid(True, color='#dddddd', linewidth=0.4)

    # Plot
    ax.plot(angles, deploy_vals_c, 'o-', linewidth=1.3, color=PAL['blue'],
            markersize=4, markerfacecolor='white', markeredgecolor=PAL['blue'],
            markeredgewidth=0.9, label='Deployment (222-d)', zorder=4)
    ax.fill(angles, deploy_vals_c, alpha=0.15, color=PAL['blue'])

    ax.plot(angles, ablation_vals_c, 's--', linewidth=1.0, color=PAL['red'],
            markersize=3.5, markerfacecolor='white', markeredgecolor=PAL['red'],
            markeredgewidth=0.8, label='Ablation (158-d)', zorder=3)
    ax.fill(angles, ablation_vals_c, alpha=0.08, color=PAL['red'])

    # Add value labels
    for i, (a, dv, av) in enumerate(zip(angles[:-1], deploy_vals, ablation_vals)):
        ax.text(a, dv + 6, f'{dv:.1f}', fontsize=5.5, ha='center', va='center',
                color=PAL['blue'], fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=7)

    ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05),
              fontsize=6, framealpha=0.9, edgecolor='#cccccc')

    fig.savefig(f'{OUTDIR}/fig9_radar_chart.pdf')
    fig.savefig(f'{OUTDIR}/fig9_radar_chart.png')
    plt.close(fig)
    print('  ✓ fig9_radar_chart')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Model comparison (horizontal grouped bar — all models Acc + F1)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_model_comparison():
    models = [
        'RamanFormer1D', 'SVM', 'kNN', 'LogisticReg',
        'Random Forest', 'XGBoost',
        'LightGBM-DART\n(PCA+Domain)',
        'LightGBM-DART\n(Deployment)'
    ]
    accs = [25.2, 48.2, 51.8, 55.5, 62.9, 64.1, 67.2, 66.1]
    f1s  = [27.3, 31.0, 41.5, 45.0, 49.4, 52.5, 56.6, 59.2]

    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars1 = ax.barh(y + height/2, accs, height, color=PAL['blue'],
                    edgecolor='white', linewidth=0.3, label='Accuracy', zorder=3)
    bars2 = ax.barh(y - height/2, f1s, height, color=PAL['green'],
                    edgecolor='white', linewidth=0.3, label='Macro-F1', zorder=3)

    for i in range(len(models)):
        ax.text(accs[i] + 0.5, y[i] + height/2, f'{accs[i]:.1f}',
                va='center', ha='left', fontsize=5.5, color=PAL['blue'])
        ax.text(f1s[i] + 0.5, y[i] - height/2, f'{f1s[i]:.1f}',
                va='center', ha='left', fontsize=5.5, color=PAL['green'])

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=6.5)
    ax.set_xlabel('Score (%)', fontsize=8)
    ax.set_xlim(0, 82)
    ax.invert_yaxis()
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.savefig(f'{OUTDIR}/fig10_model_comparison.pdf')
    fig.savefig(f'{OUTDIR}/fig10_model_comparison.png')
    plt.close(fig)
    print('  ✓ fig10_model_comparison')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating IEEE-quality figures...')
    fig2_dataset_funnel()
    fig5a_family_distribution()
    fig5b_confusion_matrix()
    fig6a_perclass_f1()
    fig6b_topk_accuracy()
    fig7a_property_prediction()
    fig7b_multimodal()
    fig8_confidence_calibration()
    fig9_radar_chart()
    fig10_model_comparison()
    print('All figures generated.')
