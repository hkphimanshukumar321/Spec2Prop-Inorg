"""
Generate a comprehensive draw.io architecture diagram for Spec2Prop-Edge.

Covers:
  1. MultiScaleSpecNet (ASPP + Channel Attention CNN)
  2. RamanFormer1D v2 (CNN-Transformer with SupCon)
  3. Domain Feature Extraction (163-d)
  4. PCA Projection (32-d)
  5. Prototype Similarity Features (3×K)
  6. Feature Concatenation & StandardScaler
  7. LightGBM-DART Classifier (Optuna-tuned)
  8. Confidence-Aware Screening Report
  9. Loss Functions (SupCon + Focal)
  10. Ensemble Layer
"""
import xml.etree.ElementTree as ET
import os

# ═══════════════════════════════════════════════════════════════════════════════
# Draw.io XML builder helpers
# ═══════════════════════════════════════════════════════════════════════════════
mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "21.1.2", "type": "device"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "full_arch", "name": "Spec2Prop-Edge Full Architecture"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "2000", "dy": "2000", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "4400",
    "pageHeight": "3200", "math": "0", "shadow": "0", "background": "#FFFFFF"
})
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

_id = [2]

def nid():
    _id[0] += 1
    return str(_id[0] - 1)

def node(x, y, w, h, label, style, parent="1"):
    i = nid()
    c = ET.SubElement(root, "mxCell", {"id": i, "value": label, "style": style, "vertex": "1", "parent": parent})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return i

def edge(src, tgt, style, label=""):
    i = nid()
    attrs = {"id": i, "style": style, "edge": "1", "parent": "1", "source": src, "target": tgt}
    if label:
        attrs["value"] = label
    e = ET.SubElement(root, "mxCell", attrs)
    ET.SubElement(e, "mxGeometry", {"relative": "1", "as": "geometry"})
    return i

def group(x, y, w, h, label, style):
    i = nid()
    c = ET.SubElement(root, "mxCell", {"id": i, "value": label, "style": style, "vertex": "1", "connectable": "0", "parent": "1"})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return i

# ═══════════════════════════════════════════════════════════════════════════════
# Colour Palette (professional, journal-ready)
# ═══════════════════════════════════════════════════════════════════════════════
# Input
C_INPUT      = "#E8EAF6"  # lavender
B_INPUT      = "#283593"  # indigo
# Preprocessing
C_PREP       = "#E3F2FD"  # light blue
B_PREP       = "#1565C0"  # blue
# CNN blocks
C_CNN        = "#E0F2F1"  # teal tint
B_CNN        = "#00695C"  # teal
# Transformer blocks
C_TFR        = "#FFF3E0"  # light amber
B_TFR        = "#E65100"  # deep orange
# Feature extraction
C_FEAT       = "#E8F5E9"  # light green
B_FEAT       = "#2E7D32"  # green
# Classifier
C_CLS        = "#FCE4EC"  # light pink
B_CLS        = "#880E4F"  # dark pink
# Output
C_OUT        = "#F3E5F5"  # light purple
B_OUT        = "#6A1B9A"  # purple
# Loss / training
C_LOSS       = "#FFF9C4"  # light yellow
B_LOSS       = "#F57F17"  # amber
# Ensemble
C_ENS        = "#EFEBE9"  # light brown
B_ENS        = "#4E342E"  # brown
# Annotation / text
C_TXT        = "none"
B_TXT        = "none"

def sty(fill, stroke, font=12, bold=True, rounded=True, dashed=False):
    b = "1" if bold else "0"
    r = "1" if rounded else "0"
    d = "1" if dashed else "0"
    dash_pat = "strokeDasharray: 8 4;" if dashed else ""
    return (f"rounded={r};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize={font};fontStyle={b};"
            f"arcSize=8;{dash_pat}")

def sty_text(font=11, align="left", color="#37474F"):
    return (f"text;html=1;strokeColor=none;fillColor=none;align={align};"
            f"verticalAlign=top;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;"
            f"fontSize={font};fontColor={color};")

def sty_group(stroke, label_bg="#FFFFFF"):
    return (f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={stroke};"
            f"strokeWidth=3;dashed=1;dashPattern=8 4;arcSize=6;verticalAlign=top;"
            f"fontFamily=Helvetica;fontSize=16;fontStyle=1;fontColor={stroke};"
            f"labelBackgroundColor={label_bg};spacingTop=5;")

S_EDGE       = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#37474F;strokeWidth=2;fontFamily=Helvetica;fontSize=11;"
S_EDGE_BOLD  = S_EDGE.replace("strokeWidth=2", "strokeWidth=3").replace("strokeColor=#37474F", "strokeColor=#1565C0")
S_EDGE_DASH  = S_EDGE.replace("strokeWidth=2", "strokeWidth=2;dashed=1;dashPattern=6 3").replace("strokeColor=#37474F", "strokeColor=#9E9E9E")
S_EDGE_EMB   = S_EDGE.replace("strokeColor=#37474F", "strokeColor=#00695C")

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
node(40, 20, 4300, 50,
     '<font style="font-size:22px;"><b>Spec2Prop-Edge: Complete Model Architecture — Flagship Deep Models → Feature Fusion → LightGBM-DART Deployment Pipeline</b></font>',
     sty_text(22, "center", "#1A237E"))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT
# ═══════════════════════════════════════════════════════════════════════════════
grp_input = group(40, 90, 260, 200, "① Input", sty_group(B_INPUT))
inp = node(60, 140, 220, 60,
           '<b>Raw Raman Spectrum</b><br/><font style="font-size:10px;">CSV/TXT: Raman shift (cm⁻¹) vs Intensity<br/>Shape: variable length</font>',
           sty(C_INPUT, B_INPUT, 12))
inp_xrd = node(60, 220, 220, 50,
               '<b>Raw XRD (optional)</b><br/><font style="font-size:10px;">2θ vs Intensity</font>',
               sty(C_INPUT, B_INPUT, 11, dashed=True))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RUNTIME PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
grp_prep = group(330, 90, 300, 200, "② Runtime Preprocessing", sty_group(B_PREP))
prep = node(350, 140, 260, 130,
            '<b>Spectral Preprocessing Pipeline</b><br/>'
            '<font style="font-size:10px;">'
            '1. File parsing &amp; invalid row removal<br/>'
            '2. Axis sorting &amp; duplicate removal<br/>'
            '3. Interpolation to fixed grid (2048 pts)<br/>'
            '4. Baseline correction<br/>'
            '5. Savitzky-Golay smoothing<br/>'
            '6. Min-max normalisation → [0, 1]<br/>'
            '<b>Output: [1 × 2048] float32 vector</b>'
            '</font>',
            sty(C_PREP, B_PREP, 12))
edge(inp, prep, S_EDGE_BOLD)
edge(inp_xrd, prep, S_EDGE_DASH)

# 2048-d vector node (shared fan-out point)
vec = node(680, 140, 180, 50,
           '<b>2048-d Spectral Vector</b><br/><font style="font-size:10px;">[B, 1, 2048] tensor</font>',
           sty(C_PREP, B_PREP, 12))
edge(prep, vec, S_EDGE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3A — MultiScaleSpecNet (Left Column, rows 320–1050)
# ═══════════════════════════════════════════════════════════════════════════════
grp_ms = group(40, 320, 420, 820, "③-A  MultiScaleSpecNet  (cnn1d.py)", sty_group(B_CNN))

BW = 380  # block width
BX = 60   # block X
BY = 380  # starting Y
BH = 55   # block height
BG = 15   # gap

ms_blocks = [
    ('<b>Stem Conv1D</b><br/><font style="font-size:10px;">k=31, pad=15, stride=1 → BN → GELU → MaxPool1D(4)</font>',
     '<font style="font-size:10px;">[B, 32, 512]</font>'),
    ('<b>ASPP Block 1</b><br/><font style="font-size:10px;">4× Conv1D(k=3) dilations=[1, 6, 12, 18]<br/>Concat → 1×1 Conv proj → BN → ReLU</font>',
     '<font style="font-size:10px;">[B, 64, 512]</font>'),
    ('<b>CAS 1 — Channel Attention Squeeze</b><br/><font style="font-size:10px;">GAP → FC(64→8) → ReLU → FC(8→64) → Sigmoid<br/>Element-wise re-weighting + MaxPool1D(4)</font>',
     '<font style="font-size:10px;">[B, 64, 128]</font>'),
    ('<b>ASPP Block 2</b><br/><font style="font-size:10px;">4× Conv1D(k=3) dilations=[1, 6, 12, 18]<br/>Concat → 1×1 Conv proj → BN → ReLU</font>',
     '<font style="font-size:10px;">[B, 128, 128]</font>'),
    ('<b>CAS 2 — Channel Attention Squeeze</b><br/><font style="font-size:10px;">GAP → FC(128→16) → ReLU → FC(16→128) → Sigmoid<br/>Element-wise re-weighting + MaxPool1D(4)</font>',
     '<font style="font-size:10px;">[B, 128, 32]</font>'),
    ('<b>Global Average Pooling</b><br/><font style="font-size:10px;">AdaptiveAvgPool1d(1) → squeeze</font>',
     '<font style="font-size:10px;">[B, 128]</font>'),
    ('<b>FC Embedding</b><br/><font style="font-size:10px;">Linear(128 → 128) → ReLU → Dropout(0.3)</font>',
     '<font style="font-size:10px;">[B, 128]</font>'),
]

prev_ms = None
ms_node_ids = []
for i, (label, shape_label) in enumerate(ms_blocks):
    y = BY + i * (BH + BG)
    n = node(BX, y, BW, BH, label, sty(C_CNN, B_CNN, 11))
    # shape annotation to the right (inside the group area)
    node(BX + BW + 5, y + 10, 100, 30, shape_label, sty_text(10, "left", B_CNN))
    if prev_ms:
        edge(prev_ms, n, S_EDGE)
    ms_node_ids.append(n)
    prev_ms = n

ms_embed = ms_node_ids[-1]  # the final embedding node

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — RamanFormer1D v2 (Right Column, rows 320–1200)
# ═══════════════════════════════════════════════════════════════════════════════
RF_X = 500
grp_rf = group(RF_X, 320, 460, 920, "③-B  RamanFormer1D v2 — SupCon-Ready  (raman_former.py)", sty_group(B_TFR))

rf_blocks = [
    ('<b>Tokenizer Stage 1</b><br/><font style="font-size:10px;">Conv1D(1→32, k=7, s=2, pad=3) → BN → GELU</font>',
     '[B, 32, 1024]', C_CNN, B_CNN),
    ('<b>Tokenizer Stage 2</b><br/><font style="font-size:10px;">Conv1D(32→64, k=5, s=2, pad=2) → BN → GELU</font>',
     '[B, 64, 512]', C_CNN, B_CNN),
    ('<b>Tokenizer Stage 3 — ResConv1D</b><br/><font style="font-size:10px;">Conv1D(64→128, k=3, s=2) + residual shortcut<br/>(1×1 conv if dim mismatch)</font>',
     '[B, 128, 256]', C_CNN, B_CNN),
    ('<b>Tokenizer Stage 4 — ResConv1D</b><br/><font style="font-size:10px;">Conv1D(128→128, k=3, s=2) + residual shortcut<br/>(identity shortcut)</font>',
     '[B, 128, 128]', C_CNN, B_CNN),
    ('<b>Token Preparation</b><br/><font style="font-size:10px;">Permute(0,2,1) → Prepend learned [CLS] token<br/>+ Learned Positional Embeddings (129 × 128)</font>',
     '[B, 129, 128]', C_TFR, B_TFR),
    ('<b>Transformer Encoder × 4</b><br/><font style="font-size:10px;">Pre-LN Attention (4 heads, d_model=128)<br/>FFN (128→512→128, GELU)<br/>Stochastic Depth (linear 0→0.1)<br/>Dropout=0.2</font>',
     '[B, 129, 128]', C_TFR, B_TFR),
    ('<b>LayerNorm + [CLS] Slice</b><br/><font style="font-size:10px;">Extract x[:, 0, :] — dedicated summary token</font>',
     '[B, 128]', C_TFR, B_TFR),
]

prev_rf = None
rf_node_ids = []
for i, (label, shape_str, fill, stroke) in enumerate(rf_blocks):
    y = BY + i * (BH + BG + 10)
    n = node(RF_X + 20, y, BW + 20, BH + 10, label, sty(fill, stroke, 11))
    node(RF_X + BW + 50, y + 12, 110, 30,
         f'<font style="font-size:10px;">{shape_str}</font>', sty_text(10, "left", stroke))
    if prev_rf:
        edge(prev_rf, n, S_EDGE)
    rf_node_ids.append(n)
    prev_rf = n

rf_embed = rf_node_ids[-1]

# SupCon Projection Head (branch off RamanFormer)
supcon_y = BY + len(rf_blocks) * (BH + BG + 10)
proj_head = node(RF_X + 20, supcon_y, 200, 60,
                 '<b>Projection Head (SupCon)</b><br/><font style="font-size:10px;">Linear(128→128) → BN → GELU<br/>Linear(128→128) → L2-Norm</font>',
                 sty(C_LOSS, B_LOSS, 11))
edge(rf_embed, proj_head, S_EDGE_DASH, "Stage 1 only")

proj_out = node(RF_X + 20, supcon_y + 75, 200, 40,
                '<font style="font-size:10px;">[B, 128] on unit hypersphere</font>',
                sty_text(10, "center", B_LOSS))

# CE Head (branch off RamanFormer)
ce_head = node(RF_X + 240, supcon_y, 200, 60,
               '<b>Classification Head (CE)</b><br/><font style="font-size:10px;">Linear(128→256) → BN → GELU<br/>Dropout(0.2) → Linear(256→K)</font>',
               sty(C_LOSS, B_LOSS, 11))
edge(rf_embed, ce_head, S_EDGE_DASH, "Stage 2 fine-tune")

ce_out = node(RF_X + 240, supcon_y + 75, 200, 40,
              '<font style="font-size:10px;">[B, num_classes] logits</font>',
              sty_text(10, "center", B_LOSS))

# Connect 2048-d vector to both deep models
edge(vec, ms_node_ids[0], S_EDGE_BOLD)
edge(vec, rf_node_ids[0], S_EDGE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS box
# ═══════════════════════════════════════════════════════════════════════════════
loss_y = supcon_y + 130
grp_loss = group(RF_X, loss_y, 460, 120, "Training Losses  (losses.py)", sty_group(B_LOSS))
node(RF_X + 20, loss_y + 35, 200, 70,
     '<b>SupConLoss</b><br/><font style="font-size:10px;">Khosla et al. NeurIPS 2020<br/>τ = 0.07, contrast_mode = all<br/>Attracts same-class, repels others<br/>on L2-normed projections</font>',
     sty(C_LOSS, B_LOSS, 10))
node(RF_X + 240, loss_y + 35, 200, 70,
     '<b>FocalLoss</b><br/><font style="font-size:10px;">Lin et al. 2017<br/>γ = 2.0, label_smoothing = 0.1<br/>Handles class imbalance<br/>Used for Stage 2 CE fine-tuning</font>',
     sty(C_LOSS, B_LOSS, 10))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DOMAIN FEATURE EXTRACTION (below models, wide block)
# ═══════════════════════════════════════════════════════════════════════════════
FEAT_Y = 1280
grp_feat = group(40, FEAT_Y, 920, 360, "④ Feature Extraction for Deployment  (domain_features.py, prototype_features.py)", sty_group(B_FEAT))

# Connect 2048-d vector down to feature extraction
feat_anchor = node(700, FEAT_Y - 40, 160, 35,
                   '<b>2048-d Raw Spectrum</b>',
                   sty(C_PREP, B_PREP, 11))
edge(vec, feat_anchor, S_EDGE_BOLD)

# Branch A — Domain features
dom = node(60, FEAT_Y + 50, 280, 170,
           '<b>Hand-Crafted Domain Features</b><br/>'
           '<font style="font-size:10px;">'
           '① <b>Peak features</b> (30-d):<br/>'
           '   top-10 positions, heights, widths<br/>'
           '② <b>Spectral moments</b> (4-d):<br/>'
           '   mean, std, skewness, kurtosis<br/>'
           '③ <b>Band means + ratios</b> (28-d):<br/>'
           '   7 diagnostic windows + ⁷C₂ = 21 ratios<br/>'
           '④ <b>Subsampled spectrum</b> (64-d):<br/>'
           '   stride-32 avg-pool from 2048→64<br/>'
           '⑤ <b>Derivative stats</b> (12-d):<br/>'
           '   1st/2nd deriv: mean, std, max, min, ZCR, energy<br/>'
           '⑥ <b>Spectral entropy</b> (1-d)<br/>'
           '⑦ <b>Band peak counts + areas</b> (14-d)<br/>'
           '⑧ <b>Peak prominences</b> (10-d)<br/>'
           '<b>Total: 163 features</b>'
           '</font>',
           sty(C_FEAT, B_FEAT, 11))
edge(feat_anchor, dom, S_EDGE)

# Branch B — PCA
pca_node = node(370, FEAT_Y + 50, 260, 80,
                '<b>PCA Global Shape</b><br/>'
                '<font style="font-size:10px;">'
                'sklearn PCA (whiten=True, seed=42)<br/>'
                'Fit on X_train only<br/>'
                'n_components = 32<br/>'
                '<b>Output: 32-d</b>'
                '</font>',
                sty(C_FEAT, B_FEAT, 11))
edge(feat_anchor, pca_node, S_EDGE)

# Branch C — Prototype Similarity
proto_node = node(370, FEAT_Y + 155, 260, 100,
                  '<b>Prototype Similarity</b><br/>'
                  '<font style="font-size:10px;">'
                  'Fit class prototypes on train only<br/>'
                  'Per sample × per class (K=9):<br/>'
                  '  • cosine similarity<br/>'
                  '  • Pearson correlation<br/>'
                  '  • Euclidean distance<br/>'
                  '<b>Output: 3 × 9 = 27-d</b>'
                  '</font>',
                  sty(C_FEAT, B_FEAT, 11))
edge(feat_anchor, proto_node, S_EDGE)

# Branch D — CNN embeddings (optional, from HeterogeneousFeatureExtractor)
cnn_emb_node = node(660, FEAT_Y + 50, 280, 80,
                    '<b>CNN Embedding (optional)</b><br/>'
                    '<font style="font-size:10px;">'
                    'From trained MultiScaleSpecNet or<br/>'
                    'RamanFormer1D .forward_features()<br/>'
                    '<b>Output: 128-d</b><br/>'
                    '(used in hybrid_models.py path)'
                    '</font>',
                    sty(C_FEAT, B_FEAT, 11, dashed=True))
edge(ms_embed, cnn_emb_node, S_EDGE_DASH)
edge(rf_embed, cnn_emb_node, S_EDGE_DASH)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONCATENATION + SCALING
# ═══════════════════════════════════════════════════════════════════════════════
CONCAT_Y = FEAT_Y + 390
concat_node = node(120, CONCAT_Y, 360, 60,
                   '<b>Feature Concatenation</b><br/>'
                   '<font style="font-size:10px;">'
                   'np.hstack([domain_163, pca_32, prototype_27])<br/>'
                   '<b>= 222-d heterogeneous feature vector</b>'
                   '</font>',
                   sty(C_FEAT, B_FEAT, 12))
edge(dom, concat_node, S_EDGE)
edge(pca_node, concat_node, S_EDGE)
edge(proto_node, concat_node, S_EDGE)

# optional CNN embedding concat
concat_ext = node(540, CONCAT_Y, 340, 60,
                  '<b>Extended Concat (hybrid path)</b><br/>'
                  '<font style="font-size:10px;">'
                  'domain_163 + pca_32 + prototype_27 + cnn_128<br/>'
                  '<b>= ~350-d vector</b>'
                  '</font>',
                  sty(C_FEAT, B_FEAT, 11, dashed=True))
edge(cnn_emb_node, concat_ext, S_EDGE_DASH)
edge(concat_node, concat_ext, S_EDGE_DASH)

# StandardScaler
scaler_node = node(200, CONCAT_Y + 85, 200, 45,
                   '<b>StandardScaler</b><br/><font style="font-size:10px;">Fit on train, transform val/test</font>',
                   sty(C_FEAT, B_FEAT, 11))
edge(concat_node, scaler_node, S_EDGE)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LightGBM-DART CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
CLS_Y = CONCAT_Y + 160
grp_cls = group(40, CLS_Y, 560, 220, "⑤ LightGBM-DART Classifier  (train_deployment_model.py)", sty_group(B_CLS))

lgbm_node = node(60, CLS_Y + 45, 300, 160,
                 '<b>LightGBM-DART</b><br/>'
                 '<font style="font-size:10px;">'
                 '<b>Optuna HPO (50 trials, TPE sampler)</b><br/>'
                 '• n_estimators: 300–2000<br/>'
                 '• max_depth: 3–10<br/>'
                 '• learning_rate: 0.01–0.3 (log)<br/>'
                 '• subsample: 0.5–1.0<br/>'
                 '• colsample_bytree: 0.3–1.0<br/>'
                 '• num_leaves: 15–127<br/>'
                 '• reg_alpha / reg_lambda: 1e-3–10 (log)<br/>'
                 '• drop_rate (DART): 0.05–0.3<br/>'
                 '• class_weight: balanced<br/>'
                 '• objective: multiclass (K=9)'
                 '</font>',
                 sty(C_CLS, B_CLS, 11))
edge(scaler_node, lgbm_node, S_EDGE_BOLD)

# Comparison classifiers
comp_node = node(390, CLS_Y + 45, 190, 100,
                 '<b>Comparison Models</b><br/>'
                 '<font style="font-size:10px;">'
                 '• CatBoost (depth=6, lr=0.05,<br/>'
                 '  auto_class_weights, 1000 iter)<br/>'
                 '• TabPFN (zero-shot,<br/>'
                 '  CPU inference)'
                 '</font>',
                 sty(C_CLS, B_CLS, 10, dashed=True))
edge(scaler_node, comp_node, S_EDGE_DASH)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
ENS_Y = CLS_Y + 240
ens_node = node(40, ENS_Y, 280, 80,
                '<b>Probability Ensemble</b><br/>'
                '<font style="font-size:10px;">'
                'Weighted probability averaging<br/>'
                'or Stacked Generalization<br/>'
                '(LogisticRegression meta-learner)<br/>'
                '<b>stacking_ensemble.py</b>'
                '</font>',
                sty(C_ENS, B_ENS, 11))
edge(lgbm_node, ens_node, S_EDGE)
edge(comp_node, ens_node, S_EDGE_DASH)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT / SCREENING REPORT
# ═══════════════════════════════════════════════════════════════════════════════
OUT_Y = ENS_Y + 110
grp_out = group(40, OUT_Y, 560, 200, "⑥ Confidence-Aware Screening Report", sty_group(B_OUT))

out_node = node(60, OUT_Y + 40, 300, 140,
                '<b>Deployment Output</b><br/>'
                '<font style="font-size:10px;">'
                '1. <b>Predicted chemical family</b><br/>'
                '2. <b>Top-3 candidate families</b><br/>'
                '3. <b>Confidence score</b> (max probability)<br/>'
                '4. <b>Prediction quality</b>: High / Med / Low<br/>'
                '5. <b>Suggested next action</b><br/>'
                '6. <b>Screening-level disclaimer</b><br/>'
                '</font>',
                sty(C_OUT, B_OUT, 11))
edge(ens_node, out_node, S_EDGE_BOLD)

class_tags = node(390, OUT_Y + 40, 190, 140,
                  '<b>9 Material Families</b><br/>'
                  '<font style="font-size:10px;">'
                  '• Silicate<br/>'
                  '• Oxide<br/>'
                  '• Carbonate<br/>'
                  '• Sulfate<br/>'
                  '• Phosphate<br/>'
                  '• Sulfide<br/>'
                  '• Halide<br/>'
                  '• Borate<br/>'
                  '• Other/Rare'
                  '</font>',
                  sty(C_OUT, B_OUT, 10))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DEPLOYMENT ARTIFACTS (saved .joblib files)
# ═══════════════════════════════════════════════════════════════════════════════
ART_Y = OUT_Y + 220
node(40, ART_Y, 560, 80,
     '<b>Saved Deployment Artifacts</b><br/>'
     '<font style="font-size:10px;">'
     '• <b>pca.joblib</b> — fitted PCA(32) on training set<br/>'
     '• <b>scaler.joblib</b> — fitted StandardScaler on training features<br/>'
     '• <b>prototype_extractor.joblib</b> — fitted class prototypes (train-only)<br/>'
     '• <b>lgbm_deployment.joblib</b> — Optuna-tuned LightGBM-DART model<br/>'
     '• <b>encoder.json</b> — label → integer mapping (9 families)'
     '</font>',
     sty("#FAFAFA", "#9E9E9E", 11, bold=False))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — HeterogeneousFeatureExtractor box (hybrid_models.py)
# ═══════════════════════════════════════════════════════════════════════════════
HFE_X = 1020
HFE_Y = 320
grp_hfe = group(HFE_X, HFE_Y, 360, 260,
                "HeterogeneousFeatureExtractor  (hybrid_models.py)", sty_group("#546E7A"))
node(HFE_X + 20, HFE_Y + 40, 320, 200,
     '<b>Hybrid Feature Pipeline</b><br/>'
     '<font style="font-size:10px;">'
     'Combines 3 heterogeneous feature sources:<br/><br/>'
     '① <b>CNN Embedding</b> (128-d)<br/>'
     '   • forward_features() from any trained encoder<br/>'
     '   • Batch inference (bs=64) with torch.no_grad()<br/><br/>'
     '② <b>PCA Components</b> (32-d)<br/>'
     '   • sklearn PCA.transform() on raw 2048-d<br/><br/>'
     '③ <b>Domain Descriptors</b> (126-d→163-d)<br/>'
     '   • extract_domain_features(n_peaks=10)<br/><br/>'
     '<b>Total: ~286–350-d vector</b><br/>'
     '→ fed to GBDT/TabPFN classifier'
     '</font>',
     sty("#ECEFF1", "#546E7A", 11))

edge(ms_embed, str(int(grp_hfe)+1), S_EDGE_EMB)
edge(rf_embed, str(int(grp_hfe)+1), S_EDGE_EMB)

# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Band Windows legend
# ═══════════════════════════════════════════════════════════════════════════════
LEGEND_X = 1020
LEGEND_Y = 620
node(LEGEND_X, LEGEND_Y, 360, 180,
     '<b>Diagnostic Raman Band Windows</b><br/>'
     '<font style="font-size:10px;">'
     '<b>Window</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Index range</b><br/>'
     'silicate_main (Si-O bend) &nbsp;&nbsp; 200–280<br/>'
     'silicate_stretch (Si-O-Si) &nbsp;&nbsp; 830–900<br/>'
     'carbonate (CO₃ bend) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 560–620<br/>'
     'phosphate (P-O stretch) &nbsp;&nbsp;&nbsp; 530–590<br/>'
     'oxide_low (M-O lattice) &nbsp;&nbsp;&nbsp; 100–180<br/>'
     'ch_stretch (C-H organics) &nbsp; 1780–1950<br/>'
     'oh_stretch (O-H hydrated) &nbsp; 1950–2048<br/><br/>'
     '<i>7 band means + ⁷C₂ = 21 pairwise ratios = 28 features</i>'
     '</font>',
     sty("#FAFAFA", "#78909C", 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════════
# SimpleCNN1D / LiteSpecNet / ResidualCNN1D baselines
# ═══════════════════════════════════════════════════════════════════════════════
BASE_X = 1020
BASE_Y = 840
node(BASE_X, BASE_Y, 360, 210,
     '<b>Baseline CNN Encoders  (cnn1d.py)</b><br/>'
     '<font style="font-size:10px;">'
     '<b>SimpleCNN1D</b><br/>'
     '  Conv1D(1→32,k=7) → BN → ReLU → Pool(4)<br/>'
     '  Conv1D(32→64,k=5) → BN → ReLU → Pool(4)<br/>'
     '  Conv1D(64→128,k=3) → BN → ReLU → Pool(4)<br/>'
     '  GAP → FC(128→128). embed_dim=128<br/><br/>'
     '<b>LiteSpecNet</b> (~15K params, edge-friendly)<br/>'
     '  Conv1D(1→16,k=11) → DepthwiseSep(16→32,k=7)<br/>'
     '  → DepthwiseSep(32→48,k=5) → GAP → FC(48→64)<br/><br/>'
     '<b>ResidualCNN1D</b><br/>'
     '  Conv→Pool→ResBlock(32)→Pool<br/>'
     '  Conv→Pool→ResBlock(64)→Pool<br/>'
     '  Conv→Pool→GAP→FC(128→128)'
     '</font>',
     sty("#FAFAFA", "#78909C", 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════════
# RamanPCAMLP
# ═══════════════════════════════════════════════════════════════════════════════
node(BASE_X, BASE_Y + 230, 360, 120,
     '<b>RamanPCAMLP  (cnn1d.py)</b><br/>'
     '<font style="font-size:10px;">'
     'Frozen PCA projection: Linear(2048→pca_dim)<br/>'
     'MLP: BN → Linear(pca_dim→512) → BN → ReLU → Drop<br/>'
     '     Linear(512→256) → BN → ReLU → Drop<br/>'
     '     Linear(256→128) → BN → ReLU → Drop<br/>'
     'Head: Linear(128→K)'
     '</font>',
     sty("#FAFAFA", "#78909C", 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════════
# Fusion / Multimodal side panel
# ═══════════════════════════════════════════════════════════════════════════════
node(BASE_X, BASE_Y + 370, 360, 130,
     '<b>Fusion &amp; Multimodal Models</b><br/>'
     '<font style="font-size:10px;">'
     '<b>FusionSpec2PropNet</b> (fusion_models.py)<br/>'
     '  Raman CNN + Descriptor MLP → Concat → Fusion FC<br/>'
     '  → Multi-task heads (classification + regression)<br/><br/>'
     '<b>DualBranchRamanXRDNet</b> (multimodal_cnn.py)<br/>'
     '  Raman CNN + XRD CNN + Optional Descriptor MLP<br/>'
     '  → Concat → Fusion FC → FusionMultiTaskHead'
     '</font>',
     sty("#FAFAFA", "#78909C", 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════════
# CAPTION
# ═══════════════════════════════════════════════════════════════════════════════
node(40, ART_Y + 110, 1340, 60,
     '<font style="font-size:11px;"><b>Figure:</b> Complete Spec2Prop-Edge architecture. '
     'Two flagship deep models (MultiScaleSpecNet with ASPP+CAS, RamanFormer1D v2 with SupCon) produce 128-d embeddings. '
     'For deployment, raw 2048-point spectra are transformed into a 222-d heterogeneous vector (163 domain + 32 PCA + 27 prototype features), '
     'scaled, and classified by an Optuna-tuned LightGBM-DART. '
     'The system outputs top-k inorganic family predictions with confidence scores for scientist verification.</font>',
     sty_text(11, "left", "#37474F"))

# ═══════════════════════════════════════════════════════════════════════════════
# Write XML
# ═══════════════════════════════════════════════════════════════════════════════
if hasattr(ET, "indent"):
    ET.indent(ET.ElementTree(mxfile), space="  ", level=0)
out_path = os.path.join(os.path.dirname(__file__), "Spec2Prop_Full_Architecture.drawio")
ET.ElementTree(mxfile).write(out_path, encoding="utf-8", xml_declaration=True)
print(f"✅ Generated: {out_path}")
print(f"   Open this file in https://app.diagrams.net/ or the Draw.io VSCode extension.")
