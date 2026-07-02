"""
Diagram 2: Core Model Architecture
====================================
MultiScaleSpecNet + RamanFormer1D + Training Losses
+ Feature Extraction + Concatenation + LightGBM-DART
+ Ensemble + Confidence-Aware Output

No baselines. No side panels. Clean SOTA figure.
"""
import xml.etree.ElementTree as ET
import os

mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "21.1.2", "type": "device"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "d2", "name": "Core Model Architecture"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "2200", "dy": "2200", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "3600",
    "pageHeight": "3400", "math": "0", "shadow": "0", "background": "#FFFFFF"
})
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

_id = [2]
def nid():
    _id[0] += 1; return str(_id[0] - 1)

def node(x, y, w, h, label, style, parent="1"):
    i = nid()
    c = ET.SubElement(root, "mxCell", {"id": i, "value": label, "style": style, "vertex": "1", "parent": parent})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return i

def edge(src, tgt, style, label=""):
    i = nid()
    attrs = {"id": i, "style": style, "edge": "1", "parent": "1", "source": src, "target": tgt}
    if label: attrs["value"] = label
    e = ET.SubElement(root, "mxCell", attrs)
    ET.SubElement(e, "mxGeometry", {"relative": "1", "as": "geometry"})
    return i

def grp(x, y, w, h, label, stroke):
    i = nid()
    st = (f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={stroke};"
          f"strokeWidth=2.5;dashed=0;arcSize=6;verticalAlign=top;"
          f"fontFamily=Helvetica;fontSize=14;fontStyle=1;fontColor={stroke};"
          f"labelBackgroundColor=#FFFFFF;spacingTop=4;container=0;")
    c = ET.SubElement(root, "mxCell", {"id": i, "value": label, "style": st, "vertex": "1", "parent": "1"})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return i

# ── Colours ──────────────────────────────────────────────────────────────
C_IN  = "#E3F2FD"; B_IN  = "#1565C0"
C_CNN = "#E0F2F1"; B_CNN = "#00695C"
C_TFR = "#FFF3E0"; B_TFR = "#E65100"
C_LSS = "#FFF9C4"; B_LSS = "#F57F17"
C_FEA = "#E8F5E9"; B_FEA = "#2E7D32"
C_CLS = "#FCE4EC"; B_CLS = "#880E4F"
C_ENS = "#EFEBE9"; B_ENS = "#4E342E"
C_OUT = "#F3E5F5"; B_OUT = "#6A1B9A"
C_ANN = "#FAFAFA"; B_ANN = "#78909C"

def sty(fill, stroke, fs=12, bold=True, dashed=False):
    b = "1" if bold else "0"
    dp = "strokeDasharray=8 4;" if dashed else ""
    return (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize={fs};fontStyle={b};"
            f"arcSize=10;{dp}")

def stxt(fs=11, al="left", col="#37474F"):
    return (f"text;html=1;strokeColor=none;fillColor=none;align={al};"
            f"verticalAlign=top;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;"
            f"fontSize={fs};fontColor={col};")

SE      = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#37474F;strokeWidth=2;fontFamily=Helvetica;fontSize=11;"
SE_BOLD = SE.replace("strokeWidth=2", "strokeWidth=3").replace("#37474F", "#1565C0")
SE_DASH = SE.replace("strokeWidth=2", "strokeWidth=2;dashed=1;dashPattern=6 3").replace("#37474F", "#9E9E9E")
SE_EMB  = SE.replace("#37474F", "#00695C").replace("strokeWidth=2", "strokeWidth=2.5")
SE_TFR  = SE.replace("#37474F", "#E65100").replace("strokeWidth=2", "strokeWidth=2.5")

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
node(40, 15, 3500, 40,
     '<font style="font-size:18px;"><b>Diagram 2 &mdash; Spec2Prop-Edge: Core Model Architecture (MultiScaleSpecNet + RamanFormer1D &rarr; Feature Fusion &rarr; LightGBM-DART)</b></font>',
     stxt(18, "center", "#1A237E"))

# ═══════════════════════════════════════════════════════════════════════════
# INPUT (from Diagram 1)
# ═══════════════════════════════════════════════════════════════════════════
vec_in = node(40, 75, 240, 50,
    '<b>From Diagram 1</b><br/><font style="font-size:11px;">2048-d Spectral Vector [B, 1, 2048]</font>',
    sty(C_IN, B_IN, 12))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: MultiScaleSpecNet (LEFT COLUMN)
# ═══════════════════════════════════════════════════════════════════════════
MS_X = 40; MS_Y = 160
grp(MS_X, MS_Y, 480, 680, "A. MultiScaleSpecNet (cnn1d.py) &mdash; ASPP + Channel Attention CNN", B_CNN)

ms_data = [
    ('<b>Stem: Conv1D</b><br/><font style="font-size:10px;">'
     'in=1, out=32, <b>kernel=31</b>, padding=15<br/>'
     'BatchNorm1d(32) &rarr; GELU &rarr; MaxPool1d(4)</font>',
     '[B, 32, 512]', 95),
    ('<b>ASPP Block 1 &mdash; Atrous Spatial Pyramid Pooling</b><br/><font style="font-size:10px;">'
     '4&times; parallel Conv1d(in=32, out=64, k=3)<br/>'
     'Dilations: <b>d=1, 6, 12, 18</b><br/>'
     'Concat(4&times;64=256) &rarr; Conv1d(256&rarr;64, k=1) &rarr; BN &rarr; ReLU</font>',
     '[B, 64, 512]', 105),
    ('<b>CAS 1 &mdash; Channel Attention Squeeze</b><br/><font style="font-size:10px;">'
     'AdaptiveAvgPool1d(1) &rarr; FC(64&rarr;8) &rarr; ReLU<br/>'
     '&rarr; FC(8&rarr;64) &rarr; Sigmoid &rarr; element-wise scale<br/>'
     '+ MaxPool1d(4) downsampling</font>',
     '[B, 64, 128]', 95),
    ('<b>ASPP Block 2</b><br/><font style="font-size:10px;">'
     '4&times; parallel Conv1d(in=64, out=128, k=3)<br/>'
     'Dilations: <b>d=1, 6, 12, 18</b><br/>'
     'Concat(4&times;128=512) &rarr; Conv1d(512&rarr;128, k=1) &rarr; BN &rarr; ReLU</font>',
     '[B, 128, 128]', 105),
    ('<b>CAS 2 &mdash; Channel Attention Squeeze</b><br/><font style="font-size:10px;">'
     'AdaptiveAvgPool1d(1) &rarr; FC(128&rarr;16) &rarr; ReLU<br/>'
     '&rarr; FC(16&rarr;128) &rarr; Sigmoid &rarr; element-wise scale<br/>'
     '+ MaxPool1d(4) downsampling</font>',
     '[B, 128, 32]', 95),
    ('<b>Global Average Pooling &rarr; FC Embed</b><br/><font style="font-size:10px;">'
     'AdaptiveAvgPool1d(1) &rarr; squeeze &rarr; Linear(128&rarr;128)<br/>'
     'ReLU &rarr; Dropout(0.3)</font>',
     '<b>[B, 128]</b>', 70),
]

prev_ms = None; ms_ids = []
cy = MS_Y + 45
for label, shape, h in ms_data:
    n = node(MS_X + 20, cy, 360, h, label, sty(C_CNN, B_CNN, 11))
    node(MS_X + 390, cy + 10, 80, 30,
         f'<font style="font-size:10px;color:{B_CNN};">{shape}</font>',
         stxt(10, "left", B_CNN))
    if prev_ms: edge(prev_ms, n, SE_EMB)
    ms_ids.append(n); prev_ms = n; cy += h + 12

ms_embed_id = ms_ids[-1]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: RamanFormer1D v2 (RIGHT COLUMN)
# ═══════════════════════════════════════════════════════════════════════════
RF_X = 560; RF_Y = 160
grp(RF_X, RF_Y, 520, 680, "B. RamanFormer1D v2 (raman_former.py) &mdash; SupCon-Ready CNN-Transformer", B_TFR)

rf_data = [
    ('<b>Tokenizer Stage 1: Conv1D</b><br/><font style="font-size:10px;">'
     'in=1, out=32, <b>k=7, stride=2</b>, pad=3<br/>'
     'BatchNorm1d(32) &rarr; GELU</font>',
     '[B, 32, 1024]', C_CNN, B_CNN, 70),
    ('<b>Tokenizer Stage 2: Conv1D</b><br/><font style="font-size:10px;">'
     'in=32, out=64, <b>k=5, stride=2</b>, pad=2<br/>'
     'BatchNorm1d(64) &rarr; GELU</font>',
     '[B, 64, 512]', C_CNN, B_CNN, 70),
    ('<b>Tokenizer Stage 3: ResConv1D</b><br/><font style="font-size:10px;">'
     'in=64, out=128, <b>k=3, stride=2</b><br/>'
     'Conv &rarr; BN &rarr; GELU + 1&times;1 shortcut (dim mismatch)</font>',
     '[B, 128, 256]', C_CNN, B_CNN, 75),
    ('<b>Tokenizer Stage 4: ResConv1D</b><br/><font style="font-size:10px;">'
     'in=128, out=128, <b>k=3, stride=2</b><br/>'
     'Conv &rarr; BN &rarr; GELU + identity shortcut</font>',
     '[B, 128, 128]', C_CNN, B_CNN, 75),
    ('<b>Token Preparation</b><br/><font style="font-size:10px;">'
     'Permute(0, 2, 1): [B, 128, 128] &rarr; [B, 128, 128]<br/>'
     'Prepend learned <b>[CLS] token</b> (trunc_normal, std=0.02)<br/>'
     '+ <b>Learned Positional Embeddings</b> (129 &times; 128)<br/>'
     '+ Embedding Dropout(0.2)</font>',
     '[B, 129, 128]', C_TFR, B_TFR, 95),
    ('<b>Transformer Encoder &times; 4 Layers</b><br/><font style="font-size:10px;">'
     '<b>Pre-LN</b> MultiheadAttention (<b>4 heads</b>, d_model=128)<br/>'
     'FFN: Linear(128&rarr;512) &rarr; GELU &rarr; Drop &rarr; Linear(512&rarr;128)<br/>'
     '<b>Stochastic Depth</b>: linear schedule 0 &rarr; 0.1<br/>'
     'DropPath per-sample residual zeroing</font>',
     '[B, 129, 128]', C_TFR, B_TFR, 100),
    ('<b>LayerNorm &rarr; [CLS] Token Slice</b><br/><font style="font-size:10px;">'
     'x[:, 0, :] &mdash; dedicated summary embedding<br/>'
     'forward_features() output for GBDT export</font>',
     '<b>[B, 128]</b>', C_TFR, B_TFR, 70),
]

prev_rf = None; rf_ids = []
cy = RF_Y + 45
for label, shape, fill, stroke, h in rf_data:
    n = node(RF_X + 20, cy, 390, h, label, sty(fill, stroke, 11))
    node(RF_X + 420, cy + 10, 90, 30,
         f'<font style="font-size:10px;color:{stroke};">{shape}</font>',
         stxt(10, "left", stroke))
    if prev_rf: edge(prev_rf, n, SE_TFR)
    rf_ids.append(n); prev_rf = n; cy += h + 8

rf_embed_id = rf_ids[-1]

# Connect input to both models
edge(vec_in, ms_ids[0], SE_BOLD)
edge(vec_in, rf_ids[0], SE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: Training Losses (below RamanFormer heads)
# ═══════════════════════════════════════════════════════════════════════════
LOSS_Y = RF_Y + 700
grp(RF_X, LOSS_Y, 520, 200, "C. Training Losses (losses.py)", B_LSS)

# Projection head
proj = node(RF_X + 20, LOSS_Y + 40, 230, 70,
    '<b>Projection Head (Stage 1)</b><br/><font style="font-size:10px;">'
    'Linear(128&rarr;128) &rarr; BN &rarr; GELU<br/>'
    'Linear(128&rarr;128) &rarr; <b>L2-Norm</b><br/>'
    'Output on unit hypersphere</font>',
    sty(C_LSS, B_LSS, 11))
edge(rf_embed_id, proj, SE_TFR, "Stage 1")

# CE head
ce = node(RF_X + 270, LOSS_Y + 40, 230, 70,
    '<b>Classification Head (Stage 2)</b><br/><font style="font-size:10px;">'
    'Linear(128&rarr;256) &rarr; BN &rarr; GELU<br/>'
    'Dropout(0.2) &rarr; Linear(256&rarr;K)<br/>'
    'K = num_classes</font>',
    sty(C_LSS, B_LSS, 11))
edge(rf_embed_id, ce, SE_TFR, "Stage 2")

# Loss labels
node(RF_X + 20, LOSS_Y + 125, 230, 50,
    '<b>SupConLoss</b><br/><font style="font-size:10px;">'
    'Khosla et al. NeurIPS 2020<br/>'
    'temp=0.07, contrast_mode=all</font>',
    sty(C_LSS, B_LSS, 10, bold=False))
node(RF_X + 270, LOSS_Y + 125, 230, 50,
    '<b>FocalLoss</b><br/><font style="font-size:10px;">'
    'Lin et al. 2017, gamma=2.0<br/>'
    'label_smoothing=0.1</font>',
    sty(C_LSS, B_LSS, 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: Feature Extraction for Deployment
# ═══════════════════════════════════════════════════════════════════════════
FE_Y = LOSS_Y + 230
grp(40, FE_Y, 1040, 380, "D. Feature Extraction for Deployment (domain_features.py, prototype_features.py)", B_FEA)

# Also need input vector reference
feat_in = node(40, FE_Y - 45, 200, 35,
    '<b>2048-d Raw Spectrum</b><br/><font style="font-size:10px;">(from preprocessing)</font>',
    sty(C_IN, B_IN, 11))

# Branch 1: Domain features
dom = node(60, FE_Y + 45, 320, 200,
    '<b>Hand-Crafted Domain Features (163-d)</b><br/>'
    '<font style="font-size:10px;">'
    '<b>Peak features</b> (30-d): top-10 positions, heights, widths<br/>'
    '<b>Spectral moments</b> (4-d): mean, std, skew, kurtosis<br/>'
    '<b>Band means + ratios</b> (28-d):<br/>'
    '&nbsp; 7 diagnostic windows (silicate, carbonate, phosphate,<br/>'
    '&nbsp; oxide, CH-stretch, OH-stretch) + <sup>7</sup>C<sub>2</sub>=21 ratios<br/>'
    '<b>Subsampled spectrum</b> (64-d): stride-32 avg-pool<br/>'
    '<b>Derivative statistics</b> (12-d):<br/>'
    '&nbsp; 1st/2nd deriv: mean, std, max, min, ZCR, energy<br/>'
    '<b>Spectral entropy</b> (1-d): Shannon entropy<br/>'
    '<b>Band peak counts + areas</b> (14-d): per-window<br/>'
    '<b>Peak prominences</b> (10-d): sorted by prominence'
    '</font>',
    sty(C_FEA, B_FEA, 11))
edge(feat_in, dom, SE)

# Branch 2: PCA
pca = node(410, FE_Y + 45, 260, 90,
    '<b>PCA Global Shape (32-d)</b><br/>'
    '<font style="font-size:10px;">'
    'sklearn.decomposition.PCA<br/>'
    'n_components=32, whiten=True<br/>'
    'random_state=42<br/>'
    '<b>Fit on training set only</b>'
    '</font>',
    sty(C_FEA, B_FEA, 11))
edge(feat_in, pca, SE)

# Branch 3: Prototype similarity
proto = node(410, FE_Y + 160, 260, 120,
    '<b>Prototype Similarity (27-d)</b><br/>'
    '<font style="font-size:10px;">'
    'Class prototypes = mean spectrum per class<br/>'
    '<b>Fit on training data only</b><br/>'
    'Per sample &times; per class (K=9):<br/>'
    '&nbsp; &bull; Cosine similarity (9-d)<br/>'
    '&nbsp; &bull; Pearson correlation (9-d)<br/>'
    '&nbsp; &bull; Euclidean distance (9-d)'
    '</font>',
    sty(C_FEA, B_FEA, 11))
edge(feat_in, proto, SE)

# Branch 4: CNN embedding (optional, from trained models)
cnn_emb = node(700, FE_Y + 45, 360, 90,
    '<b>CNN Embedding (128-d, optional hybrid path)</b><br/>'
    '<font style="font-size:10px;">'
    'From trained MultiScaleSpecNet or RamanFormer1D<br/>'
    '.forward_features() with torch.no_grad()<br/>'
    'Batch inference (bs=64)<br/>'
    '<i>(Used in hybrid_models.py HeterogeneousFeatureExtractor)</i>'
    '</font>',
    sty(C_FEA, B_FEA, 11, dashed=True))
edge(ms_embed_id, cnn_emb, SE_DASH, "128-d embed")
edge(rf_embed_id, cnn_emb, SE_DASH, "128-d embed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: Feature Concatenation + StandardScaler
# ═══════════════════════════════════════════════════════════════════════════
CONCAT_Y = FE_Y + 410

concat = node(120, CONCAT_Y, 400, 60,
    '<b>Feature Concatenation (np.hstack)</b><br/>'
    '<font style="font-size:11px;">'
    'domain(163) + PCA(32) + prototype(27) = <b>222-d vector</b>'
    '</font>',
    sty(C_FEA, B_FEA, 12))
edge(dom, concat, SE)
edge(pca, concat, SE)
edge(proto, concat, SE)

# Optional extended
concat_ext = node(570, CONCAT_Y, 360, 60,
    '<b>Extended (hybrid path, optional)</b><br/>'
    '<font style="font-size:10px;">'
    'domain(163) + PCA(32) + prototype(27) + CNN(128) = <b>~350-d</b>'
    '</font>',
    sty(C_FEA, B_FEA, 11, dashed=True))
edge(cnn_emb, concat_ext, SE_DASH)
edge(concat, concat_ext, SE_DASH)

# Scaler
scaler = node(220, CONCAT_Y + 80, 200, 45,
    '<b>StandardScaler</b><br/><font style="font-size:10px;">Fit on train features, transform val/test</font>',
    sty(C_FEA, B_FEA, 11))
edge(concat, scaler, SE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION F: LightGBM-DART Classifier
# ═══════════════════════════════════════════════════════════════════════════
CLS_Y = CONCAT_Y + 155
grp(40, CLS_Y, 620, 260, "E. LightGBM-DART Classifier (train_deployment_model.py)", B_CLS)

lgbm = node(60, CLS_Y + 40, 340, 200,
    '<b>LightGBM-DART</b><br/>'
    '<font style="font-size:10px;">'
    '<b>Hyperparameter Optimisation: Optuna</b><br/>'
    'Sampler: TPE | Trials: 50 | Direction: maximize F1<br/><br/>'
    '<b>Search Space:</b><br/>'
    '&bull; n_estimators: 300&ndash;2000<br/>'
    '&bull; max_depth: 3&ndash;10<br/>'
    '&bull; learning_rate: 0.01&ndash;0.3 (log scale)<br/>'
    '&bull; subsample: 0.5&ndash;1.0<br/>'
    '&bull; colsample_bytree: 0.3&ndash;1.0<br/>'
    '&bull; num_leaves: 15&ndash;127<br/>'
    '&bull; reg_alpha / reg_lambda: 1e-3&ndash;10 (log)<br/>'
    '&bull; drop_rate (DART): 0.05&ndash;0.3<br/>'
    '&bull; class_weight: balanced<br/>'
    '&bull; objective: multiclass (K=9)'
    '</font>',
    sty(C_CLS, B_CLS, 11))
edge(scaler, lgbm, SE_BOLD)

# Comparison classifiers
comp = node(430, CLS_Y + 40, 210, 120,
    '<b>Comparison Models</b><br/>'
    '<font style="font-size:10px;">'
    '<b>CatBoost</b><br/>'
    'iterations=1000, depth=6<br/>'
    'lr=0.05, auto_class_weights<br/><br/>'
    '<b>TabPFN</b><br/>'
    'Zero-shot, CPU inference'
    '</font>',
    sty(C_CLS, B_CLS, 10, dashed=True))
edge(scaler, comp, SE_DASH)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: Ensemble
# ═══════════════════════════════════════════════════════════════════════════
ENS_Y = CLS_Y + 280
grp(40, ENS_Y, 620, 120, "F. Probability Ensemble (stacking_ensemble.py)", B_ENS)

ens = node(60, ENS_Y + 40, 300, 60,
    '<b>ProbabilityEnsemble</b><br/>'
    '<font style="font-size:10px;">'
    'Weighted probability averaging across models<br/>'
    'Optional: StackedEnsemble (LogisticRegression meta-learner)'
    '</font>',
    sty(C_ENS, B_ENS, 11))
edge(lgbm, ens, SE_BOLD)
edge(comp, ens, SE_DASH)

ens_out_shape = node(400, ENS_Y + 45, 240, 45,
    '<font style="font-size:10px;">'
    'Output: [B, K=9] averaged probabilities<br/>'
    'Final prediction: argmax'
    '</font>',
    stxt(10, "left", B_ENS))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION H: Confidence-Aware Output
# ═══════════════════════════════════════════════════════════════════════════
OUT_Y = ENS_Y + 140
grp(40, OUT_Y, 620, 240, "G. Confidence-Aware Screening Report", B_OUT)

out_node = node(60, OUT_Y + 40, 340, 180,
    '<b>Deployment Output</b><br/>'
    '<font style="font-size:11px;">'
    '1. <b>Predicted chemical family</b><br/>'
    '2. <b>Top-3 candidate families</b> with scores<br/>'
    '3. <b>Confidence score</b> (max probability)<br/>'
    '4. <b>Prediction quality tier:</b> High / Med / Low<br/>'
    '5. <b>Suggested next action</b><br/>'
    '6. <b>Screening-level disclaimer</b><br/><br/>'
    '<i>Output is a screening suggestion,<br/>'
    'not final experimental confirmation.</i>'
    '</font>',
    sty(C_OUT, B_OUT, 11))
edge(ens, out_node, SE_BOLD)

class_tags = node(430, OUT_Y + 40, 210, 180,
    '<b>9 Inorganic Material Families</b><br/>'
    '<font style="font-size:11px;">'
    '&bull; Silicate<br/>'
    '&bull; Oxide<br/>'
    '&bull; Carbonate<br/>'
    '&bull; Sulfate<br/>'
    '&bull; Phosphate<br/>'
    '&bull; Sulfide<br/>'
    '&bull; Halide<br/>'
    '&bull; Borate<br/>'
    '&bull; Other / Rare'
    '</font>',
    sty(C_OUT, B_OUT, 11, bold=False))

# ═══════════════════════════════════════════════════════════════════════════
# SAVED DEPLOYMENT ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
ART_Y = OUT_Y + 260
node(40, ART_Y, 620, 70,
    '<b>Saved Deployment Artifacts</b><br/>'
    '<font style="font-size:10px;">'
    '&bull; <b>pca.joblib</b> &mdash; fitted PCA(32) on training spectra &nbsp;'
    '&bull; <b>scaler.joblib</b> &mdash; fitted StandardScaler on training features<br/>'
    '&bull; <b>prototype_extractor.joblib</b> &mdash; class prototypes (train-only) &nbsp;'
    '&bull; <b>lgbm_deployment.joblib</b> &mdash; Optuna-tuned LightGBM-DART<br/>'
    '&bull; <b>encoder.json</b> &mdash; label &rarr; integer mapping (9 families)'
    '</font>',
    sty(C_ANN, B_ANN, 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════
# CAPTION
# ═══════════════════════════════════════════════════════════════════════════
node(40, ART_Y + 90, 1040, 60,
     '<font style="font-size:11px;"><b>Figure (b):</b> Core Spec2Prop-Edge model architecture. '
     'Two flagship deep encoders &mdash; MultiScaleSpecNet (ASPP + Channel Attention) and RamanFormer1D v2 '
     '(4-stage CNN tokenizer + 4-layer Transformer with SupCon) &mdash; produce 128-d spectral embeddings. '
     'For deployment, raw 2048-point spectra are transformed into a 222-d heterogeneous feature vector '
     '(163 domain + 32 PCA + 27 prototype features), standardised, and classified by an Optuna-tuned '
     'LightGBM-DART classifier. The ensemble layer combines predictions from multiple classifiers. '
     'The system outputs top-k inorganic family predictions with confidence scores for scientist verification.</font>',
     stxt(11, "left", "#37474F"))

# ═══════════════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(__file__), "Diagram2_Core_Model_Architecture.drawio")
if hasattr(ET, "indent"):
    ET.indent(ET.ElementTree(mxfile), space="  ", level=0)
ET.ElementTree(mxfile).write(out_path, encoding="utf-8", xml_declaration=True)
print(f"Generated: {out_path}")
