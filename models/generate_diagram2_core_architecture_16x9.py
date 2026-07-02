"""
Diagram 2: Core Model Architecture (16:9 Widescreen Layout)
===========================================================
MultiScaleSpecNet + RamanFormer1D + Training Losses
+ Feature Extraction + Concatenation + LightGBM-DART
+ Ensemble + Confidence-Aware Output

Optimised for a 16:9 aspect ratio presentation.
No baselines. No side panels. Clean SOTA figure.
"""
import xml.etree.ElementTree as ET
import os

mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "21.1.2", "type": "device"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "d2_16x9", "name": "Core Model 16:9"})
# 3200 x 1800 is 16:9
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "2200", "dy": "2200", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "3200",
    "pageHeight": "1800", "math": "0", "shadow": "0", "background": "#FFFFFF"
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
node(40, 25, 2800, 40,
     '<font style="font-size:22px;"><b>Diagram 2 &mdash; Spec2Prop-Edge: Core Model Architecture (16:9 Widescreen)</b></font>',
     stxt(22, "center", "#1A237E"))

# ═══════════════════════════════════════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════════════════════════════════════
vec_in = node(350, 100, 320, 50,
    '<b>From Preprocessing (Diagram 1)</b><br/><font style="font-size:11px;">2048-d Spectral Vector [B, 1, 2048]</font>',
    sty(C_IN, B_IN, 12))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: MultiScaleSpecNet (COLUMN 1)
# ═══════════════════════════════════════════════════════════════════════════
MS_X = 60; MS_Y = 200
grp(MS_X, MS_Y, 480, 680, "A. MultiScaleSpecNet (cnn1d.py)", B_CNN)

ms_data = [
    ('<b>Stem: Conv1D</b><br/><font style="font-size:10px;">in=1, out=32, k=31, pad=15<br/>BN &rarr; GELU &rarr; MaxPool1d(4)</font>', '[B, 32, 512]', 85),
    ('<b>ASPP Block 1</b><br/><font style="font-size:10px;">4&times; parallel Conv1d(k=3, d=1,6,12,18)<br/>Concat &rarr; 1&times;1 Conv &rarr; BN &rarr; ReLU</font>', '[B, 64, 512]', 85),
    ('<b>CAS 1 &mdash; Channel Attention Squeeze</b><br/><font style="font-size:10px;">GAP &rarr; FC(64&rarr;8) &rarr; ReLU &rarr; FC(8&rarr;64) &rarr; Sigmoid<br/>Scale + MaxPool1d(4)</font>', '[B, 64, 128]', 85),
    ('<b>ASPP Block 2</b><br/><font style="font-size:10px;">4&times; parallel Conv1d(k=3, d=1,6,12,18)<br/>Concat &rarr; 1&times;1 Conv &rarr; BN &rarr; ReLU</font>', '[B, 128, 128]', 85),
    ('<b>CAS 2 &mdash; Channel Attention Squeeze</b><br/><font style="font-size:10px;">GAP &rarr; FC(128&rarr;16) &rarr; ReLU &rarr; FC(16&rarr;128) &rarr; Sigmoid<br/>Scale + MaxPool1d(4)</font>', '[B, 128, 32]', 85),
    ('<b>Global Average Pooling &rarr; FC Embed</b><br/><font style="font-size:10px;">GAP &rarr; squeeze &rarr; Linear(128&rarr;128) &rarr; ReLU &rarr; Drop(0.3)</font>', '<b>[B, 128]</b>', 70),
]

prev_ms = None; ms_ids = []
cy = MS_Y + 45
for label, shape, h in ms_data:
    n = node(MS_X + 20, cy, 340, h, label, sty(C_CNN, B_CNN, 11))
    node(MS_X + 370, cy + 10, 90, 30, f'<font style="font-size:10px;color:{B_CNN};">{shape}</font>', stxt(10, "left", B_CNN))
    if prev_ms: edge(prev_ms, n, SE_EMB)
    ms_ids.append(n); prev_ms = n; cy += h + 22

ms_embed_id = ms_ids[-1]
edge(vec_in, ms_ids[0], SE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: RamanFormer1D v2 (COLUMN 2)
# ═══════════════════════════════════════════════════════════════════════════
RF_X = 580; RF_Y = 200
grp(RF_X, RF_Y, 520, 680, "B. RamanFormer1D v2 (raman_former.py)", B_TFR)

rf_data = [
    ('<b>Tokenizer 1:</b> Conv1D(1&rarr;32, k=7, s=2) &rarr; BN &rarr; GELU', '[B, 32, 1024]', C_CNN, B_CNN, 50),
    ('<b>Tokenizer 2:</b> Conv1D(32&rarr;64, k=5, s=2) &rarr; BN &rarr; GELU', '[B, 64, 512]', C_CNN, B_CNN, 50),
    ('<b>Tokenizer 3:</b> ResConv1D(64&rarr;128, k=3, s=2) + shortcut', '[B, 128, 256]', C_CNN, B_CNN, 50),
    ('<b>Tokenizer 4:</b> ResConv1D(128&rarr;128, k=3, s=2) + shortcut', '[B, 128, 128]', C_CNN, B_CNN, 50),
    ('<b>Token Preparation</b><br/><font style="font-size:10px;">Permute(0,2,1) &rarr; Prepend learned <b>[CLS] token</b><br/>+ Learned Positional Embeddings &rarr; Dropout</font>', '[B, 129, 128]', C_TFR, B_TFR, 80),
    ('<b>Transformer Encoder &times; 4 Layers</b><br/><font style="font-size:10px;">Pre-LN Attention (4 heads, d=128)<br/>FFN(128&rarr;512&rarr;128, GELU)<br/>Stochastic Depth (0&rarr;0.1)</font>', '[B, 129, 128]', C_TFR, B_TFR, 90),
    ('<b>LayerNorm &rarr; [CLS] Token Slice</b><br/><font style="font-size:10px;">Extract x[:, 0, :]</font>', '<b>[B, 128]</b>', C_TFR, B_TFR, 60),
]

prev_rf = None; rf_ids = []
cy = RF_Y + 45
for label, shape, fill, stroke, h in rf_data:
    n = node(RF_X + 20, cy, 360, h, label, sty(fill, stroke, 11))
    node(RF_X + 390, cy + 10, 110, 30, f'<font style="font-size:10px;color:{stroke};">{shape}</font>', stxt(10, "left", stroke))
    if prev_rf: edge(prev_rf, n, SE_TFR)
    rf_ids.append(n); prev_rf = n; cy += h + 22

rf_embed_id = rf_ids[-1]
edge(vec_in, rf_ids[0], SE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: Training Losses
# ═══════════════════════════════════════════════════════════════════════════
LOSS_Y = RF_Y + 720
grp(RF_X, LOSS_Y, 520, 220, "C. Training Losses", B_LSS)

proj = node(RF_X + 20, LOSS_Y + 40, 230, 70,
    '<b>Proj Head (Stage 1)</b><br/><font style="font-size:10px;">Linear(128&rarr;128) &rarr; BN &rarr; GELU<br/>Linear &rarr; <b>L2-Norm</b></font>', sty(C_LSS, B_LSS, 11))
edge(rf_embed_id, proj, SE_TFR, "Stage 1")

ce = node(RF_X + 270, LOSS_Y + 40, 230, 70,
    '<b>Class Head (Stage 2)</b><br/><font style="font-size:10px;">Linear(128&rarr;256) &rarr; BN &rarr; GELU<br/>Drop(0.2) &rarr; Linear(256&rarr;K)</font>', sty(C_LSS, B_LSS, 11))
edge(rf_embed_id, ce, SE_TFR, "Stage 2")

node(RF_X + 20, LOSS_Y + 130, 230, 60,
    '<b>SupConLoss</b><br/><font style="font-size:10px;">Khosla et al. 2020<br/>temp=0.07, mode=all</font>', sty(C_LSS, B_LSS, 10, bold=False))
node(RF_X + 270, LOSS_Y + 130, 230, 60,
    '<b>FocalLoss</b><br/><font style="font-size:10px;">Lin et al. 2017, gamma=2.0<br/>label_smoothing=0.1</font>', sty(C_LSS, B_LSS, 10, bold=False))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: Feature Extraction (COLUMN 3 & 4)
# ═══════════════════════════════════════════════════════════════════════════
FE_X = 1140; FE_Y = 200
grp(FE_X, FE_Y, 1060, 440, "D. Feature Extraction for Deployment (domain_features.py)", B_FEA)

feat_in = node(FE_X + 40, FE_Y - 50, 200, 40, '<b>2048-d Raw Spectrum</b>', sty(C_IN, B_IN, 11))
edge(vec_in, feat_in, SE)

dom = node(FE_X + 20, FE_Y + 45, 340, 220,
    '<b>Domain Features (163-d)</b><br/>'
    '<font style="font-size:10px;">'
    '<b>Peaks</b> (30): top-10 pos, height, width<br/>'
    '<b>Moments</b> (4): mean, std, skew, kurtosis<br/>'
    '<b>Bands</b> (28): 7 windows + 21 ratios<br/>'
    '<b>Subsampled</b> (64): stride-32 avg-pool<br/>'
    '<b>Derivatives</b> (12): stats of 1st/2nd deriv<br/>'
    '<b>Entropy</b> (1): Shannon entropy<br/>'
    '<b>Band areas</b> (14): peak count + area<br/>'
    '<b>Prominences</b> (10): sorted'
    '</font>', sty(C_FEA, B_FEA, 11))
edge(feat_in, dom, SE)

pca = node(FE_X + 390, FE_Y + 45, 300, 90,
    '<b>PCA Global Shape (32-d)</b><br/><font style="font-size:10px;">sklearn PCA, n_comp=32, whiten=True<br/><b>Fit on training set only</b></font>', sty(C_FEA, B_FEA, 11))
edge(feat_in, pca, SE)

proto = node(FE_X + 390, FE_Y + 160, 300, 110,
    '<b>Prototype Similarity (27-d)</b><br/><font style="font-size:10px;">Class prototypes = mean spectrum per class<br/>Per sample &times; per class (K=9):<br/>Cosine, Pearson, Euclidean</font>', sty(C_FEA, B_FEA, 11))
edge(feat_in, proto, SE)

cnn_emb = node(FE_X + 720, FE_Y + 45, 320, 100,
    '<b>CNN Embedding (128-d, optional)</b><br/><font style="font-size:10px;">From trained MultiScaleSpecNet or RamanFormer<br/>.forward_features() with torch.no_grad()<br/><i>(hybrid_models.py path)</i></font>', sty(C_FEA, B_FEA, 11, dashed=True))
edge(ms_embed_id, cnn_emb, SE_DASH, "128-d")
edge(rf_embed_id, cnn_emb, SE_DASH, "128-d")

# ═══════════════════════════════════════════════════════════════════════════
# CONCATENATION & SCALER
# ═══════════════════════════════════════════════════════════════════════════
CONCAT_Y = FE_Y + 470
concat = node(FE_X + 20, CONCAT_Y, 450, 60,
    '<b>Feature Concatenation</b><br/><font style="font-size:11px;">domain(163) + PCA(32) + prototype(27) = <b>222-d</b></font>', sty(C_FEA, B_FEA, 12))
edge(dom, concat, SE)
edge(pca, concat, SE)
edge(proto, concat, SE)

concat_ext = node(FE_X + 500, CONCAT_Y, 400, 60,
    '<b>Extended Concat</b><br/><font style="font-size:10px;">domain(163) + PCA(32) + prototype(27) + CNN(128) = <b>~350-d</b></font>', sty(C_FEA, B_FEA, 11, dashed=True))
edge(cnn_emb, concat_ext, SE_DASH)
edge(concat, concat_ext, SE_DASH)

scaler = node(FE_X + 100, CONCAT_Y + 90, 290, 50,
    '<b>StandardScaler</b><br/><font style="font-size:10px;">Fit on train features, transform val/test</font>', sty(C_FEA, B_FEA, 11))
edge(concat, scaler, SE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: LightGBM-DART & F: Ensemble (COLUMN 3)
# ═══════════════════════════════════════════════════════════════════════════
LGBM_X = 1140; LGBM_Y = CONCAT_Y + 180
grp(LGBM_X, LGBM_Y, 750, 320, "E. Deployment Classifiers", B_CLS)

lgbm = node(LGBM_X + 30, LGBM_Y + 45, 380, 220,
    '<b>LightGBM-DART</b><br/>'
    '<font style="font-size:10px;">'
    '<b>Optuna HPO (50 trials, TPE sampler)</b><br/>'
    '&bull; n_estimators: 300&ndash;2000<br/>'
    '&bull; max_depth: 3&ndash;10<br/>'
    '&bull; learning_rate: 0.01&ndash;0.3 (log)<br/>'
    '&bull; subsample: 0.5&ndash;1.0<br/>'
    '&bull; colsample_bytree: 0.3&ndash;1.0<br/>'
    '&bull; num_leaves: 15&ndash;127<br/>'
    '&bull; reg_alpha / lambda: 1e-3&ndash;10 (log)<br/>'
    '&bull; drop_rate: 0.05&ndash;0.3<br/>'
    '&bull; class_weight: balanced<br/>'
    '</font>', sty(C_CLS, B_CLS, 11))
edge(scaler, lgbm, SE_BOLD)

comp = node(LGBM_X + 440, LGBM_Y + 45, 280, 130,
    '<b>Comparison Models</b><br/>'
    '<font style="font-size:10px;">'
    '<b>CatBoost</b><br/>'
    'depth=6, lr=0.05, 1000 iter<br/><br/>'
    '<b>TabPFN</b><br/>'
    'Zero-shot, CPU inference'
    '</font>', sty(C_CLS, B_CLS, 10, dashed=True))
edge(scaler, comp, SE_DASH)

ENS_Y = LGBM_Y + 350
grp(LGBM_X, ENS_Y, 750, 150, "F. Ensemble", B_ENS)
ens = node(LGBM_X + 30, ENS_Y + 45, 360, 70,
    '<b>ProbabilityEnsemble</b><br/><font style="font-size:10px;">Weighted probability averaging<br/>or StackedEnsemble (LogisticRegression)</font>', sty(C_ENS, B_ENS, 11))
edge(lgbm, ens, SE_BOLD)
edge(comp, ens, SE_DASH)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: Output & Artifacts (COLUMN 4)
# ═══════════════════════════════════════════════════════════════════════════
OUT_X = 1950; OUT_Y = LGBM_Y
grp(OUT_X, OUT_Y, 750, 320, "G. Confidence-Aware Screening Report", B_OUT)

out_node = node(OUT_X + 30, OUT_Y + 45, 400, 200,
    '<b>Deployment Output</b><br/>'
    '<font style="font-size:11px;">'
    '1. <b>Predicted chemical family</b><br/>'
    '2. <b>Top-3 candidate families</b> with scores<br/>'
    '3. <b>Confidence score</b> (max prob)<br/>'
    '4. <b>Prediction quality:</b> High / Med / Low<br/>'
    '5. <b>Suggested next action</b><br/>'
    '6. <b>Screening-level disclaimer</b><br/><br/>'
    '<i>Screening suggestion, not final verification.</i>'
    '</font>', sty(C_OUT, B_OUT, 11))
edge(ens, out_node, SE_BOLD)

tags = node(OUT_X + 460, OUT_Y + 45, 260, 200,
    '<b>9 Material Families</b><br/>'
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
    '</font>', sty(C_OUT, B_OUT, 11, bold=False))

# Artifacts
grp(OUT_X, ENS_Y, 750, 150, "Deployment Artifacts", B_ANN)
node(OUT_X + 30, ENS_Y + 45, 690, 80,
    '<font style="font-size:10px;">'
    '&bull; <b>pca.joblib</b> &mdash; fitted PCA(32)<br/>'
    '&bull; <b>scaler.joblib</b> &mdash; fitted StandardScaler<br/>'
    '&bull; <b>prototype_extractor.joblib</b> &mdash; class prototypes<br/>'
    '&bull; <b>lgbm_deployment.joblib</b> &mdash; Optuna-tuned LightGBM<br/>'
    '&bull; <b>encoder.json</b> &mdash; label mappings'
    '</font>', sty(C_ANN, B_ANN, 11, bold=False))

# ═══════════════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(__file__), "Diagram2_Core_Model_Architecture.drawio")
if hasattr(ET, "indent"):
    ET.indent(ET.ElementTree(mxfile), space="  ", level=0)
ET.ElementTree(mxfile).write(out_path, encoding="utf-8", xml_declaration=True)
print(f"Generated: {out_path}")
