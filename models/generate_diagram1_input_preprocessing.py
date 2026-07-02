"""
Diagram 1: Input Sources & Runtime Spectral Preprocessing Pipeline
===================================================================
Clean, publication-ready draw.io diagram.
"""
import xml.etree.ElementTree as ET
import os

mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "21.1.2", "type": "device"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "d1", "name": "Input and Preprocessing"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1400", "dy": "900", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "1800",
    "pageHeight": "900", "math": "0", "shadow": "0", "background": "#FFFFFF"
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

def grp(x, y, w, h, label, stroke, fill="none"):
    i = nid()
    st = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"strokeWidth=2;dashed=0;arcSize=8;verticalAlign=top;"
          f"fontFamily=Helvetica;fontSize=15;fontStyle=1;fontColor={stroke};"
          f"labelBackgroundColor=#FFFFFF;spacingTop=5;container=0;")
    c = ET.SubElement(root, "mxCell", {"id": i, "value": label, "style": st, "vertex": "1", "parent": "1"})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return i

# Colours
C_IN   = "#E8EAF6"; B_IN   = "#283593"
C_PREP = "#E3F2FD"; B_PREP = "#1565C0"
C_VEC  = "#BBDEFB"; B_VEC  = "#0D47A1"

def sty(fill, stroke, fs=12, bold=True, dashed=False):
    b = "1" if bold else "0"
    dp = "strokeDasharray=8 4;" if dashed else ""
    return (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontFamily=Helvetica;fontSize={fs};fontStyle={b};"
            f"arcSize=10;{dp}")

def stxt(fs=11, al="left", col="#37474F"):
    return (f"text;html=1;strokeColor=none;fillColor=none;align={al};"
            f"verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;"
            f"fontSize={fs};fontColor={col};")

SE = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#37474F;strokeWidth=2;fontFamily=Helvetica;fontSize=11;"
SE_BOLD = SE.replace("strokeWidth=2", "strokeWidth=3").replace("#37474F", "#1565C0")
SE_DASH = SE + "dashed=1;dashPattern=6 3;strokeColor=#9E9E9E;"

# ═══════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════
node(40, 20, 1700, 40,
     '<font style="font-size:18px;"><b>Diagram 1 &mdash; Input Sources &amp; Runtime Spectral Preprocessing Pipeline</b></font>',
     stxt(18, "center", "#1A237E"))

# ═══════════════════════════════════════════════════════════════════════════
# Section A: Input Sources
# ═══════════════════════════════════════════════════════════════════════════
grp(40, 80, 340, 380, "Input Sources", B_IN)

raman_box = node(60, 130, 300, 120,
    '<b>Raman Spectrometer Output</b><br/>'
    '<font style="font-size:11px;">'
    '<b>File format:</b> CSV / TXT<br/>'
    '<b>X-axis:</b> Raman shift (cm<sup>-1</sup>)<br/>'
    '<b>Y-axis:</b> Intensity (a.u.)<br/>'
    '<b>Physical meaning:</b> Vibrational fingerprint<br/>'
    'of molecular bonds and crystal structure<br/>'
    '<b>Role:</b> Primary input (v1 deployment)'
    '</font>',
    sty(C_IN, B_IN, 12))

xrd_box = node(60, 280, 300, 120,
    '<b>XRD Diffractometer Output</b><br/>'
    '<font style="font-size:11px;">'
    '<b>File format:</b> CSV / TXT<br/>'
    '<b>X-axis:</b> 2&#x3b8; angle (degrees)<br/>'
    '<b>Y-axis:</b> Intensity (counts)<br/>'
    '<b>Physical meaning:</b> Structural fingerprint<br/>'
    'of crystallographic planes<br/>'
    '<b>Role:</b> Optional / multimodal extension'
    '</font>',
    sty(C_IN, B_IN, 12, dashed=True))

# label: primary vs optional
node(60, 420, 130, 25, '<font style="font-size:10px;"><b>Primary input</b></font>', stxt(10, "center", "#2E7D32"))
node(230, 420, 130, 25, '<font style="font-size:10px;"><i>Optional input</i></font>', stxt(10, "center", "#9E9E9E"))

# ═══════════════════════════════════════════════════════════════════════════
# Section B: Preprocessing Pipeline (8 sequential steps)
# ═══════════════════════════════════════════════════════════════════════════
grp(420, 80, 920, 380, "Runtime Spectral Preprocessing (executed at inference time, not cached)", B_PREP)

steps = [
    ("1. Raw File Parsing",
     "Read CSV/TXT, detect delimiter,\nextract (x, y) columns"),
    ("2. Invalid Row Removal",
     "Drop NaN, Inf, negative intensities,\nnon-numeric entries"),
    ("3. Axis Sorting &amp;\n    Duplicate Removal",
     "Sort by wavenumber/angle,\nremove duplicate x-values"),
    ("4. Interpolation to\n    Fixed Grid",
     "Cubic spline interpolation\nto exactly 2048 equally-spaced points"),
    ("5. Baseline Correction",
     "Asymmetric least-squares (ALS)\nor polynomial baseline subtraction"),
    ("6. Savitzky-Golay\n    Smoothing",
     "Polynomial smoothing filter\npreserves peak shapes"),
    ("7. Min-Max\n    Normalisation",
     "Scale intensities to [0, 1]\nper-spectrum normalisation"),
    ("8. Output: 2048-point\n    Spectral Vector",
     "Fixed-length representation\nready for feature extraction"),
]

BX = 440; BY = 130; BW = 200; BH = 70; GAP = 10
prev = None
step_ids = []
for i, (title, desc) in enumerate(steps):
    col = i % 4
    row = i // 4
    x = BX + col * (BW + GAP)
    y = BY + row * (BH + 85)
    fill = C_PREP if i < 7 else C_VEC
    stroke = B_PREP if i < 7 else B_VEC
    n = node(x, y, BW, BH + 20,
             f'<b>{title}</b><br/><font style="font-size:10px;">{desc}</font>',
             sty(fill, stroke, 11))
    step_ids.append(n)

# Connect steps sequentially
for i in range(len(step_ids) - 1):
    edge(step_ids[i], step_ids[i+1], SE)

# Connect inputs to step 1
edge(raman_box, step_ids[0], SE_BOLD, "Primary")
edge(xrd_box, step_ids[0], SE_DASH, "Optional")

# ═══════════════════════════════════════════════════════════════════════════
# Section C: Output vectors
# ═══════════════════════════════════════════════════════════════════════════
grp(1380, 80, 380, 380, "Preprocessed Output", B_VEC)

raman_vec = node(1400, 150, 340, 80,
    '<b>Raman Spectral Vector</b><br/>'
    '<font style="font-size:12px;">'
    '<b>Shape: [1 &times; 2048]</b><br/>'
    'dtype: float32, range: [0, 1]<br/>'
    'Fixed grid: ~100&ndash;4000 cm<sup>-1</sup>'
    '</font>',
    sty(C_VEC, B_VEC, 13))

xrd_vec = node(1400, 270, 340, 80,
    '<b>XRD Spectral Vector (optional)</b><br/>'
    '<font style="font-size:12px;">'
    '<b>Shape: [1 &times; 2048]</b><br/>'
    'dtype: float32, range: [0, 1]<br/>'
    'Fixed grid: 2&#x3b8; range'
    '</font>',
    sty(C_VEC, B_VEC, 12, dashed=True))

edge(step_ids[-1], raman_vec, SE_BOLD)
edge(step_ids[-1], xrd_vec, SE_DASH)

# Arrow annotation: "To Diagram 2"
dest = node(1400, 390, 340, 40,
    '<font style="font-size:12px;"><b>&rarr; To Diagram 2: Feature Extraction &amp; Model Pipeline</b></font>',
    stxt(12, "center", "#0D47A1"))

# ═══════════════════════════════════════════════════════════════════════════
# Caption
# ═══════════════════════════════════════════════════════════════════════════
node(40, 490, 1700, 50,
     '<font style="font-size:11px;"><b>Figure (a):</b> Input sources and runtime spectral preprocessing pipeline. '
     'Raw Raman spectrometer files (primary) or XRD diffractometer files (optional) are parsed, cleaned, interpolated '
     'to a fixed 2048-point grid, baseline-corrected, smoothed, and normalised at inference time. '
     'The resulting float32 spectral vectors are passed to the feature extraction and classification pipeline (Diagram 2). '
     'All preprocessing is performed at runtime on each new sample, not from pre-cached data.</font>',
     stxt(11, "left", "#37474F"))

# Write
out = os.path.join(os.path.dirname(__file__), "Diagram1_Input_Preprocessing.drawio")
if hasattr(ET, "indent"):
    ET.indent(ET.ElementTree(mxfile), space="  ", level=0)
ET.ElementTree(mxfile).write(out, encoding="utf-8", xml_declaration=True)
print(f"Generated: {out}")
