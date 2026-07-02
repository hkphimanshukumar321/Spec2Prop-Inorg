import xml.etree.ElementTree as ET

mxfile = ET.Element("mxfile", {"host": "Electron", "version": "21.1.2", "type": "device"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "page_1", "name": "Custom Models"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1000", "dy": "1000", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "1600",
    "pageHeight": "1200", "math": "0", "shadow": "0"
})
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

cell_id = 2

def add_node(x, y, w, h, text, style, parent="1"):
    global cell_id
    node = ET.SubElement(root, "mxCell", {
        "id": str(cell_id), "value": text, "style": style, "vertex": "1", "parent": parent
    })
    ET.SubElement(node, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    cell_id += 1
    return cell_id - 1

def add_edge(src, tgt, style):
    global cell_id
    edge = ET.SubElement(root, "mxCell", {
        "id": str(cell_id), "style": style, "edge": "1", "parent": "1", "source": str(src), "target": str(tgt)
    })
    ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    cell_id += 1
    return cell_id - 1

# Styles
s_main = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontFamily=Helvetica;fontSize=13;fontStyle=1;"
s_out = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontFamily=Helvetica;fontSize=13;fontStyle=1;"
s_txt = "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;fontSize=14;fontColor=#444444;"
s_title = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;fontSize=20;fontStyle=1;"
s_edge = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#333333;strokeWidth=2;"

# RamanFormer1D
add_node(150, 40, 300, 40, "RamanFormer1D v2 (SupCon-Ready)", s_title)

rf_nodes = [
    ("Raw Spectrum", "[B, 1, 2048]"),
    ("Tokenizer 1: Conv1D (k=7, s=2)\nBN + GELU", "Resolution: [B, 32, 1024]"),
    ("Tokenizer 2: Conv1D (k=5, s=2)\nBN + GELU", "Resolution: [B, 64, 512]"),
    ("Tokenizer 3: ResConv1D (k=3, s=2)", "Resolution: [B, 128, 256]"),
    ("Tokenizer 4: ResConv1D (k=3, s=2)", "Resolution: [B, 128, 128]"),
    ("Token Prep: Permute & Prepend [CLS]\n+ Learned Positional Emb", "Tokens: [B, 129, 128]"),
    ("Transformer Encoder (x4)\n4-Heads, FFN=512, StochDepth", "Attention: [B, 129, 128]"),
    ("Slice [CLS] Token", "Embedding: [B, 128]")
]

y = 100
prev = None
for name, res in rf_nodes:
    n = add_node(150, y, 240, 60, name, s_main)
    add_node(410, y, 200, 60, res, s_txt)
    if prev:
        add_edge(prev, n, s_edge)
    prev = n
    y += 90

# Heads
h1 = add_node(50, y, 200, 60, "Proj Head (SupCon)\nMLP -> L2 Norm", s_out)
h2 = add_node(290, y, 200, 60, "Classification Head (CE)\nMLP -> Softmax", s_out)
add_node(80, y+70, 180, 30, "[B, proj_dim=128]", s_txt)
add_node(320, y+70, 180, 30, "[B, num_classes]", s_txt)
add_edge(prev, h1, s_edge)
add_edge(prev, h2, s_edge)


# MultiScaleSpecNet
add_node(750, 40, 300, 40, "MultiScaleSpecNet (CNN1D)", s_title)

ms_nodes = [
    ("Raw Spectrum", "[B, 1, 2048]"),
    ("Stem: Conv1D (k=31, p=15, s=1)\n+ MaxPool1D(4)", "Resolution: [B, 32, 512]"),
    ("ASPP 1\nDilations: 1, 6, 12, 18", "Resolution: [B, 64, 512]"),
    ("CAS 1: Channel Attention (red=8)\n+ MaxPool1D(4)", "Resolution: [B, 64, 128]"),
    ("ASPP 2\nDilations: 1, 6, 12, 18", "Resolution: [B, 128, 128]"),
    ("CAS 2: Channel Attention (red=8)\n+ MaxPool1D(4)", "Resolution: [B, 128, 32]"),
    ("Global Average Pooling (GAP)", "Embedding: [B, 128]"),
    ("FC Embedding Layer", "Embedding: [B, 128]"),
    ("Classification Head", "Output: [B, num_classes]")
]

y = 100
prev = None
for name, res in ms_nodes:
    style = s_main if "Head" not in name else s_out
    n = add_node(750, y, 240, 60, name, style)
    add_node(1010, y, 200, 60, res, s_txt)
    if prev:
        add_edge(prev, n, s_edge)
    prev = n
    y += 90

tree = ET.ElementTree(mxfile)
if hasattr(ET, "indent"):
    ET.indent(tree, space="  ", level=0)
tree.write("c:/Users/hkphi/OneDrive/Desktop/WORK/FICS/models/Custom_Models_Architecture.drawio", encoding="utf-8", xml_declaration=True)
