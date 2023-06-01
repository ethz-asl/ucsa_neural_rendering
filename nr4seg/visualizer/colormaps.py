import numpy as np

from collections import OrderedDict
from matplotlib import cm

ORDERED_DICT = OrderedDict([
    ("unlabeled", (0, 0, 0)),
    ("wall", (174, 199, 232)),
    ("floor", (152, 223, 138)),
    ("cabinet", (31, 119, 180)),
    ("bed", (255, 187, 120)),
    ("chair", (188, 189, 34)),
    ("sofa", (140, 86, 75)),
    ("table", (255, 152, 150)),
    ("door", (214, 39, 40)),
    ("window", (197, 176, 213)),
    ("bookshelf", (148, 103, 189)),
    ("picture", (196, 156, 148)),
    ("counter", (23, 190, 207)),
    ("blinds", (178, 76, 76)),
    ("desk", (247, 182, 210)),
    ("shelves", (66, 188, 102)),
    ("curtain", (219, 219, 141)),
    ("dresser", (140, 57, 197)),
    ("pillow", (202, 185, 52)),
    ("mirror", (51, 176, 203)),
    ("floormat", (200, 54, 131)),
    ("clothes", (92, 193, 61)),
    ("ceiling", (78, 71, 183)),
    ("books", (172, 114, 82)),
    ("refrigerator", (255, 127, 14)),
    ("television", (91, 163, 138)),
    ("paper", (153, 98, 156)),
    ("towel", (140, 153, 101)),
    ("showercurtain", (158, 218, 229)),
    ("box", (100, 125, 154)),
    ("whiteboard", (178, 127, 135)),
    ("person", (120, 185, 128)),
    ("nightstand", (146, 111, 194)),
    ("toilet", (44, 160, 44)),
    ("sink", (112, 128, 144)),
    ("lamp", (96, 207, 209)),
    ("bathtub", (227, 119, 194)),
    ("bag", (213, 92, 176)),
    ("otherstructure", (94, 106, 211)),
    ("otherfurniture", (82, 84, 163)),
    ("otherprop", (100, 85, 144)),
])
SCANNET_CLASSES = [i for i, v in enumerate(ORDERED_DICT.values())]
SCANNET_COLORS = [v for i, v in enumerate(ORDERED_DICT.values())]

jet = cm.get_cmap("jet")
BINARY_COLORS = (np.stack([jet(v) for v in np.linspace(0, 1, 2)]) * 255).astype(
    np.uint8)
