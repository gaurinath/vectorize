"""Core raster-to-vector conversion functions.

All functions accept a PIL Image and return an SVG string, so they can be
reused from both the CLI scripts (vectoriser.py, multicolor_vectoriser.py)
and the Streamlit web app (app.py).
"""

import io

import numpy as np
import vtracer
from PIL import Image, ImageFilter
from potrace import POTRACE_TURNPOLICY_MINORITY, Bitmap
from sklearn.cluster import KMeans

MAX_KMEANS_PIXELS = 1_000_000


# ---------------------------------------------------------------- Potrace B&W


def to_binary(image: Image.Image, threshold: int = 180, blur: float = 0.5) -> np.ndarray:
    """Binarize a PIL image into a bool foreground mask.

    Darker pixels become foreground (True) by default. Raise `threshold` to also
    capture lighter shades; use `invert=True` in potrace_bw_svg for the reverse.
    """
    gray = image.convert("L")
    if blur > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.array(gray)
    return arr < threshold


def potrace_bw_svg(
    image: Image.Image,
    threshold: int = 180,
    blur: float = 0.5,
    turdsize: int = 4,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    opticurve: bool = True,
    invert: bool = False,
    fill: str = "#000000",
) -> str:
    """Trace a binarized image into an SVG using Potrace and return SVG text."""
    gray = image.convert("L")
    if blur > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.array(gray)
    binary = (arr < threshold) if not invert else (arr >= threshold)

    height, width = binary.shape
    bmp = Bitmap(binary)
    path_list = bmp.trace(
        turdsize=turdsize,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=alphamax,
        opticurve=opticurve,
        opttolerance=opttolerance,
    )

    path_strings = []
    for curve in path_list:
        start = curve.start_point
        parts = [f"M {start.x:.2f} {start.y:.2f}"]
        for segment in curve.segments:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                parts.append(f"L {c.x:.2f} {c.y:.2f} L {end.x:.2f} {end.y:.2f}")
            else:
                c1, c2, end = segment.c1, segment.c2, segment.end_point
                parts.append(
                    f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}"
                )
        parts.append("Z")
        path_strings.append(" ".join(parts))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        f'  <path d="{" ".join(path_strings)}" fill="{fill}" fill-rule="evenodd" />\n'
        f"</svg>\n"
    )


# --------------------------------------------------------- Potrace multi-color


def rgb_to_hex(rgb_tuple) -> str:
    return f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"


def quantize_image(img: Image.Image, n_colors: int) -> tuple[np.ndarray, list]:
    """Cluster image colors into n_colors clusters, return label grid + centroids."""
    img_rgb = img.convert("RGB")
    width, height = img_rgb.size
    data = np.array(img_rgb).reshape((-1, 3))

    total = data.shape[0]
    if total > MAX_KMEANS_PIXELS:
        rng = np.random.RandomState(42)
        idx = rng.choice(total, MAX_KMEANS_PIXELS, replace=False)
        sample = data[idx]
    else:
        sample = data

    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(sample)
    labels = kmeans.predict(data)
    centers = [tuple(map(int, c)) for c in kmeans.cluster_centers_]
    label_grid = labels.reshape((height, width))
    return label_grid, centers


def trace_mask_to_path(binary_mask: np.ndarray, turdsize: int = 4, alphamax: float = 1.0) -> str:
    bmp = Bitmap(binary_mask)
    path_list = bmp.trace(
        turdsize=turdsize,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=alphamax,
        opticurve=True,
        opttolerance=0.2,
    )

    path_strings = []
    for curve in path_list:
        start = curve.start_point
        parts = [f"M {start.x:.2f} {start.y:.2f}"]
        for segment in curve.segments:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                parts.append(f"L {c.x:.2f} {c.y:.2f} L {end.x:.2f} {end.y:.2f}")
            else:
                c1, c2, end = segment.c1, segment.c2, segment.end_point
                parts.append(
                    f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}"
                )
        parts.append("Z")
        path_strings.append(" ".join(parts))
    return " ".join(path_strings)


def potrace_multicolor_svg(
    image: Image.Image,
    n_colors: int = 4,
    turdsize: int = 6,
    median_filter: int = 3,
    alphamax: float = 1.0,
    ignore_lightest: bool = False,
) -> str:
    """Vectorise into discrete layered colored paths using Potrace + KMeans."""
    img = image.copy()
    if median_filter and median_filter > 1:
        img = img.filter(ImageFilter.MedianFilter(size=median_filter))
    width, height = img.size

    label_grid, centers = quantize_image(img, n_colors)

    def luminance(rgb):
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    layer_indices = sorted(range(n_colors), key=lambda idx: luminance(centers[idx]))
    if ignore_lightest and len(layer_indices) > 1:
        layer_indices = layer_indices[:-1]

    svg_layers = []
    for idx in layer_indices:
        rgb = centers[idx]
        hex_color = rgb_to_hex(rgb)
        mask = label_grid == idx
        path_data = trace_mask_to_path(mask, turdsize=turdsize, alphamax=alphamax)
        if not path_data.strip():
            continue
        layer_tag = (
            f'  <g id="layer_{idx}_{hex_color.strip("#")}">\n'
            f'    <path d="{path_data}" fill="{hex_color}" fill-rule="evenodd" />\n'
            f"  </g>"
        )
        svg_layers.append(layer_tag)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        f"{chr(10).join(svg_layers)}\n"
        f"</svg>\n"
    )


# -------------------------------------------------------------------- VTracer


def vtracer_svg(
    image: Image.Image,
    color_precision: int = 6,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
    length_threshold: float = 4.0,
    max_iterations: int = 10,
    splice_threshold: int = 45,
    path_precision: int = 3,
    mode: str = "spline",
    hierarchical: str = "stacked",
    layer_difference: int = 16,
    max_colors: int | None = None,
    bw: bool = False,
) -> str:
    """Trace a full-color image into an SVG using VTracer, return SVG text."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    png_bytes = buf.getvalue()

    cfg = vtracer.Config()
    cfg.color_precision = max(1, min(8, int(color_precision)))
    cfg.filter_speckle = filter_speckle
    cfg.corner_threshold = corner_threshold
    cfg.length_threshold = length_threshold
    cfg.max_iterations = max_iterations
    cfg.splice_threshold = splice_threshold
    cfg.path_precision = path_precision
    cfg.mode = mode
    cfg.hierarchical = hierarchical
    cfg.layer_difference = layer_difference
    if bw:
        cfg.clustering = "binary"
        cfg.max_colors = 2
    elif max_colors is not None:
        cfg.max_colors = max_colors

    return cfg.convert_bytes(png_bytes)
