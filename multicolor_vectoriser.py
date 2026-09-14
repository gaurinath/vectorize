import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY
import vtracer


MAX_VTRACER_DIM = 2048
MAX_KMEANS_PIXELS = 1_000_000


def rgb_to_hex(rgb_tuple: tuple[int, int, int]) -> str:
    """Convert RGB values to a standard hexadecimal string."""
    return f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"


def quantize_image(img: Image.Image, n_colors: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Cluster image colors into n_colors discrete clusters using k-means.
    Returns the 2D label matrix and RGB centroid colors.
    """
    img_rgb = img.convert("RGB")
    width, height = img_rgb.size

    data = np.array(img_rgb).reshape((-1, 3))

    # Subsample if image is too large for fast KMeans
    total = data.shape[0]
    if total > MAX_KMEANS_PIXELS:
        rng = np.random.RandomState(42)
        idx = rng.choice(total, MAX_KMEANS_PIXELS, replace=False)
        sample = data[idx]
    else:
        sample = data

    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(sample)

    # Predict all pixels using the fitted model
    labels = kmeans.predict(data)
    centers = [tuple(map(int, c)) for c in kmeans.cluster_centers_]

    label_grid = labels.reshape((height, width))
    return label_grid, centers


def trace_mask_to_path(binary_mask: np.ndarray, turdsize: int = 4) -> str:
    """Run Potrace on a 1-bit binary mask and output an SVG path string."""
    bmp = Bitmap(binary_mask)
    path_list = bmp.trace(
        turdsize=turdsize,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1.0,
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
                parts.append(f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}")
        parts.append("Z")
        path_strings.append(" ".join(parts))

    return " ".join(path_strings)


def resize_for_vtracer(image_path: Path, tmp_dir: Path) -> Path:
    """Resize large images to a VTracer-friendly dimension, returning path to temp PNG.
    
    VTracer hangs on JPEG input, so always convert to PNG first.
    """
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) > MAX_VTRACER_DIM:
            scale = MAX_VTRACER_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out = tmp_dir / f"{image_path.stem}_vtracer_input.png"
        img.convert("RGB").save(out, "PNG")
        return out


def vectorize_multicolor_potrace(
    image_path: Path,
    output_path: Path,
    n_colors: int = 4,
    turdsize: int = 6,
    ignore_lightest: bool = False,
):
    """
    Vectorises an image into discrete, layered colored paths in an SVG using Potrace.

    Parameters:
    - ignore_lightest: If True, treats the brightest cluster (e.g. white/cream background)
      as transparent and skips exporting it.
    """
    with Image.open(image_path) as raw_img:
        img = raw_img.filter(ImageFilter.MedianFilter(size=3))
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

        mask = (label_grid == idx)

        path_data = trace_mask_to_path(mask, turdsize=turdsize)
        if not path_data.strip():
            continue

        layer_tag = f'  <g id="layer_{idx}_{hex_color.strip("#")}">\n' \
                    f'    <path d="{path_data}" fill="{hex_color}" fill-rule="evenodd" />\n' \
                    f'  </g>'
        svg_layers.append(layer_tag)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {width} {height}"
     width="{width}"
     height="{height}">
{chr(10).join(svg_layers)}
</svg>
"""
    output_path.write_text(svg_content, encoding="utf-8")
    print(f"Potrace multi-color SVG written to: {output_path}")


def vectorize_multicolor_vtracer(
    image_path: Path,
    output_path: Path,
):
    """Trace a raster image to a full-color SVG using VTracer."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = resize_for_vtracer(image_path, tmp_path)
        cfg = vtracer.Config()
        cfg.color_precision = 6
        cfg.corner_threshold = 60
        cfg.length_threshold = 4.0
        cfg.max_iterations = 10
        cfg.splice_threshold = 45
        cfg.path_precision = 3
        cfg.hierarchical = "stacked"
        cfg.mode = "spline"
        cfg.filter_speckle = 4
        vtracer.convert_file(str(src), str(output_path), cfg)
    print(f"VTracer multi-color SVG written to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run multicolor_vectoriser.py <path-to-image> [n_colors] [--no-bg]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    n_colors = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
    ignore_bg = "--no-bg" in sys.argv

    potrace_output = input_file.with_stem(f"{input_file.stem}_potrace_{n_colors}c").with_suffix(".svg")
    vtracer_output = input_file.with_stem(f"{input_file.stem}_vtracer").with_suffix(".svg")

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    vectorize_multicolor_potrace(input_file, potrace_output, n_colors=n_colors, ignore_lightest=ignore_bg)
    vectorize_multicolor_vtracer(input_file, vtracer_output)


if __name__ == "__main__":
    main()
