import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY
import vtracer


MAX_VTRACER_DIM = 2048


def preprocess_image(image_path: Path, threshold: int = 180) -> np.ndarray:
    """Load and binarize the input image for clean edge tracing."""
    with Image.open(image_path) as img:
        gray = img.convert("L").filter(ImageFilter.GaussianBlur(radius=0.5))
        arr = np.array(gray)
        binary = (arr < threshold)
        return binary


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


def trace_to_svg_potrace(binary_data: np.ndarray, output_path: Path, turdsize: int = 4):
    """Trace bitmap into an SVG with Bezier curves using Potrace."""
    height, width = binary_data.shape

    bmp = Bitmap(binary_data)
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
                c1 = segment.c1
                c2 = segment.c2
                end = segment.end_point
                parts.append(f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}")

        parts.append("Z")
        path_strings.append(" ".join(parts))

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {width} {height}"
     width="{width}"
     height="{height}">
  <path d="{' '.join(path_strings)}" fill="#000000" fill-rule="evenodd" />
</svg>
"""
    output_path.write_text(svg_content, encoding="utf-8")
    print(f"Potrace SVG written to: {output_path}")


def trace_to_svg_vtracer(image_path: Path, output_path: Path):
    """Trace a raster image to SVG using VTracer (Rust-based, full color)."""
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
    print(f"VTracer SVG written to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run vectoriser.py <path-to-image> [threshold]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    thresh = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    potrace_output = input_file.with_stem(f"{input_file.stem}_potrace").with_suffix(".svg")
    vtracer_output = input_file.with_stem(f"{input_file.stem}_vtracer").with_suffix(".svg")

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    binary_map = preprocess_image(input_file, threshold=thresh)
    trace_to_svg_potrace(binary_map, potrace_output)
    trace_to_svg_vtracer(input_file, vtracer_output)


if __name__ == "__main__":
    main()
