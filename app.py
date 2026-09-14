"""Streamlit web UI for raster-to-vector conversion.

Run with:  uv run streamlit run app.py
"""

import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image

import core

st.set_page_config(page_title="Raster → Vector Studio", page_icon="🎯", layout="wide")

MODE_HELP = {
    "spline": "Smooth Bézier curves (best for photos & organic shapes)",
    "polygon": "Straight-edged polygons (best for technical drawings)",
    "pixel": "Pixel-art style square blocks",
}


def svg_data_uri(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def render_svg(svg: str):
    uri = svg_data_uri(svg)
    st.markdown(
        f'<div style="background:repeat; '
        f'background-image:linear-gradient(45deg,#eee 25%,transparent 25%),'
        f'linear-gradient(-45deg,#eee 25%,transparent 25%),'
        f'linear-gradient(45deg,transparent 75%,#eee 75%),'
        f'linear-gradient(-45deg,transparent 75%,#eee 75%);'
        f'background-size:20px 20px;padding:8px;border-radius:8px;">'
        f'<img src="{uri}" style="width:100%;max-width:800px;display:block;margin:auto;"/></div>',
        unsafe_allow_html=True,
    )


def download_svg_button(svg: str, filename: str, label: str = "⬇ Download SVG"):
    st.download_button(
        label,
        data=svg.encode("utf-8"),
        file_name=filename,
        mime="image/svg+xml",
    )


def load_image(uploaded) -> Image.Image | None:
    data = uploaded.getvalue()
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


st.title("🎯 Raster → Vector Studio")
st.caption("Convert raster images (JPG/PNG/WebP/…) into SVG vectors with **Potrace** or **VTracer**. "
           "Use the sliders to tune sensitivity so lighter shades are captured too.")

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("1. Input image")
uploaded = st.sidebar.file_uploader(
    "Upload a raster image",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
)
source_path = None
if uploaded is None:
    import pathlib
    samples = sorted(str(p) for p in pathlib.Path(".").glob("*.jpg"))
    samples += sorted(str(p) for p in pathlib.Path(".").glob("*.png"))
    samples += sorted(str(p) for p in pathlib.Path(".").glob("*.jpeg"))
    if samples:
        chosen = st.sidebar.selectbox("…or pick a sample image:", samples)
        source_path = chosen
        st.sidebar.caption(f"Using sample: {chosen}")

image: Image.Image | None = None
base_name = "image"
if uploaded is not None:
    image = load_image(uploaded)
    base_name = uploaded.name.rsplit(".", 1)[0]
    if image is None:
        st.error("Could not read that image file.")
elif source_path:
    image = Image.open(source_path)
    base_name = Path(source_path).stem

if image is None:
    st.info("👈 Upload an image or choose a sample to begin.")
    st.stop()

st.sidebar.header("2. Processing size")
max_dim = st.sidebar.slider("Max dimension (px)", 256, 4096, 1600, step=128,
                            help="Large images are downscaled to this size before tracing for speed.")
orig_w, orig_h = image.size
if max(orig_w, orig_h) > max_dim:
    scale = max_dim / max(orig_w, orig_h)
    image = image.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
    st.sidebar.caption(f"Resized {orig_w}×{orig_h} → {image.size[0]}×{image.size[1]}")

# ------------------------------------------------------------------ Layout
col_source, col_out = st.columns([1, 1.4], gap="medium")

with col_source:
    st.subheader("Original")
    st.image(image, width="stretch")
    st.caption(f"{image.size[0]} × {image.size[1]} px · {image.mode}")

with col_out:
    tab_bw, tab_mc, tab_vt = st.tabs(
        ["Potrace — B&W", "Potrace — Multi-color", "VTracer — Color"]
    )

    with tab_bw:
        st.subheader("Potrace (binary / B&W)")
        c1, c2 = st.columns(2)
        threshold = c1.slider("Threshold", 0, 255, 180,
                              help="Pixel luminance below this → foreground. Raise it to capture "
                                   "LIGHTER shades as vectors; lower it to keep only dark outlines.")
        blur = c2.slider("Gaussian blur", 0.0, 4.0, 0.5, 0.1,
                         help="Pre-filter to reduce raster noise before binarizing.")
        turdsize = st.slider("Turdsize (speckle removal)", 0, 20, 4,
                             help="Suppress tiny noisy blobs smaller than this many pixels.")
        alphamax = st.slider("Corner threshold (alphamax)", 0.0, 1.34, 1.0, 0.01,
                             help="0.0 = sharp corners, 1.34 = max rounding.")
        opttolerance = st.slider("Optimization tolerance", 0.0, 2.0, 0.2, 0.05,
                                 help="Larger → fewer anchor points, smoother but less faithful.")
        invert = st.checkbox("Invert (trace LIGHT pixels instead of dark)", value=False)
        fill = st.color_picker("Vector fill color", "#000000")

        if st.button("▶ Vectorize (B&W)", type="primary"):
            with st.spinner("Tracing with Potrace…"):
                svg = core.potrace_bw_svg(
                    image, threshold=threshold, blur=blur, turdsize=turdsize,
                    alphamax=alphamax, opttolerance=opttolerance, invert=invert, fill=fill,
                )
            render_svg(svg)
            download_svg_button(svg, f"{base_name}_potrace_bw.svg")
            st.caption(f"SVG: {len(svg) / 1024:.1f} KB")

    with tab_mc:
        st.subheader("Potrace (KMeans color quantization)")
        c1, c2 = st.columns(2)
        n_colors = c1.slider("Number of colors", 2, 16, 4,
                             help="Cluster count fed to KMeans. More colors → more shades captured.")
        median = c2.slider("Median filter size", 1, 7, 3, step=2,
                           help="Removes grain before quantization. 1 = off.")
        turdsize = st.slider("Turdsize (speckle removal)", 0, 20, 6)
        alphamax = st.slider("Corner threshold (alphamax)", 0.0, 1.34, 1.0, 0.01)
        ignore_bg = st.checkbox("Skip the lightest (background) cluster", value=False)

        if st.button("▶ Vectorize (Multi-color)", type="primary"):
            with st.spinner("Clustering colors & tracing layers… this can take a while."):
                svg = core.potrace_multicolor_svg(
                    image, n_colors=n_colors, turdsize=turdsize,
                    median_filter=median, alphamax=alphamax, ignore_lightest=ignore_bg,
                )
            render_svg(svg)
            download_svg_button(svg, "image_potrace_multicolor.svg")
            st.caption(f"SVG: {len(svg) / 1024:.1f} KB")

    with tab_vt:
        st.subheader("VTracer (full color, Rust engine)")
        c1, c2, c3 = st.columns(3)
        color_precision = c1.slider("Color precision (bits/channel)", 1, 8, 6,
                                    help="Higher → finer color buckets, so subtle & LIGHTER shades are "
                                         "kept as separate vectors. Lower → fewer, flatter colors.")
        filter_speckle = c2.slider("Filter speckle", 0, 20, 4,
                                   help="Removes tiny blobs; 0 = keep everything.")
        layer_difference = c3.slider("Layer difference", 1, 100, 16,
                                     help="Min color distance between stacked layers.")
        c1, c2, c3 = st.columns(3)
        corner_threshold = c1.slider("Corner threshold", 0, 180, 60,
                                     help="0 = all rounded, 180 = keep all corners.")
        length_threshold = c2.slider("Length threshold", 0.0, 20.0, 4.0, 0.5,
                                     help="Minimum segment length retained.")
        splice_threshold = c3.slider("Splice threshold", 0, 90, 45,
                                     help="Curve-splicing aggressiveness.")
        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Curve mode", ["spline", "polygon", "pixel"])
        mode_help = MODE_HELP.get(mode, "")
        if mode_help:
            st.caption(mode_help)
        hierarchical = c2.selectbox("Hierarchical", ["stacked", "cutout"],
                                    help="stacked = no holes, compact; cutout = boolean cut-out shapes.")
        max_iterations = c3.slider("Max iterations", 1, 20, 10)
        max_colors = st.slider("Max colors (0 = unlimited)", 0, 256, 0,
                               help="Cap palette size. 0 = no limit.")
        bw = st.checkbox("Black & white output", value=False,
                         help="Force VTracer into binary (B&W) mode instead of color.")

        if st.button("▶ Vectorize (VTracer)", type="primary"):
            with st.spinner("Tracing with VTracer…"):
                svg = core.vtracer_svg(
                    image,
                    color_precision=color_precision,
                    filter_speckle=filter_speckle,
                    corner_threshold=corner_threshold,
                    length_threshold=length_threshold,
                    max_iterations=max_iterations,
                    splice_threshold=splice_threshold,
                    mode=mode,
                    hierarchical=hierarchical,
                    layer_difference=layer_difference,
                    max_colors=max_colors or None,
                    bw=bw,
                )
            render_svg(svg)
            download_svg_button(svg, "image_vtracer.svg")
            st.caption(f"SVG: {len(svg) / 1024:.1f} KB")
