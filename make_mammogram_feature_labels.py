from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_DATA_DIR = Path(
    r"G:\614\data\dst_proto_dedup_pixel_only_dcm_png_review_2026-06-14"
    r"\data_png_no_crop_with_prototype\Mammogram_png\data"
)

DEFAULT_IMAGES = ["C1-p07", "TP_02", "C1-p08", "TP_06"]

FIELD_ORDER = [
    ("massshape", "Mass shape", "shape"),
    ("massmargin", "Mass margin", "margin"),
    ("massdens", "Mass density", "density"),
    ("calcfind", "Calcification morphology", "calc"),
]

VALUE_MAPS = {
    "massshape": {
        "F": "Focal asymmetry",
        "X": "Irregular",
    },
    "massmargin": {
        "U": "Obscured",
    },
    "massdens": {
        "=": "Isodense",
    },
    "calcfind": {
        "H": "Coarse heterogeneous",
    },
}

COLORS = {
    "shape": ((17, 93, 116, 238), (112, 190, 207, 255)),
    "margin": ((143, 70, 31, 238), (230, 152, 91, 255)),
    "density": ((74, 72, 142, 238), (171, 167, 231, 255)),
    "calc": ((38, 112, 78, 238), (128, 211, 160, 255)),
}

TEXT = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 140)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mammogram PNGs with large color-coded feature labels from "
            "MagView-style CSV fields."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing *_no_label.png and data_with_magview_fields.csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path. Defaults to <data-dir>/data_with_magview_fields.csv.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=DEFAULT_IMAGES,
        help="Image stems to process, for example C1-p07 TP_02 C1-p08 TP_06.",
    )
    parser.add_argument(
        "--input-suffix",
        default="_no_label.png",
        help="Input filename suffix appended to each image stem.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_label_only.png",
        help="Output filename suffix appended to each image stem.",
    )
    parser.add_argument(
        "--preview",
        default="clinical_label_only_preview.png",
        help="Preview contact-sheet filename saved under data-dir.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            image_name = row.get("image_name", "").strip()
            if image_name:
                rows[Path(image_name).stem] = row
        return rows


def get_font_path() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No suitable Windows font found.")


def estimate_left_tissue_x(im: Image.Image) -> int:
    """Estimate where breast tissue begins so labels stay in the left black field."""
    gray = im.convert("L")
    threshold = 24
    try:
        import numpy as np

        arr = np.array(gray)
        col_counts = (arr > threshold).sum(axis=0)
        min_pixels = max(80, int(gray.height * 0.025))
        xs = np.where(col_counts > min_pixels)[0]
        return int(xs[0]) if len(xs) else gray.width
    except Exception:
        # Fallback without numpy: use a downsampled image to keep loops small.
        scale = 4
        small = gray.resize((max(1, gray.width // scale), max(1, gray.height // scale)))
        pixels = small.load()
        min_pixels = max(20, int(small.height * 0.025))
        for x in range(small.width):
            count = 0
            for y in range(small.height):
                if pixels[x, y] > threshold:
                    count += 1
            if count > min_pixels:
                return x * scale
        return gray.width


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_font_and_wrap(
    draw: ImageDraw.ImageDraw,
    raw_lines: list[str],
    font_path: Path,
    font_size: int,
    max_width: int,
) -> tuple[ImageFont.FreeTypeFont, int, list[list[str]]]:
    while font_size >= 68:
        font = ImageFont.truetype(str(font_path), font_size)
        wrapped = [wrap_line(draw, line, font, max_width) for line in raw_lines]
        if all(len(lines) <= 3 for lines in wrapped):
            return font, font_size, wrapped
        font_size = int(font_size * 0.94)

    font = ImageFont.truetype(str(font_path), font_size)
    wrapped = [wrap_line(draw, line, font, max_width) for line in raw_lines]
    return font, font_size, wrapped


def label_specs_from_row(row: dict[str, str]) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for field, display_name, color_key in FIELD_ORDER:
        raw_value = row.get(field, "").strip()
        if not raw_value:
            continue
        display_value = VALUE_MAPS.get(field, {}).get(raw_value, raw_value)
        specs.append((display_name, display_value, color_key))
    return specs


def draw_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    bg: tuple[int, int, int, int],
    border: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
    line_gap: int,
    border_width: int,
) -> tuple[int, int]:
    widths = [text_size(draw, line, font)[0] for line in lines]
    heights = [text_size(draw, line, font)[1] for line in lines]
    text_w = max(widths)
    text_h = sum(heights) + line_gap * (len(lines) - 1)
    x1 = x + text_w + 2 * pad_x
    y1 = y + text_h + 2 * pad_y

    draw.rectangle([x, y, x1, y1], fill=bg, outline=border, width=border_width)
    ty = y + pad_y
    for line, line_h in zip(lines, heights):
        draw.text((x + pad_x + 4, ty + 4), line, font=font, fill=SHADOW)
        draw.text((x + pad_x, ty), line, font=font, fill=TEXT)
        ty += line_h + line_gap
    return x1, y1


def draw_feature_labels(
    im: Image.Image,
    specs: list[tuple[str, str, str]],
    font_path: Path,
) -> tuple[Image.Image, int, int]:
    base_im = im.convert("RGB")
    tissue_x = estimate_left_tissue_x(base_im)
    im_rgba = base_im.convert("RGBA")
    w, _ = im_rgba.size
    overlay = Image.new("RGBA", im_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    raw_lines = [f"{name}: {value}" for name, value, _ in specs]
    margin_x = max(34, int(w * 0.018))
    margin_y = max(34, int(w * 0.016))
    pad_x = max(34, int(w * 0.019))
    pad_y = max(24, int(w * 0.013))
    box_gap = max(16, int(w * 0.009))
    border_width = max(4, int(w * 0.0022))
    safety_gap = max(80, int(w * 0.045))

    max_box_right = max(int(w * 0.42), tissue_x - safety_gap)
    max_text_width = max(360, max_box_right - margin_x - 2 * pad_x)
    font_size = max(102, int(w * 0.065))
    font, font_size, wrapped_lines = fit_font_and_wrap(
        draw, raw_lines, font_path, font_size, max_text_width
    )
    line_gap = max(7, int(font_size * 0.10))

    x, y = margin_x, margin_y
    actual_right = x
    for spec, lines in zip(specs, wrapped_lines):
        _, _, color_key = spec
        bg, border = COLORS[color_key]
        x1, y = draw_box(
            draw, x, y, lines, font, bg, border, pad_x, pad_y, line_gap, border_width
        )
        actual_right = max(actual_right, x1)
        y += box_gap

    return Image.alpha_composite(im_rgba, overlay).convert("RGB"), tissue_x, actual_right


def build_preview(
    data_dir: Path,
    image_stems: list[str],
    output_suffix: str,
    preview_name: str,
    font_path: Path,
) -> Path:
    thumb_w, thumb_h = 500, 600
    caption_font = ImageFont.truetype(str(font_path), 44)
    canvas_w = thumb_w * 2 + 180
    rows = (len(image_stems) + 1) // 2
    canvas_h = rows * (thumb_h + 140) + 110
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, stem in enumerate(image_stems):
        row, col = divmod(idx, 2)
        x0 = 60 + col * (thumb_w + 60)
        y0 = 40 + row * (thumb_h + 140)
        img = Image.open(data_dir / f"{stem}{output_suffix}").convert("RGB")
        thumb = ImageOps.contain(img, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        x = x0 + (thumb_w - thumb.width) // 2
        y = y0 + (thumb_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        tw = text_size(draw, stem, caption_font)[0]
        draw.text((x0 + (thumb_w - tw) // 2, y0 + thumb_h + 18), stem, fill=(0, 0, 0), font=caption_font)

    preview_path = data_dir / preview_name
    canvas.save(preview_path)
    return preview_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    csv_path = args.csv or data_dir / "data_with_magview_fields.csv"
    rows = load_rows(csv_path)
    font_path = get_font_path()

    for stem in args.images:
        if stem not in rows:
            raise KeyError(f"{stem!r} not found in {csv_path}")
        src = data_dir / f"{stem}{args.input_suffix}"
        if not src.exists():
            raise FileNotFoundError(src)

        specs = label_specs_from_row(rows[stem])
        out, tissue_x, label_right = draw_feature_labels(Image.open(src), specs, font_path)
        out_path = data_dir / f"{stem}{args.output_suffix}"
        out.save(out_path)
        print(f"{out_path} | tissue_x~{tissue_x} label_right~{int(label_right)}")

    preview_path = build_preview(
        data_dir, args.images, args.output_suffix, args.preview, font_path
    )
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
