import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def resize_long_side(img: Image.Image, long_side: int = 1080) -> Image.Image:
    w, h = img.size
    if max(w, h) == long_side:
        return img
    if w >= h:
        new_w = long_side
        new_h = int(round(h * (long_side / w)))
    else:
        new_h = long_side
        new_w = int(round(w * (long_side / h)))
    return img.resize((new_w, new_h), Image.BICUBIC)

def main(png_root: str, out_root: str, long_side: int = 1080):
    png_root = Path(png_root)
    out_root = Path(out_root)

    png_paths = list(png_root.rglob("*.png"))
    if len(png_paths) == 0:
        raise RuntimeError(f"No png found under: {png_root}")

    for p in tqdm(png_paths, desc="Converting PNG -> JPG"):
        # p: <png_root>/<patient_id>/<image_id>.png
        patient_id = p.parent.name
        image_id = p.stem

        out_dir = out_root / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_id}_resized.jpg"

        try:
            img = Image.open(p).convert("RGB")
            img = resize_long_side(img, long_side=long_side)
            img.save(out_path, quality=95)
        except Exception as e:
            print(f"[FAIL] {p} -> {out_path}: {e}")

if __name__ == "__main__":
    PNG_ROOT = "/mnt/f/MammoCLIP/train_images_png" 
    OUT_ROOT = "/mnt/f/MammoCLIP/train_images_jpg"
    main(PNG_ROOT, OUT_ROOT, long_side=1080)
