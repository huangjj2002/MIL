import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_CSV = Path(r"G:\data\meta_data.csv")
DEFAULT_OUTPUT_ROOT = Path(r"G:\data")
DEFAULT_OUTPUT_CSV = "meta_data_mil.csv"
DEFAULT_IMG_DIR = "images_png"

REQUIRED_COLUMNS = {
    "image_path",
    "empi_anon",
    "laterality_label",
    "cohort_num",
    "split",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert meta_data.csv into the MIL input CSV format without changing "
            "the project dataloader logic."
        )
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Source meta_data.csv path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output data root.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV filename.")
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR, help="Image directory under output-root.")
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Only write the MIL CSV; do not copy image files.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite destination images when copying.",
    )
    parser.add_argument(
        "--continue-on-copy-error",
        action="store_true",
        help="Write the output CSV even if some image copies fail.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print stats without writing files or copying images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N input rows for quick testing. 0 means all rows.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print copy progress every N images.",
    )
    return parser.parse_args()


def clean_text(value):
    return str(value).strip()


def basename_from_path(path_text):
    text = clean_text(path_text).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def image_id_from_row(row):
    image_id = basename_from_path(row.get("image_path", ""))
    if image_id:
        return image_id

    dicom_name = basename_from_path(row.get("anon_dicom_path", ""))
    if dicom_name.lower().endswith(".dcm"):
        return dicom_name[:-4] + ".png"
    return dicom_name


def parse_binary_label(value, row_number):
    text = clean_text(value)
    if text in {"0", "1"}:
        return text

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: laterality_label must be 0 or 1, got {value!r}."
        ) from exc

    if number in (0.0, 1.0):
        return str(int(number))

    raise ValueError(f"Row {row_number}: laterality_label must be 0 or 1, got {value!r}.")


def parse_cohort(value, row_number):
    text = clean_text(value)
    try:
        return str(int(float(text)))
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: cohort_num must be numeric, got {value!r}.") from exc


def build_record(row, row_number):
    patient_id = clean_text(row.get("empi_anon", ""))
    image_id = image_id_from_row(row)
    cancer = parse_binary_label(row.get("laterality_label", ""), row_number)
    cohort_num = parse_cohort(row.get("cohort_num", ""), row_number)
    split = clean_text(row.get("split", ""))
    source_image_path = clean_text(row.get("image_path", ""))

    missing = []
    if not patient_id:
        missing.append("empi_anon")
    if not image_id:
        missing.append("image_id")
    if not split:
        missing.append("split")
    if not source_image_path:
        missing.append("image_path")
    if missing:
        raise ValueError(f"Row {row_number}: missing required values: {missing}")

    return {
        "patient_id": patient_id,
        "image_id": image_id,
        "split": split,
        "cancer": cancer,
        "cohort_num": cohort_num,
        "source_image_path": source_image_path,
        "source_row": row_number,
    }


def load_and_deduplicate(input_csv, limit):
    records_by_image_path = {}
    conflicts = []
    input_rows = 0

    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {missing_columns}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            input_rows += 1
            if limit and input_rows > limit:
                input_rows -= 1
                break

            record = build_record(row, row_number)
            dedup_key = record["source_image_path"]
            previous = records_by_image_path.get(dedup_key)
            if previous is None:
                records_by_image_path[dedup_key] = record
                continue

            comparable_keys = ("patient_id", "image_id", "split", "cancer", "cohort_num")
            if any(previous[key] != record[key] for key in comparable_keys):
                conflicts.append(
                    {
                        "image_path": dedup_key,
                        "first_source_row": previous["source_row"],
                        "conflict_source_row": record["source_row"],
                        "first_values": repr({key: previous[key] for key in comparable_keys}),
                        "conflict_values": repr({key: record[key] for key in comparable_keys}),
                    }
                )

    return input_rows, list(records_by_image_path.values()), conflicts


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_images(records, output_root, img_dir, overwrite, progress_every):
    copied = 0
    skipped = 0
    failed = []

    for index, record in enumerate(records, start=1):
        source = Path(record["source_image_path"])
        dest = output_root / img_dir / record["patient_id"] / record["image_id"]

        if dest.exists() and not overwrite:
            skipped += 1
        else:
            try:
                if not source.exists():
                    raise FileNotFoundError(str(source))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                copied += 1
            except Exception as exc:
                failed.append(
                    {
                        "patient_id": record["patient_id"],
                        "image_id": record["image_id"],
                        "source_image_path": record["source_image_path"],
                        "dest_image_path": str(dest),
                        "error": repr(exc),
                    }
                )

        if progress_every > 0 and index % progress_every == 0:
            print(
                f"copy progress: {index}/{len(records)} "
                f"(copied={copied}, skipped={skipped}, failed={len(failed)})"
            )

    return copied, skipped, failed


def print_stats(input_rows, records):
    cancer_counts = Counter(record["cancer"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    cohort_counts = Counter(record["cohort_num"] for record in records)

    print(f"input rows: {input_rows}")
    print(f"output rows after image_path dedup: {len(records)}")
    print(f"deduplicated rows: {input_rows - len(records)}")
    print(f"cancer counts: {dict(sorted(cancer_counts.items()))}")
    print(f"split counts: {dict(sorted(split_counts.items()))}")
    print(
        "cohort counts: "
        f"{dict(sorted(cohort_counts.items(), key=lambda item: int(item[0])))}"
    )


def main():
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_csv = output_root / args.output_csv
    conflict_csv = output_root / "dedup_conflicts.csv"
    failure_csv = output_root / "copy_failures.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"input csv: {input_csv}")
    print(f"output root: {output_root}")
    print(f"output csv: {output_csv}")

    input_rows, records, conflicts = load_and_deduplicate(input_csv, args.limit)
    print_stats(input_rows, records)

    if conflicts:
        print(f"dedup conflicts: {len(conflicts)}")
        if not args.dry_run:
            write_csv(
                conflict_csv,
                conflicts,
                ["image_path", "first_source_row", "conflict_source_row", "first_values", "conflict_values"],
            )
            print(f"wrote conflict report: {conflict_csv}")
        return 2

    mil_rows = [
        {
            "patient_id": record["patient_id"],
            "image_id": record["image_id"],
            "split": record["split"],
            "cancer": record["cancer"],
            "cohort_num": record["cohort_num"],
        }
        for record in records
    ]

    if args.dry_run:
        print("dry run: no files written")
        return 0

    if args.no_copy_images:
        print("image copy disabled by --no-copy-images")
    else:
        copied, skipped, copy_failures = copy_images(
            records,
            output_root,
            args.img_dir,
            args.overwrite_images,
            args.progress_every,
        )
        print(f"images copied: {copied}, skipped: {skipped}, failed: {len(copy_failures)}")
        if copy_failures:
            write_csv(
                failure_csv,
                copy_failures,
                ["patient_id", "image_id", "source_image_path", "dest_image_path", "error"],
            )
            print(f"wrote copy failure report: {failure_csv}")
            if not args.continue_on_copy_error:
                return 3

    write_csv(output_csv, mil_rows, ["patient_id", "image_id", "split", "cancer", "cohort_num"])
    print(f"wrote MIL CSV: {output_csv}")
    print("\nUse with:")
    print(f"  --data_dir {output_root}")
    print(f"  --img_dir {args.img_dir}")
    print(f"  --csv_file {args.output_csv}")
    print("  --label cancer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
