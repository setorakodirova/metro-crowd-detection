"""Evaluate one model and record its scores in a comparison table.

Run this once for the baseline and once for your fine-tuned model. Both rows
end up in the same table, which is what goes in your report.

    python evaluate.py --model baseline
    python evaluate.py --model weights/finetuned.pt --name finetuned

Outputs:
    results/results.json   machine-readable, one entry per model
    results/results.md     the table you paste into your report
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
RESULTS_MD = RESULTS_DIR / "results.md"

BASELINE_ALIAS = "yolov8n.pt"


def counting_error(model: YOLO, images_dir: Path, labels_dir: Path, conf: float):
    """Compare predicted person counts against ground-truth counts."""
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        return None

    abs_errors, sq_errors, compared = [], [], 0

    for path in image_paths:
        label_file = labels_dir / (path.stem + ".txt")
        if not label_file.exists():
            continue
        true_count = sum(1 for line in label_file.read_text().splitlines() if line.strip())

        result = model.predict(source=str(path), conf=conf, classes=[0], verbose=False)[0]
        predicted = len(result.boxes)

        diff = predicted - true_count
        abs_errors.append(abs(diff))
        sq_errors.append(diff ** 2)
        compared += 1

    if compared == 0:
        return None

    return {
        "images_compared": compared,
        "count_mae": round(sum(abs_errors) / compared, 3),
        "count_rmse": round(math.sqrt(sum(sq_errors) / compared), 3),
    }


def write_table(entries):
    RESULTS_DIR.mkdir(exist_ok=True)

    header = (
        "| Model | mAP@0.5 | Precision | Recall | Count MAE | Count RMSE | Images | Evaluated |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for e in entries:
        rows += (
            f"| {e['name']} | {e.get('map50', '–')} | {e.get('precision', '–')} | "
            f"{e.get('recall', '–')} | {e.get('count_mae', '–')} | {e.get('count_rmse', '–')} | "
            f"{e.get('images_compared', '–')} | {e['evaluated_at'][:10]} |\n"
        )

    RESULTS_MD.write_text(
        "# Model comparison\n\n"
        "Lower counting error is better. Higher mAP, precision and recall are better.\n\n"
        + header + rows
        + "\nAll models evaluated on the same held-out data with the same confidence threshold.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline",
                        help="'baseline' or a path to a .pt file")
    parser.add_argument("--name", default=None,
                        help="label for this row in the results table")
    parser.add_argument("--data", default="crowdhuman.yaml",
                        help="YOLO dataset yaml used for validation")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--count-images", default=None,
                        help="folder of images for the counting check")
    parser.add_argument("--count-labels", default=None,
                        help="folder of matching YOLO label files")
    args = parser.parse_args()

    weights = BASELINE_ALIAS if args.model == "baseline" else args.model
    name = args.name or ("baseline" if args.model == "baseline" else Path(weights).stem)

    print(f"Evaluating: {name}  ({weights})")
    model = YOLO(weights)

    metrics = model.val(data=args.data, conf=args.conf, verbose=False)
    entry = {
        "name": name,
        "weights": str(weights),
        "confidence": args.conf,
        "data": args.data,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if args.count_images and args.count_labels:
        counts = counting_error(model, Path(args.count_images), Path(args.count_labels), args.conf)
        if counts:
            entry.update(counts)
        else:
            print("Counting check skipped: no matching image/label pairs found.")

    RESULTS_DIR.mkdir(exist_ok=True)
    entries = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []
    entries = [e for e in entries if e["name"] != name]  # replace, don't duplicate
    entries.append(entry)
    entries.sort(key=lambda e: e["name"])
    RESULTS_JSON.write_text(json.dumps(entries, indent=2))

    write_table(entries)

    print(f"\nSaved to {RESULTS_JSON} and {RESULTS_MD}")
    print(f"  mAP@0.5   {entry['map50']}")
    print(f"  precision {entry['precision']}")
    print(f"  recall    {entry['recall']}")
    if "count_mae" in entry:
        print(f"  count MAE {entry['count_mae']}")


if __name__ == "__main__":
    main()
