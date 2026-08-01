"""Fine-tune YOLO for dense person detection.

Saves the trained model to a NEW file. It will refuse to write over the
baseline weights, so your comparison stays valid.

    python train.py --data crowdhuman.yaml --epochs 30 --run-name run1
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PROTECTED_NAMES = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
WEIGHTS_DIR = Path(__file__).parent / "weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="crowdhuman.yaml")
    parser.add_argument("--base", default="yolov8n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--run-name", default="run1",
                        help="name for this experiment; also names the output file")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    output = WEIGHTS_DIR / f"{args.run_name}.pt"

    if output.name in PROTECTED_NAMES:
        raise SystemExit(
            f"Refusing to write '{output.name}' — that is a baseline file name. "
            "Choose a different --run-name."
        )
    if output.exists():
        raise SystemExit(
            f"'{output}' already exists. Choose a different --run-name so you do not "
            "lose a previous experiment."
        )

    print(f"Starting from : {args.base}")
    print(f"Will save to  : {output}")
    print(f"Baseline file '{args.base}' will not be modified.\n")

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.run_name,
        project="runs",
        exist_ok=False,
    )

    best = Path("runs") / args.run_name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but {best} was not found.")

    shutil.copy(best, output)
    print(f"\nSaved trained model to {output}")
    print("\nNext steps:")
    print(f"  python evaluate.py --model baseline --name baseline")
    print(f"  python evaluate.py --model {output} --name {args.run_name}")
    print(f"  copy {output} to weights/finetuned.pt to use it in the demo")


if __name__ == "__main__":
    main()
