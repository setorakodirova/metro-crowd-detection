# weights

Trained model files go here. They are not committed to git — they are too large.

- `run1.pt` and similar appear after you run `train.py --run-name run1`
- `finetuned.pt` is the file the demo loads. Copy your best run to this name:

```bash
cp weights/run1.pt weights/finetuned.pt
```

If `finetuned.pt` does not exist, the demo falls back to the baseline model, so
both dropdown options will give identical results until you train something.

The baseline file `yolov8n.pt` downloads automatically on first run and must
never be overwritten — it is your comparison point.
