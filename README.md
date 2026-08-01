# Metro platform crowd density detection

A prototype that looks at a camera view of a crowded platform, counts the people
in it, and reports whether it is Low, Moderate or Crowded.

Built for the IT Park AI/ML capstone. The model is a YOLO detector fine-tuned on
public crowd imagery. It counts people — it never identifies anyone.

## Project structure

```
metro-crowd-detection/
├── README.md
├── .gitignore
├── backend/
│   ├── app.py              the API — receives images, returns results
│   ├── detector.py         runs the model, counts people, works out density
│   ├── train.py            fine-tunes YOLO (never overwrites the baseline)
│   ├── evaluate.py         measures a model, writes the comparison table
│   ├── crowdhuman.yaml     tells YOLO where the dataset is
│   ├── requirements.txt
│   ├── weights/            trained models (not in git)
│   └── results/            comparison table (committed to git)
├── frontend/
│   └── index.html          the interface
├── data/                   the dataset (not in git — download it yourself)
├── notebooks/
│   └── 01_data_check.ipynb check the data before training
└── runs/                   created automatically by YOLO during training
```

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Order of work

### 1. Run the demo on baseline weights

Confirms the setup works before touching any data.

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd frontend
python -m http.server 3000
```

Open http://localhost:3000 and drop in any crowd photo. Green boxes should
appear. The baseline weights download automatically on first run.

### 2. Get the data

See `data/README.md`. Download CrowdHuman and convert it to YOLO format so it
matches the folder layout described there.

### 3. Check the data

Open `notebooks/01_data_check.ipynb` and run it. It verifies every image has a
label file, that box values are valid, and shows sample images with their boxes
drawn on. Fix any problem here before training — it saves hours.

### 4. Evaluate the baseline first

Do this **before** training. It is your "before" measurement.

```bash
cd backend
python evaluate.py --model baseline --name baseline \
  --data crowdhuman.yaml \
  --count-images ../data/val/images --count-labels ../data/val/labels
```

### 5. Train

```bash
python train.py --data crowdhuman.yaml --epochs 30 --run-name run1
```

The script saves to `weights/run1.pt`. It refuses to overwrite baseline weights
or an existing run, so your comparison stays intact.

### 6. Evaluate your model

```bash
python evaluate.py --model weights/run1.pt --name run1 \
  --data crowdhuman.yaml \
  --count-images ../data/val/images --count-labels ../data/val/labels
```

Both models now appear in `backend/results/results.md`. That table is the core
evidence for your report.

### 7. Use your model in the demo

```bash
cp weights/run1.pt weights/finetuned.pt
```

The dropdown in the interface now compares baseline against your model.

## Using a phone camera

Browsers only allow camera access over https or on localhost. Opening the page
by local IP address on a phone will fail silently — this is a browser rule, not
a bug. Expose the frontend through a tunnel and open the https link it gives you:

```bash
ngrok http 3000
```

Expose the backend too, and update the `API` constant near the top of the script
in `frontend/index.html` to point at that https address.

## Density thresholds

Set in `backend/detector.py` as `DENSITY_THRESHOLDS`. They are provisional and
not calibrated against real platform area or official capacity figures. State
this limitation in the report.

## Privacy

The system counts people. No face recognition, no identification, no tracking of
individuals. Frames are processed in memory and not stored.

## Data and licensing

- CrowdHuman (Shao et al., 2018) — academic research use; cite the paper.
- Mall dataset (Loy et al.) — research only, non-commercial; used for testing.
- Ultralytics YOLO — AGPL-3.0. Fine for a public academic repository; a
  commercial product would need a paid licence.

## Limitations

The model is trained on public crowd photographs, not real metro footage. Real
station cameras are mounted high and look down, so performance there is
unverified. The model undercounts in very dense crowds, which is exactly where
accuracy matters most. Both limitations are stated in the project brief.
