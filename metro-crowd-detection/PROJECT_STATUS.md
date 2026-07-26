# Project Status

## Project

Tashkent Metro Crowd Density Detection

## Current stage

Modeling (baseline training complete)

## Completed

- Dataset prepared (CrowdHuman head-box annotations)
- YOLOv8n fine-tuned at imgsz 960
- Baseline MAE: 10.93 → Run1 MAE: 5.30
- Recall: 0.649, mAP@0.5: 0.621
- Frontend prototype built

## Current task

Evaluate retraining at imgsz 1280 for improved accuracy

## Next

- C3 Data Audit
- Error analysis (report section 6)
- Cross-domain test on Mall dataset

## Known problems / blockers

- Trained model weights stored in Google Drive (too large for Git)
- No public station-level ridership data available
