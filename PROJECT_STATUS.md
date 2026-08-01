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

## What I did (steps)

1. Downloaded CrowdHuman dataset — it has labeled human heads in crowded scenes
2. Set up YOLOv8n model and wrote the config file for training
3. Ran first training — model gave MAE 10.93 and mAP 0.0 (completely wrong detections)
4. Found the problem — labels were for full bodies but I needed heads only. Fixed it
5. Retrained at image size 960 — MAE dropped to 5.30, much better
6. Got recall 0.649 and mAP@0.5 0.621 — model now detects heads well
7. Built a simple web page to show crowd density results visually
8. Pushed code to GitHub with .gitignore so large files stay out

## Reflection

- The hardest part was figuring out why mAP was 0.0 — turned out the labels didn't match what the model was looking for. Fixing data helped more than changing the model
- I learned that clean data matters more than a fancy model
- Next I want to try training at image size 1280 to see if bigger images give better results in dense crowds

## Known problems / blockers

- Trained model weights stored in Google Drive (too large for Git)
- No public station-level ridership data available
