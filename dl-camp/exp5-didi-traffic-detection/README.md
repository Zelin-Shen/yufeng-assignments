# Exp 5: Didi Traffic Scene Object Detection

Two-stage object detection on the Didi GAIA traffic-scene dataset using MMDetection.

- Original assignment: 滴滴出行——交通场景目标检测
- Key techniques: Faster R-CNN (baseline), Cascade R-CNN (improved), config-level tuning
- Artifacts: `report.pdf`, `configs/` (default + 4 tuning iterations)

## MMDetection Environment

The config files inherit from the standard MMDetection `_base_` configs (e.g. `../_base_/schedules/schedule_1x.py`, `../_base_/datasets/coco_detection.py`) and therefore must be placed inside an MMDetection project tree to run:

```
mmdetection/configs/didi/<your_config>.py
```

Adjust `data_root` to your local copy of the Didi GAIA dataset. Results in the report were produced with MMDetection on CUDA 11 + PyTorch 1.x.
