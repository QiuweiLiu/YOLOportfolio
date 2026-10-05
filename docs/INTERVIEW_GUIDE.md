# Interview Guide — YOLO Model Optimization Portfolio

This document turns the repository into an interview-ready case study. It separates what was actually implemented and measured from follow-up ideas that were not part of the reported experiment.

## 30-second version

I built a reproducible YOLO optimization workflow on the TACO litter dataset. Instead of tuning the model blindly, I first audited the 60-class long-tail label distribution, froze an 8-class task and a 640/80/80 train/val/test split, then used FP/FN and object-size analysis to identify the main bottleneck. On the validation set, 75.5% of objects were tiny, so I tested the hypothesis that a larger input resolution would help. Keeping the task and evaluation split fixed, moving YOLOv8n from 416px to 640px improved mAP50 from 0.238 to 0.315, about +32%, and recall from 0.266 to 0.346. The main value of the project is the diagnosis-and-validation workflow, not just the final score.

## 1-minute version

The starting point was TACO, which has 60 litter categories but a very long-tailed class distribution. I first ran a scope study and found that many classes had too few examples to support a meaningful small portfolio experiment, so I froze a focused 8-class task and a fixed 640/80/80 split.

Then I trained a YOLOv8n baseline at 416x416 and evaluated it with a standardized script. The baseline mAP50 was 0.238 and recall was 0.266. I did not immediately change the model architecture. Instead, I analyzed false positives, false negatives and object sizes. The key finding was that 154 of 204 validation objects, or 75.5%, occupied less than 1% of the image area, and the cigarette class had especially poor recall.

That led to a concrete hypothesis: the model was resolution-limited on tiny objects. I increased the input size from 416 to 640 while keeping the task definition and frozen evaluation split unchanged. mAP50 increased to 0.315, mAP50-95 to 0.233, and recall to 0.346. The project demonstrates how I turn an error analysis into a controlled experiment and an auditable before/after result.

## 3-minute story

### 1. Problem

The original dataset is not a clean balanced detection benchmark. TACO contains 60 fine-grained litter classes and the label distribution is highly skewed. A naive 60-class training run therefore mixes two problems: model quality and task-definition quality.

### 2. Make the task measurable

I audited the class distribution first and compared 60-class, 23-class and 8-class scopes. Those scores are only a scope study and are not treated as direct optimization comparisons. I selected the 8 most common classes and froze the task definition and train/validation/test manifests.

The final frozen split contains 640 training images, 80 validation images and 80 test images.

### 3. Establish a baseline

The baseline uses YOLOv8n, 416x416 input resolution, 30 epochs, batch size 8 and seed 42.

Frozen validation metrics:

- mAP50: 0.238
- mAP50-95: 0.181
- Precision: 0.400
- Recall: 0.266

### 4. Diagnose instead of guessing

I added a separate FP/FN analysis path using IoU >= 0.5 and a low confidence threshold for diagnostics. This is intentionally kept separate from COCO-style AP evaluation.

The strongest signal was object size:

- Tiny (<1% image area): 154 / 204 = 75.5%
- Small (1-5%): 28 / 204 = 13.7%
- Large (>5%): 22 / 204 = 10.8%

The cigarette class was the hardest example in the diagnostic analysis, with 5 TP and 48 FN.

### 5. Form a hypothesis

If tiny objects dominate the failures, increasing the image resolution should preserve more spatial detail for the detector.

The important point is that this change was motivated by an observed failure mode, rather than by trying arbitrary hyperparameters.

### 6. Run the controlled comparison

I kept the final task definition and frozen validation split unchanged and compared the 416px baseline with a 640px run.

| Metric | 416px baseline | 640px best |
|---|---:|---:|
| mAP50 | 0.238 | 0.315 |
| mAP50-95 | 0.181 | 0.233 |
| Precision | 0.400 | 0.306 |
| Recall | 0.266 | 0.346 |

mAP50 improved by about 32% relative, and recall improved by about 30% relative.

### 7. Interpret the result correctly

The project does not claim that 640px solves the dataset. The gain supports the small-object hypothesis, but the absolute accuracy remains limited and the class imbalance still exists.

Also, the single reported precision value is not the same thing as AP. AP integrates the precision-recall curve across confidence thresholds, while the displayed precision/recall values correspond to the evaluator's selected operating point. So a lower reported precision at one operating point does not contradict a higher AP.

### 8. What this project demonstrates

The strongest engineering signal is the workflow:

Dataset audit -> frozen benchmark -> baseline -> error analysis -> hypothesis -> controlled experiment -> standardized evaluation -> reproducible inference.

That is the part I would transfer to a real client or production CV task.

## Likely interview questions

### Why did you reduce the task from 60 classes to 8?

Because the original taxonomy is extremely long-tailed. The scope study showed that many classes have too few samples for a stable portfolio-scale experiment. The 8-class task was therefore frozen as a more measurable benchmark. I do not compare scores across different class scopes as if they were model improvements.

### Why use YOLOv8n instead of a larger model?

The goal was not to maximize a leaderboard score with compute. It was to demonstrate a reproducible optimization workflow. YOLOv8n makes iterations cheap and makes it easier to attribute changes to the experimental variable rather than model scale.

### Why did you increase resolution before changing architecture?

Error analysis showed that 75.5% of validation objects were tiny. Resolution was therefore the most direct variable connected to the observed failure mode. It was a simpler and more interpretable intervention than immediately replacing the architecture.

### Why is the scope-study 8-class score different from the frozen baseline score?

They come from different stages of the project. The scope study was used to choose a task definition; the later frozen baseline is the benchmark used for controlled optimization. The repository explicitly avoids treating the scope-study scores as optimization comparisons.

### How did you avoid data leakage or cherry-picking?

The final train/val/test membership is stored in a frozen manifest, and all reported before/after evaluation uses the same validation split. Evaluation and error-analysis scripts are also separated from training.

### Why are your FP/FN counts different from Ultralytics precision/AP?

The FP/FN script is a diagnostic matcher with an explicit IoU and confidence threshold. Ultralytics AP uses a different evaluation procedure across confidence thresholds. I use FP/FN counts to understand failure modes, not as a replacement for COCO-style AP.

### Why did precision decrease while mAP increased?

The displayed precision value is a single operating point, whereas AP summarizes the precision-recall curve. The 640px model recovered more objects and improved the overall ranking/PR behavior enough to raise AP and recall, even though the reported precision at the selected operating point was lower.

### What would you do next if the goal were maximum accuracy?

I would keep the frozen benchmark and test one hypothesis at a time. The next candidates would be interventions specifically tied to tiny-object and imbalance failures, such as tiling/cropping, class-aware sampling or augmentation, and then possibly model scaling. These are follow-up ideas, not results claimed by this repository.

### What would change in a production project?

I would add target-hardware latency and memory benchmarks, choose confidence thresholds based on business cost, test distribution shift, log model/data versions, and define monitoring for false positives and false negatives after deployment. The current repository focuses on the model-development and evaluation workflow.

### What part did you personally implement?

The repository includes the dataset preparation and frozen-manifest workflow, training wrapper, standardized evaluation, FP/FN analysis, run comparison, inference CLI, tests and the experiment documentation. In an interview, answer this question only with the parts you personally wrote or verified; do not attribute work done entirely by external tools to yourself.

## Whiteboard version

If asked to explain the project on a whiteboard, draw this sequence:

TACO 60 classes
    -> dataset audit / long-tail problem
    -> frozen 8-class task + fixed split
    -> YOLOv8n @ 416 baseline
    -> FP/FN + size analysis
    -> 75.5% tiny objects
    -> hypothesis: spatial resolution bottleneck
    -> YOLOv8n @ 640
    -> mAP50 0.238 -> 0.315
    -> remaining errors / next experiments

## Numbers worth memorizing

Do not memorize every number. These are enough:

- 60 original classes -> focused 8-class benchmark
- frozen split: 640 / 80 / 80 images
- 75.5% of validation objects are tiny
- baseline mAP50: 0.238
- best mAP50: 0.315
- relative mAP50 gain: about 32%
- recall: 0.266 -> 0.346
- model: YOLOv8n
- main controlled change: input resolution 416 -> 640

## What not to overclaim

Do not say:

- that the project reaches state-of-the-art accuracy;
- that reducing 60 classes to 8 is itself a model optimization gain;
- that the 32% relative mAP improvement proves resolution is the only bottleneck;
- that diagnostic FP/FN counts are equivalent to COCO AP;
- that untested ideas such as tiling, deployment optimization or larger models were already implemented.

The strongest claim is narrower and more credible: the repository demonstrates a disciplined, reproducible computer-vision optimization workflow in which error analysis leads to a measurable controlled improvement.
