"""Test: FP/FN matching logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_errors import compute_iou, match_predictions


def test_iou_perfect():
    assert abs(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6


def test_iou_none():
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half():
    # 50% overlap
    assert abs(compute_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_matching_simple():
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    pred_boxes = [[0, 0, 10, 10]]
    pred_classes = [0]
    pred_scores = [0.9]
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5)
    assert len(tp) == 1
    assert len(fp) == 0
    assert len(fn) == 0


def test_matching_duplicate():
    # 两个预测对应同一个 GT,只有一个能匹配
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    pred_boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    pred_classes = [0, 0]
    pred_scores = [0.9, 0.8]
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5)
    assert len(tp) == 1
    assert len(fp) == 1
    assert len(fn) == 0


def test_matching_wrong_class():
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    pred_boxes = [[0, 0, 10, 10]]
    pred_classes = [1]  # 错误类别
    pred_scores = [0.9]
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5)
    # 错误类别 = FP, GT 被该预测占用(不算 FN,是错分类)
    assert len(tp) == 0
    assert len(fp) == 1
    assert len(fn) == 0


def test_matching_no_prediction():
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, [], [], [], 0.5)
    assert len(tp) == 0
    assert len(fp) == 0
    assert len(fn) == 1
