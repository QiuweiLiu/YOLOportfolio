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
    # 错误类别 = FP, GT 未被正确匹配 = FN (class-aware)
    assert len(tp) == 0
    assert len(fp) == 1
    assert len(fn) == 1


def test_confidence_reordering():
    # 原始 predictions: index 0 低分 0.2, index 1 高分 0.9
    # 排序后高分先匹配, 返回的 index 应为原始 index
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    pred_boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    pred_classes = [0, 0]
    pred_scores = [0.2, 0.9]  # index 1 是高分
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5)
    assert len(tp) == 1
    assert tp[0][0] == 1  # original index 1 (高分) 匹配
    assert fp == [0]  # original index 0 成为 FP


def test_class_aware_matching():
    # 一个错误类别框 IoU 0.95, 一个正确类别框 IoU 0.8 同时存在
    # 正确类别框应成功匹配 GT (class-aware)
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    pred_boxes = [[0, 0, 10, 10], [1, 1, 9, 9]]  # both overlap GT
    pred_classes = [1, 0]  # first wrong class (high IoU), second correct (slightly lower IoU)
    pred_scores = [0.95, 0.9]
    # 按 confidence 排序: wrong class 0.95先处理 -> FP (no same-class GT), then correct 0.9 -> TP
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5)
    assert len(tp) == 1
    assert tp[0][0] == 1  # correct class prediction (original index 1) 匹配
    assert len(fp) == 1
    assert fp[0] == 0  # wrong class (original index 0) FP
    assert len(fn) == 0


def test_matching_no_prediction():
    gt_boxes = [[0, 0, 10, 10]]
    gt_classes = [0]
    tp, fp, fn = match_predictions(gt_boxes, gt_classes, [], [], [], 0.5)
    assert len(tp) == 0
    assert len(fp) == 0
    assert len(fn) == 1
