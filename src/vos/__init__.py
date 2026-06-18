"""Video object segmentation backends, propagation, memory, and ensembling."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ReliabilityConfig",
    "ReliabilityResult",
    "AnchorMiningConfig",
    "AnchorMiningResult",
    "AnchorScore",
    "CandidateMask",
    "CandidatePath",
    "MaskCandidate",
    "MemoryAnchor",
    "MemoryDecision",
    "MemoryGovernanceConfig",
    "MemoryTreeConfig",
    "MemoryTreeResult",
    "MultiAnchorPropagationConfig",
    "MultiAnchorPropagationResult",
    "MergeConfig",
    "MergeResult",
    "EnsembleConfig",
    "EnsembleResult",
    "ModelFramePrediction",
    "ObjectMaskPrediction",
    "ObjectTrackState",
    "PostprocessConfig",
    "PostprocessResult",
    "PredictionSetConfig",
    "PropagationAnchor",
    "AnchorPropagationResult",
    "TTAConfig",
    "TTAFusionResult",
    "TTAVariant",
    "bbox_iou",
    "build_frame_candidates",
    "build_tta_variants",
    "classify_state",
    "collect_object_ids_for_video",
    "compute_reliability",
    "cosine_similarity",
    "detect_drift",
    "detect_lost",
    "draw_reliability_overlay",
    "extract_masked_feature",
    "load_model_frame_prediction",
    "mask_area",
    "mask_iou",
    "mask_to_bbox",
    "mine_anchors_for_video",
    "merge_object_predictions",
    "normalize_prediction_sets",
    "parse_prediction_set_config",
    "morphological_close_open",
    "postprocess_object_mask",
    "reliability_to_json",
    "remove_small_components",
    "initialize_object_track_state",
    "run_memory_tree_search",
    "run_ensemble_for_video",
    "run_multi_anchor_bidirectional_propagation",
    "save_indexed_png",
    "save_memory_governance_debug",
    "smooth_boundary",
    "fill_holes",
    "fuse_tta_results",
    "apply_tta_transform",
    "invert_tta_result",
    "sigmoid",
    "update_tracking_state",
    "write_ensemble_debug_csv",
]


def __getattr__(name: str) -> Any:
    """Lazily expose VOS helpers."""

    if name in {
        "ReliabilityConfig",
        "ReliabilityResult",
        "bbox_iou",
        "classify_state",
        "compute_reliability",
        "cosine_similarity",
        "detect_drift",
        "detect_lost",
        "draw_reliability_overlay",
        "extract_masked_feature",
        "mask_area",
        "mask_iou",
        "mask_to_bbox",
        "reliability_to_json",
        "sigmoid",
    }:
        from src.vos import reliability

        return getattr(reliability, name)
    if name in {
        "AnchorMiningConfig",
        "AnchorMiningResult",
        "AnchorScore",
        "CandidateMask",
        "mine_anchors_for_video",
    }:
        from src.vos import anchor_mining

        return getattr(anchor_mining, name)
    if name in {
        "CandidatePath",
        "MaskCandidate",
        "MemoryTreeConfig",
        "MemoryTreeResult",
        "build_frame_candidates",
        "run_memory_tree_search",
    }:
        from src.vos import memory_tree

        return getattr(memory_tree, name)
    if name in {
        "MemoryAnchor",
        "MemoryDecision",
        "MemoryGovernanceConfig",
        "ObjectTrackState",
        "initialize_object_track_state",
        "save_memory_governance_debug",
        "update_tracking_state",
    }:
        from src.vos import memory_governance

        return getattr(memory_governance, name)
    if name in {
        "AnchorPropagationResult",
        "MultiAnchorPropagationConfig",
        "MultiAnchorPropagationResult",
        "PropagationAnchor",
        "run_multi_anchor_bidirectional_propagation",
    }:
        from src.vos import multi_anchor_propagation

        return getattr(multi_anchor_propagation, name)
    if name in {
        "EnsembleConfig",
        "EnsembleResult",
        "ModelFramePrediction",
        "PredictionSetConfig",
        "collect_object_ids_for_video",
        "load_model_frame_prediction",
        "normalize_prediction_sets",
        "parse_prediction_set_config",
        "run_ensemble_for_video",
        "write_ensemble_debug_csv",
    }:
        from src.vos import ensemble

        return getattr(ensemble, name)
    if name in {
        "MergeConfig",
        "MergeResult",
        "ObjectMaskPrediction",
        "PostprocessConfig",
        "PostprocessResult",
        "TTAConfig",
        "TTAFusionResult",
        "TTAVariant",
        "apply_tta_transform",
        "build_tta_variants",
        "fill_holes",
        "fuse_tta_results",
        "invert_tta_result",
        "merge_object_predictions",
        "morphological_close_open",
        "postprocess_object_mask",
        "remove_small_components",
        "save_indexed_png",
        "smooth_boundary",
    }:
        from src.vos import postprocess

        return getattr(postprocess, name)
    raise AttributeError(f"module 'src.vos' has no attribute {name!r}")
