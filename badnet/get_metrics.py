import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter
import pandas as pd


def get_test_metrics(y_pred, y_true, sessions=None, tolerance=0, y_proba=None, 
                    participant_ids=None, video_ids=None, window_size=5, slide_length=1):
    """
    Comprehensive metrics calculation including basic metrics, AUC, and temporal window analysis.
    
    Args:
        y_pred (array): Predicted labels
        y_true (array): True labels
        sessions (array, optional): Session identifiers for tolerance calculation
        tolerance (int): Tolerance for "close enough" predictions
        y_proba (array, optional): Prediction probabilities for AUC calculation (N x num_classes)
        participant_ids (array, optional): Participant identifiers for each sample
        video_ids (array, optional): Video/question identifiers for each sample
        window_size (int): Size of sliding window for temporal analysis
        slide_length (int): Step size for sliding window
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if sessions is not None:
        sessions = np.array(sessions)
    if participant_ids is not None:
        participant_ids = np.array(participant_ids)
    if video_ids is not None:
        video_ids = np.array(video_ids)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(classes)

    # =============================================================================
    # 1. BASIC METRICS WITH TOLERANCE
    # =============================================================================
    
    y_pred_tolerant = y_pred.copy()

    if tolerance > 0:
        for i in range(len(y_pred)):
            if sessions is not None:
                same_session_indices = np.where(sessions == sessions[i])[0]
                
                if i == same_session_indices[same_session_indices <= i][0]:
                    start = same_session_indices[same_session_indices <= i][0]
                else:
                    start = max(same_session_indices[same_session_indices <= i][-1] - tolerance, same_session_indices[same_session_indices <= i][0])
                
                if i == same_session_indices[same_session_indices >= i][-1]:
                    end = same_session_indices[same_session_indices >= i][-1]
                else:
                    end = min(same_session_indices[same_session_indices >= i][0] + tolerance + 1, same_session_indices[same_session_indices >= i][-1] + 1)
            else:
                start = max(0, i - tolerance)
                end = min(len(y_true), i + tolerance + 1)
            
            if y_pred[i] in y_true[start:end]:
                y_pred_tolerant[i] = y_true[i]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    # Tolerant metrics
    accuracy_tolerant = accuracy_score(y_true, y_pred_tolerant)
    precision_tolerant = precision_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    recall_tolerant = recall_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    f1_tolerant = f1_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)

    # =============================================================================
    # 2. AUC CALCULATION
    # =============================================================================
    
    auc_score = None
    if y_proba is not None:
        try:
            if num_classes == 2:
                # Binary classification: use probabilities for positive class
                auc_score = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class: use one-vs-rest AUC
                auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Warning: Could not calculate AUC - {e}")
            auc_score = None

    # =============================================================================
    # 3. TEMPORAL WINDOW ANALYSIS ("ON THE FLY" METRICS)
    # =============================================================================
    
    window_metrics = None
    if participant_ids is not None and video_ids is not None:
        window_metrics = calculate_temporal_window_metrics(
            y_pred, y_true, y_proba, participant_ids, video_ids, 
            window_size, slide_length
        )

    # =============================================================================
    # 4. COMPILE RESULTS
    # =============================================================================
    
    results = {
        # Basic metrics
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        
        # Tolerant metrics
        "test_accuracy_tolerant": accuracy_tolerant,
        "test_precision_tolerant": precision_tolerant,
        "test_recall_tolerant": recall_tolerant,
        "test_f1_tolerant": f1_tolerant,
        
        # AUC
        "test_auc": auc_score,
        
        # Raw predictions for further analysis
        "predictions": y_pred,
        "true_labels": y_true,
        "probabilities": y_proba
    }
    
    # Add window metrics if available
    if window_metrics is not None:
        results.update(window_metrics)
    
    return results


def calculate_temporal_window_metrics(y_pred, y_true, y_proba, participant_ids, video_ids, 
                                     window_size=5, slide_length=1):
    """
    Calculate temporal window analysis metrics for each participant-video combination.
    
    This function analyzes predictions using sliding windows to understand:
    1. Video-level accuracy: Is the mode prediction across all windows correct?
    2. Window-level accuracy: What percentage of windows have correct predictions?
    3. Early detection: How quickly is the correct label detected?
    
    Args:
        y_pred: Frame-level predictions
        y_true: Frame-level true labels
        y_proba: Frame-level prediction probabilities
        participant_ids: Participant identifier for each frame
        video_ids: Video identifier for each frame
        window_size: Number of frames per window
        slide_length: Step size between windows
    
    Returns:
        dict: Window analysis metrics
    """
    
    # Get unique participant-video combinations
    unique_combinations = list(set(zip(participant_ids, video_ids)))
    
    # Storage for results
    video_level_results = []
    window_level_results = []
    early_detection_results = []
    
    print(f"\nAnalyzing {len(unique_combinations)} participant-video combinations...")
    print(f"Window size: {window_size}, Slide length: {slide_length}")
    
    for participant_id, video_id in unique_combinations:
        # Get all frames for this participant-video combination
        mask = (participant_ids == participant_id) & (video_ids == video_id)
        video_indices = np.where(mask)[0]
        
        if len(video_indices) < window_size:
            continue  # Skip if not enough frames
        
        video_preds = y_pred[video_indices]
        video_true = y_true[video_indices]
        video_proba = y_proba[video_indices] if y_proba is not None else None
        
        # Get the true label for this video (should be consistent)
        true_video_label = Counter(video_true).most_common(1)[0][0]
        
        # Apply sliding window analysis
        window_predictions = []
        window_confidences = []
        
        # Slide window across the video
        for start_idx in range(0, len(video_preds) - window_size + 1, slide_length):
            end_idx = start_idx + window_size
            
            # Get predictions and probabilities for this window
            window_pred_frames = video_preds[start_idx:end_idx]
            
            if video_proba is not None:
                window_proba_frames = video_proba[start_idx:end_idx]
                # Average probabilities across window
                avg_proba = np.mean(window_proba_frames, axis=0)
                window_prediction = np.argmax(avg_proba)
                window_confidence = np.max(avg_proba)
            else:
                # Use mode of frame predictions
                window_prediction = Counter(window_pred_frames).most_common(1)[0][0]
                window_confidence = None
            
            window_predictions.append(window_prediction)
            if window_confidence is not None:
                window_confidences.append(window_confidence)
        
        if len(window_predictions) == 0:
            continue
        
        # =============================================================================
        # METRIC 1: VIDEO-LEVEL ACCURACY (Mode prediction across all windows)
        # =============================================================================
        
        # Get the mode prediction across all windows
        mode_prediction = Counter(window_predictions).most_common(1)[0][0]
        video_level_correct = (mode_prediction == true_video_label)
        
        video_level_results.append({
            'participant_id': participant_id,
            'video_id': video_id,
            'true_label': true_video_label,
            'mode_prediction': mode_prediction,
            'correct': video_level_correct,
            'num_windows': len(window_predictions)
        })
        
        # =============================================================================
        # METRIC 2: WINDOW-LEVEL ACCURACY (Percentage of correct windows)
        # =============================================================================
        
        correct_windows = np.array(window_predictions) == true_video_label
        window_accuracy = np.mean(correct_windows)
        
        window_level_results.append({
            'participant_id': participant_id,
            'video_id': video_id,
            'window_accuracy': window_accuracy,
            'correct_windows': np.sum(correct_windows),
            'total_windows': len(window_predictions)
        })
        
        # =============================================================================
        # METRIC 3: EARLY DETECTION (How quickly is correct label detected?)
        # =============================================================================
        
        # Find first window where correct prediction is made
        first_correct_window = None
        for i, pred in enumerate(window_predictions):
            if pred == true_video_label:
                first_correct_window = i
                break
        
        # Calculate detection metrics
        if first_correct_window is not None:
            detection_time_windows = first_correct_window + 1  # 1-indexed
            detection_time_frames = first_correct_window * slide_length + window_size
            detection_percentage = (detection_time_frames / len(video_preds)) * 100
        else:
            detection_time_windows = len(window_predictions) + 1  # Never detected
            detection_time_frames = len(video_preds) + 1
            detection_percentage = 100.0
        
        early_detection_results.append({
            'participant_id': participant_id,
            'video_id': video_id,
            'first_correct_window': first_correct_window,
            'detection_time_windows': detection_time_windows,
            'detection_time_frames': detection_time_frames,
            'detection_percentage': detection_percentage,
            'total_frames': len(video_preds)
        })
    
    # =============================================================================
    # AGGREGATE METRICS
    # =============================================================================
    
    if len(video_level_results) == 0:
        return {"window_analysis": "No valid participant-video combinations found"}
    
    # Video-level metrics
    video_level_accuracy = np.mean([r['correct'] for r in video_level_results])
    
    # Window-level metrics  
    avg_window_accuracy = np.mean([r['window_accuracy'] for r in window_level_results])
    
    # Early detection metrics
    avg_detection_time_windows = np.mean([r['detection_time_windows'] for r in early_detection_results])
    avg_detection_percentage = np.mean([r['detection_percentage'] for r in early_detection_results])
    
    # Count videos where detection never happened
    never_detected = sum(1 for r in early_detection_results if r['first_correct_window'] is None)
    
    return {
        # Aggregated metrics
        "video_level_accuracy": video_level_accuracy,
        "avg_window_accuracy": avg_window_accuracy, 
        "avg_detection_time_windows": avg_detection_time_windows,
        "avg_detection_percentage": avg_detection_percentage,
        "never_detected_count": never_detected,
        "total_videos": len(video_level_results),
        
        # Detailed results for further analysis
        "video_level_results": video_level_results,
        "window_level_results": window_level_results,
        "early_detection_results": early_detection_results,
        
        # Window analysis parameters
        "window_analysis_params": {
            "window_size": window_size,
            "slide_length": slide_length,
            "num_videos_analyzed": len(video_level_results)
        }
    }

