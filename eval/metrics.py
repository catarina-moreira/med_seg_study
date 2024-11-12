import numpy as np

def dice_coefficient(pred, gt):
    """Calculate Dice Similarity Coefficient"""
    intersection = np.sum((pred > 0) * (gt > 0))
    return 2.0 * intersection / (np.sum(pred > 0) + np.sum(gt > 0))

def iou(pred, gt):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.sum((pred > 0) * (gt > 0))
    union = np.sum((pred > 0) + (gt > 0))
    return intersection / union

def mean_squared_error(pred, gt):
    """Calculate Mean Squared Error (MSE)"""
    return np.mean((pred - gt) ** 2)

def pixel_accuracy(pred, gt):
    """Calculate Pixel-wise Accuracy"""
    return np.sum(pred == gt) / pred.size


