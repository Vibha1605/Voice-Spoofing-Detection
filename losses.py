import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# OC-Softmax Loss (One-Class Softmax)
def oc_softmax_loss(logits, labels, margin=0.35, scale=30):
    """
    OC-Softmax Loss implementation. This loss encourages the separation between the classes
    by enforcing an angular margin between them.
    
    Args:
        logits: Raw model outputs (before softmax).
        labels: Ground truth labels (0 for bonafide, 1 for spoof).
        margin: Angular margin applied to logits for hard separation.
        scale: Scaling factor to control logits magnitude.
    
    Returns:
        loss: OC-Softmax loss value.
    """
    # Convert logits to cosine similarity and normalize
    cos_theta = F.normalize(logits)
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

    # Apply margin for OC-Softmax
    cos_theta_margin = torch.acos(cos_theta) + margin * one_hot
    cos_theta_margin = torch.cos(cos_theta_margin)

    # Scale the logits for stability
    scaled_logits = scale * cos_theta_margin

    # Compute softmax and cross-entropy loss
    loss = F.cross_entropy(scaled_logits, labels)
    return loss

# AM-Softmax Loss (Additive Margin Softmax)
def am_softmax_loss(logits, labels, margin=0.35, scale=30):
    """
    AM-Softmax Loss implementation. Similar to OC-Softmax, but directly adds margin to the logits
    in a simpler way.
    
    Args:
        logits: Raw model outputs (before softmax).
        labels: Ground truth labels (0 for bonafide, 1 for spoof).
        margin: Margin added to logits for better class separation.
        scale: Scaling factor to control logits magnitude.
    
    Returns:
        loss: AM-Softmax loss value.
    """
    # Normalize logits (cosine similarity)
    cos_theta = F.normalize(logits)
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

    # Add margin
    cos_theta_margin = cos_theta - margin * one_hot

    # Scale the logits for better convergence
    scaled_logits = scale * cos_theta_margin

    # Compute softmax and cross-entropy loss
    loss = F.cross_entropy(scaled_logits, labels)
    return loss

# DOC-Softmax Loss (Double Output Cosine Softmax)
def doc_softmax_loss(logits, labels, margin1=0.35, margin2=0.10, scale=30):
    """
    DOC-Softmax Loss implementation. This loss applies two margins (inner and outer) 
    for separating genuine and spoofed classes more effectively.
    
    Args:
        logits: Raw model outputs (before softmax).
        labels: Ground truth labels (0 for bonafide, 1 for spoof).
        margin1: Outer margin applied to bonafide samples.
        margin2: Inner margin applied to spoof samples.
        scale: Scaling factor to control logits magnitude.
    
    Returns:
        loss: DOC-Softmax loss value.
    """
    # Normalize logits
    cos_theta = F.normalize(logits)
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

    # Apply two different margins: margin1 for bonafide, margin2 for spoof
    cos_theta_margin = cos_theta - margin1 * one_hot + margin2 * (1 - one_hot)

    # Scale logits for better performance
    scaled_logits = scale * cos_theta_margin

    # Compute softmax and cross-entropy loss
    loss = F.cross_entropy(scaled_logits, labels)
    return loss