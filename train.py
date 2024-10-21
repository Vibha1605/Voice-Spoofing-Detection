import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ASVspoof2021
from model.models import TeacherModel, StudentModel
from losses import oc_softmax_loss, am_softmax_loss, doc_softmax_loss

# Knowledge Distillation Loss
def knowledge_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Computes the knowledge distillation loss using soft targets from the teacher model.
    
    Args:
        student_logits: Logits from the student model.
        teacher_logits: Logits from the teacher model.
        temperature: Temperature scaling factor for distillation.
    
    Returns:
        loss: Knowledge distillation loss.
    """
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
    student_probs = torch.log_softmax(student_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs) * (temperature ** 2)
    return loss

def train_model(
    model, 
    dataloader, 
    optimizer, 
    loss_fn, 
    epoch, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains the model for one epoch.
    
    Args:
        model: The model to train (student or teacher).
        dataloader: The DataLoader for the training data.
        optimizer: Optimizer for the model.
        loss_fn: Loss function (could be oc_softmax_loss, am_softmax_loss, doc_softmax_loss).
        epoch: Current epoch number.
        device: Device to train on (GPU or CPU).
    
    Returns:
        avg_loss: Average loss over the epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, (lfcc_feat, mfcc_feat, filenames, tags, labels) in enumerate(dataloader):
        # Move data to device
        lfcc_feat = lfcc_feat.to(device)
        mfcc_feat = mfcc_feat.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(torch.cat([lfcc_feat, mfcc_feat], dim=1))  # Concatenate LFCC and MFCC
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
    return avg_loss

def distillation_train(
    student_model, 
    teacher_model, 
    dataloader, 
    optimizer, 
    loss_fn, 
    temperature=2.0, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains the student model using knowledge distillation.
    
    Args:
        student_model: The student model to be trained.
        teacher_model: The teacher model providing soft targets.
        dataloader: The DataLoader for the training data.
        optimizer: Optimizer for the student model.
        loss_fn: Loss function (e.g., oc_softmax_loss, am_softmax_loss, doc_softmax_loss).
        temperature: Temperature for distillation.
        device: Device to train on (GPU or CPU).
    
    Returns:
        avg_loss: Average loss over the epoch.
    """
    student_model.train()
    teacher_model.eval()  # Teacher model is fixed

    total_loss = 0

    for batch_idx, (lfcc_feat, mfcc_feat, filenames, tags, labels) in enumerate(dataloader):
        # Move data to device
        lfcc_feat = lfcc_feat.to(device)
        mfcc_feat = mfcc_feat.to(device)
        labels = labels.to(device)

        # Teacher model prediction (soft labels)
        with torch.no_grad():
            teacher_logits = teacher_model(torch.cat([lfcc_feat, mfcc_feat], dim=1))

        # Student model prediction
        student_logits = student_model(torch.cat([lfcc_feat, mfcc_feat], dim=1))

        # Compute knowledge distillation loss + regular loss
        kd_loss = knowledge_distillation_loss(student_logits, teacher_logits, temperature)
        task_loss = loss_fn(student_logits, labels)
        total_loss_val = kd_loss + task_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        total_loss += total_loss_val.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Distillation Training, Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    temperature = 2.0  # For knowledge distillation
    loss_type = "oc_softmax"  # Choose 'oc_softmax', 'am_softmax', or 'doc_softmax'

    # Paths
    path_to_features = r"C:\Users\vibha srinivasan\Desktop\spoofing system\VSD_ASV\features_extracted"
    path_to_protocol = r"C:\Users\vibha srinivasan\Desktop\spoofing system\VSD_ASV\protocols"

    # Load dataset and dataloader
    dataset = ASVspoof2021(path_to_features=path_to_features, path_to_protocol=path_to_protocol, part='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # Select loss function based on loss_type
    if loss_type == "oc_softmax":
        loss_fn = oc_softmax_loss
    elif loss_type == "am_softmax":
        loss_fn = am_softmax_loss
    else:
        loss_fn = doc_softmax_loss

    # Train the teacher model (if necessary)
    for epoch in range(epochs):
        train_model(teacher_model, dataloader, optimizer, loss_fn, epoch)

    # Train the student model using distillation
    for epoch in range(epochs):
        distillation_train(student_model, teacher_model, dataloader, optimizer, loss_fn, temperature)