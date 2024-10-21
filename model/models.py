import torch
import torch.nn as nn
import torch.nn.functional as F

# Teacher Model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Feature extractor (e.g., CNN layers)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 9 * 9, 512)  # Adjust according to feature size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # Output 2 classes: spoof or bonafide

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the output
        x = x.view(-1, 128 * 9 * 9)  # Adjust according to the feature size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Student Model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Simplified version of TeacherModel with fewer parameters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)  # Adjust according to feature size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Output 2 classes: spoof or bonafide

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(-1, 32 * 9 * 9)  # Adjust according to the feature size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion_ce = nn.CrossEntropyLoss()  # For hard labels (true labels)
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')  # For soft labels (teacher's predictions)

    def forward(self, student_outputs, teacher_outputs, labels):
        # Compute soft targets from teacher outputs
        soft_teacher_outputs = F.log_softmax(teacher_outputs / self.temperature, dim=1)
        soft_student_outputs = F.log_softmax(student_outputs / self.temperature, dim=1)

        # Compute the distillation loss (KL divergence between student and teacher soft predictions)
        distillation_loss = self.criterion_kl(soft_student_outputs, soft_teacher_outputs) * (self.temperature ** 2)

        # Compute standard classification loss (Cross-Entropy between student outputs and true labels)
        ce_loss = self.criterion_ce(student_outputs, labels)

        # Total loss is a weighted sum of the two losses
        total_loss = (self.alpha * ce_loss) + ((1 - self.alpha) * distillation_loss)
        return total_loss
