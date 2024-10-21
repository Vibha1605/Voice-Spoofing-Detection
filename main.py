from dataset import ASVspoof2021
from model import TeacherModel, StudentModel
from train import train_teacher, train_student

if __name__ == "__main__":
    # Paths
    path_to_features = 'path_to_features'
    path_to_protocol = 'path_to_protocol'
    
    # Initialize dataset
    dataset = ASVspoof2021(path_to_features, path_to_protocol)
    
    # Train teacher model
    teacher_model = TeacherModel()
    train_teacher(teacher_model, dataset)
    
    # Train student model
    student_model = StudentModel()
    train_student(student_model, teacher_model, dataset)
