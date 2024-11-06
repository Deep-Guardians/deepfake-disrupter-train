import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from io import BytesIO
from sklearn.metrics import precision_score, recall_score
import cv2
import numpy as np

# Validation function to evaluate model on validation set
def validate(dataloader, perturbation_generator, face_detector, device):
    perturbation_generator.eval()
    all_labels = []
    all_preds = []
    all_face_embedding_loss = 0
    # all_detection_loss_fake = 0
    all_perturbation_loss = 0
    all_identity_loss = 0
    counts = 0
    fake_pred = 0

    with torch.no_grad():
        for real_images in dataloader:
            counts+=1
            real_images = real_images.to(device)
            perturbations = perturbation_generator(real_images)
            perturbed_images = real_images + perturbations
            # perturbed_images = torch.clamp(perturbed_images, 0, 1)
 
            real_embe = face_detector(real_images)
            disrupt_embe = face_detector(perturbed_images)

            # Predictions for real and fake images
            # outputs_real = deepfake_detector(perturbed_images)
            # outputs_fake = deepfake_detector(fake_images)

            perturbation_loss = hinge_loss(perturbations)
            distance = F.pairwise_distance(real_embe, disrupt_embe)
            face_embedding_loss = torch.mean(F.relu(2.0 - distance))
            # detection_loss_real = criterion_bce(outputs_real, torch.ones_like(outputs_real).to(device))
            # detection_loss_fake = criterion_bce(outputs_fake, torch.zeros_like(outputs_fake).to(device))
            identity_loss = criterion_identity(perturbed_images, real_images)

            all_perturbation_loss += perturbation_loss.item()
            all_face_embedding_loss += face_embedding_loss.item()
            # all_detection_loss_fake+=detection_loss_fake.item()
            all_identity_loss += identity_loss.item()

            # Collect predictions and labels
            # real_preds = (distance < 1.0).int().view(-1).cpu().numpy()
            if torch.any(distance >= 1.0):
                fake_pred += 1
            # real_labels = torch.ones_like(outputs_real).int().view(-1).cpu().numpy()
            # fake_labels = torch.zeros_like(outputs_fake).int().view(-1).cpu().numpy()

            # all_preds.extend(real_preds)
            # all_preds.extend(fake_preds)
            # all_labels.extend(real_labels)
            # all_labels.extend(fake_labels)

    # Calculate precision and recall
    # precision = precision_score(all_labels, all_preds)
    # recall = recall_score(all_labels, all_preds)

    all_perturbation_loss /= counts
    all_face_embedding_loss /= counts
    # all_detection_loss_fake /= counts
    all_identity_loss /= counts

    perturbation_generator.train()

    wandb.log({
        'val_perturbation_loss': all_perturbation_loss,
        'val_face_emnedding_loss': all_face_embedding_loss,
        'val_identity_loss': all_identity_loss,
        'val_pred_counts': fake_pred,
        'val_counts': counts,
    })


def pil_to_tensor(pil_image):
    to_tensor = ToTensor()
    return to_tensor(pil_image)


def tensor_to_image(tensor):
    to_pil = ToPILImage()
    return to_pil(tensor.cpu().detach())


def criterion_identity(predicted, target):
    return -nn.MSELoss()(predicted, target)


def hinge_loss(perturbation, epsilon = 0.05):
    perturbation_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3))
    loss = torch.mean(torch.clamp(perturbation_norm - epsilon, min=0.001))
    return loss

def cv2_image_to_tensor(cv2_image):
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # (H, W, C) -> (C, H, W)
    
    tensor_image = torch.from_numpy(image_rgb).permute(2, 0, 1)

    
    return tensor_image
