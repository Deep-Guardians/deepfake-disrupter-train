import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.nn.functional as F
from torchvision import transforms
from model.U_Net import UNet
from model.detect_models import model_selection
import insightface
# from PIL import Image
import cv2
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
# from face2face import Face2Face
from utils.dataloader import create_dataloader
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
# from io import BytesIO
# from joblib import Parallel, delayed
from utils.evaluation import validate
import numpy as np
from res_facenet.models import model_921
# fake_image_generator = Face2Face(device_id=3)

def pil_to_tensor(pil_image):
    to_tensor = ToTensor()
    return to_tensor(pil_image)

def tensor_to_cv2_image(tensor_image):
    # Tensor: (C, H, W) -> Numpy: (H, W, C)
    image_np = tensor_image.permute(1, 2, 0).cpu().detach().numpy()
    
    # Scale pixel values from [0, 1] to [0, 255]
    image_np = image_np.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def cv2_image_to_tensor(cv2_image):
    # BGR -> RGB로 변환
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # (H, W, C) -> (C, H, W)
    tensor_image = torch.from_numpy(image_rgb).permute(2, 0, 1)

    return tensor_image

def tensor_to_image(tensor):
    to_pil = ToPILImage()
    return to_pil(tensor.cpu().detach())


def criterion_identity(predicted, target):
    return 1- torch.sigmoid(nn.MSELoss()(predicted, target))


def hinge_loss(perturbation, epsilon):
    perturbation_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3))
    loss = torch.mean(torch.clamp(perturbation_norm - epsilon, min=0.001))
    return loss


def gradnorm_update(losses, initial_losses, alpha, weights):
    avg_loss = sum(losses) / len(losses)
    loss_ratios = [loss / initial_loss for loss, initial_loss in zip(losses, initial_losses)]
    relative_losses = [ratio / avg_loss for ratio in loss_ratios]
    grad_norms = [(relative_loss ** alpha) * weight for relative_loss, weight in zip(relative_losses, weights)]
    new_weights = [weight * (grad_norm / sum(grad_norms)) for weight, grad_norm in zip(weights, grad_norms)]
    return new_weights


def train(hp, train_loader, valid_loader, chkpt_path):
    run = wandb.init(
        project=hp.log.project_name,
        config={
            "learning_rate": hp.train.lr,
            "architecture": "U-Net",
            "dataset": "celeba",
            "epochs": hp.train.epochs,
        }
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    init_epoch = 0
    perturbation_generator = UNet(3).to(torch.device("cuda:0")).train()
    face_detector = model_921().to(device).eval()
    # deepfake_detector = model_selection(modelname='xception', num_out_classes=2, dropout=0.5).to(torch.device("cuda:0")).eval()
    
    # face_detector = FaceAnalysis(name='buffalo_l')
    # face_detector.prepare(ctx_id=0, det_size=(hp.data.image_size, hp.data.image_size))
    # fake_image_generator = insightface.model_zoo.get_model('./pretrained_model/inswapper_128.onnx', download=False, download_zip=False)
    # fake_image_generator = Face2Face(device_id=3)
    # fake_image_generator = torch.jit.script(fake_image_generator)
    
    perturbation_generator = nn.DataParallel(perturbation_generator, device_ids=[0, 1, 2, 3])
    face_detector = nn.DataParallel(face_detector, device_ids=[0, 1, 2, 3])
    # fake_image_generator = nn.DataParallel(fake_image_generator, device_ids=[0, 1, 2])

    optimizer = optim.Adam(perturbation_generator.parameters(), lr=hp.train.lr)
    criterion_bce = nn.BCEWithLogitsLoss()

    initial_losses = None
    if chkpt_path is not None:
        print("checkpoint loaded : ", chkpt_path)
        checkpoint = torch.load(chkpt_path)
        # state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        perturbation_generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # initial_losses = checkpoint['loss']
        init_epoch = checkpoint['epoch']

    w1, w2, w3 = 1.0, 1.0, 1.0
    wandb.watch(perturbation_generator)
    # run.log_model(path="./unet", name="v1")
    
    # sample_img = cv2.imread("./sample/src.jpg")
    # transform = transforms.Compose([
    #    transforms.Resize([hp.data.image_size, hp.data.image_size]),
    #    transforms.ToTensor(),
    # ])
    # sample_img = transform(sample_img)
    # sample_face = face_detector.get(sample_img)
    # sample_face = sample_face[0]

    alpha = 0.1
    margin = 2.0

    for epoch in range(init_epoch, hp.train.epochs):
        for i, real_images in enumerate(train_loader):
            # validate(valid_loader, perturbation_generator, face_detector, device)

            real_images = real_images.to(torch.device("cuda:0"))
            perturbations = perturbation_generator(real_images)
            perturbed_images = real_images + perturbations
            # perturbed_images = torch.clamp(perturbed_images, 0, 1)
            
            real_embe = face_detector(real_images)
            disrupt_embe = face_detector(perturbed_images)

            perturbation_loss = hinge_loss(perturbations, hp.train.epsilon)
            # detection_loss_real = criterion_bce(outputs_real, torch.ones_like(outputs_real).to(device))
            # detection_loss_fake = criterion_bce(outputs_fake, torch.zeros_like(outputs_fake).to(device))
            face_embedding_loss = torch.mean(F.relu(margin - F.pairwise_distance(real_embe, disrupt_embe)))
            identity_loss = 1-torch.sigmoid(nn.MSELoss()(perturbed_images, real_images))

            losses = [face_embedding_loss, perturbation_loss, identity_loss]

            if initial_losses is None:
                initial_losses = [loss.item() for loss in losses]
            # else:
                # weights = gradnorm_update([loss.item() for loss in losses], initial_losses, alpha, [w1, w2, w3])
                # weights = gradnorm_update([loss.item() for loss in losses], initial_losses, alpha, [w1, w2, w3])
                # w1, w2, w3 = weights

            total_loss = (
                face_embedding_loss + 0.1*perturbation_loss + identity_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print('epoch : ', epoch, 'loss : ', total_loss.item(),
                  'face_embedding_loss : ', face_embedding_loss.item(),
                  'perturbation_loss : ', perturbation_loss.item(),
                  'identity_loss : ', identity_loss.item())
            wandb.log({
                'epoch': epoch,
                'loss': total_loss.item(),
                'face_embedding_loss' : face_embedding_loss.item(),
                'perturbation_loss': perturbation_loss.item(),
                'identity_loss': identity_loss.item(),
            })

        checkpoint_path = f"{hp.log.chkpt}/unetv2_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': perturbation_generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss.item()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        print(f"Epoch [{epoch + 1}/{hp.train.epochs}], Loss: {total_loss.item()}")
        validate(valid_loader, perturbation_generator, face_detector, device)
