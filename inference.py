from torchvision.transforms import transforms
from PIL import Image
from model.U_Net import UNet
import torch
from torchvision.transforms import ToPILImage
import os
from utils.train import tensor_to_image


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

pretrained_model = torch.load('./checkpoints/unet_epoch_1.pth')
state_dict = {k.replace("module.", ""): v for k, v in pretrained_model['model_state_dict'].items()}

disrupt_model = UNet(3)
disrupt_model.load_state_dict(state_dict)
disrupt_model.cuda().eval()

ori_image = Image.open("./data/img_align_celeba/img_align_celeba/000001.jpg").convert("RGB")
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

ori_image = transform(ori_image).unsqueeze(0).cuda()

perturbations = disrupt_model(ori_image.cuda())
perturbed_images = ori_image + perturbations

final_image = tensor_to_image(perturbed_images.squeeze(0))

final_image.save('output_image.jpg')

def tensor_to_image(tensor):
    to_pil = ToPILImage()
    return to_pil(tensor.cpu().detach())
