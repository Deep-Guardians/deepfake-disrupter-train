from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2

def create_dataloader(hp, data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([hp.data.image_size, hp.data.image_size])
    ])

    with open(data_path + '/train_set_images.ctrl', 'r') as f:
        train_image_paths = [line.strip() for line in f if line.strip()]

    # test_set_images.ctrl 파일에서 경로 읽기
    with open(data_path + '/test_set_images.ctrl', 'r') as f:
        test_image_paths = [line.strip() for line in f if line.strip()]

    # CustomDataset 인스턴스 생성
    train_set = CustomDataset(train_image_paths, transform)
    test_set = CustomDataset(test_image_paths, transform)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=hp.train.batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image
