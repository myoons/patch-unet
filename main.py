import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset

from models.WordSegmenter import WordSegmenter


class PatchDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self. labels = labels
        assert len(self.images) == len(self.labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == '__main__':
    model = WordSegmenter(3, 1, True).cuda()
    
    images = glob('dataset/images/*.png')
    split = int(len(images) * 0.95)
    tr_images, tr_labels = [], []
    for image in tqdm(images[:split]):
        label = image.replace("images", "labels")
        tr_images.append(np.transpose(cv2.imread(image), (2, 0, 1)) / 255)
        tr_labels.append(cv2.imread(label), flags=cv2.IMREAD_GRAYSCALE / 255)

    eval_images, eval_labels = [], []
    for image in tqdm(images[split:]):
        label = image.replace("images", "labels")
        eval_images.append(np.transpose(cv2.imread(image), (2, 0, 1)) / 255)
        eval_labels.append(cv2.imread(label), flags=cv2.IMREAD_GRAYSCALE / 255)

    # Dataset & Dataloader
    train_dataset = PatchDataset(tr_images, tr_labels)
    eval_dataset = PatchDataset(eval_images, eval_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6, persistent_workers=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=6, persistent_workers=True)

    # Criterion & Optimizer & Scheduler
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    max_loss = 1000.
    model.train()
    for i in tqdm(range(500)):
        for (images, labels) in train_dataloader:
            optimizer.zero_grad()

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            eval_loss = []
            for (images, labels) in eval_dataloader:

                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                eval_loss.append(loss.item())


        if np.mean(eval_loss) < max_loss:
            max_loss = np.mean(eval_loss)
            torch.save(model.load_state_dict(), 'best.pt')