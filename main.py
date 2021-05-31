import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from models.WordSegmenter import WordSegmenter


def rescale_pad_image(image, label, term=128):
    shape = image.shape[:2]
    hpad, wpad = (128 - shape[0] % term) / 2, (128 - shape[1] % term) / 2
    top, bottom = int(round(hpad - 0.1)), int(round(hpad + 0.1))
    left, right = int(round(wpad - 0.1)), int(round(wpad + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return image, label, (top, bottom, left, right)


if __name__ == '__main__':
    model = WordSegmenter(3, 1, True)

    raw_image = cv2.imread('data/test.png')
    raw_label = cv2.imread('data/label.png', flags=cv2.IMREAD_GRAYSCALE)
    assert raw_image.shape[:2] == raw_label.shape

    term = 128
    pad_image, pad_label, _ = rescale_pad_image(raw_image, raw_label, term)

    height, width = pad_image.shape[:2]
    hq, wq = height // term, width // term

    tile_images, tile_labels = [], []
    for y in range(0, height, term):
        for x in range(0, width, term):
            tile_images.append(np.transpose(pad_image[y:y+term, x:x+term, :], (2, 0, 1)))
            tile_labels.append(pad_label[y:y+term, x:x+term])

    # Criterion & Optimizer & Scheduler
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    model.train()
    tile_images = torch.from_numpy(np.array(tile_images)).float() / 255
    tile_labels = torch.from_numpy(np.array(tile_labels)) / 255
    for i in tqdm(range(500)):
        optimizer.zero_grad()
        tile_logits = model(tile_images).squeeze(1)
        loss = criterion(tile_logits, tile_labels)
        loss.backward()
        optimizer.steip()

    logits = model(tile_images).squeeze(1)
    pred = np.array(torch.sigmoid(logits).detach())
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

    count = 0
    canvas = np.ones((height, width)) * 255
    for y in range(0, height, term):
        for x in range(0, width, term):
            canvas[y:y+term, x:x+term] = pred[count]
            count += 1

    cv2.imwrite('result.png', canvas)