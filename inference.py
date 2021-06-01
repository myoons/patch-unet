import cv2
import torch
import numpy as np
from models.WordSegmenter import WordSegmenter
from data_utils.make_dataset import rescale_pad_image


if __name__ == '__main__':
    model = WordSegmenter(3, 1, True)
    model.load_state_dict(torch.load('best.pt', map_location='cpu'))

    term = 128
    raw_image = cv2.imread('test.jpg')
    pad_image, pad_label, _ = rescale_pad_image(raw_image, None, term)

    height, width = pad_image.shape[:2]
    hq, wq = height // term, width // term
    tile_images = []
    for y in range(0, height, term):
        for x in range(0, width, term):
            tile_images.append(np.transpose(pad_image[y:y+term, x:x+term, :], (2, 0, 1)))

    
    tile_images = np.array(tile_images)
    images = torch.from_numpy(tile_images).float() / 255
    with torch.no_grad():
        logits = model(images).permute(0, 2, 3, 1)
    pred = np.clip(logits.detach().numpy(), 0, 1) * 255
    pred = np.uint8(logits)

    canvas = np.ones((height, width)) * 255
    count = 0
    for y in range(0, height, term):
        for x in range(0, width, term):
            canvas[y:y+term, x:x+term] = np.reshape(pred[count, :, :], (term, term))
            count += 1

    cv2.imwrite('mask.png', canvas)