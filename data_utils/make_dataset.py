import cv2
import os
from tqdm import tqdm
from glob import glob


def rescale_pad_image(image, label, term=128):
    shape = image.shape[:2]
    hpad, wpad = (128 - shape[0] % term) / 2, (128 - shape[1] % term) / 2
    top, bottom = int(round(hpad - 0.1)), int(round(hpad + 0.1))
    left, right = int(round(wpad - 0.1)), int(round(wpad + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(177, 177, 177))
    label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return image, label, (top, bottom, left, right)


if __name__ == '__main__':
    ROOT_DIR = '/data/flatter/images/korea'
    COUNT = 0
    for f in glob(f'{ROOT_DIR}/*'):
        for image in tqdm(glob(f'{f}/*.png'), total=len(glob(f'{f}/*.png'))):
            label = image.replace("images", "labels")

            raw_image = cv2.imread(image)
            raw_label = cv2.imread(label, flags=cv2.IMREAD_GRAYSCALE)
            assert raw_image.shape[:2] == raw_label.shape

            term = 128
            pad_image, pad_label, _ = rescale_pad_image(raw_image, raw_label, term)

            height, width = pad_image.shape[:2]
            hq, wq = height // term, width // term

            tile_images, tile_labels = [], []
            for y in range(0, height, term):
                for x in range(0, width, term):
                    cv2.imwrite(f'dataset/images/{os.path.basename(image)[:-4]}_{str(COUNT).zfill(5)}.png', pad_image[y:y+term, x:x+term, :])
                    cv2.imwrite(f'dataset/labels/{os.path.basename(image)[:-4]}_{str(COUNT).zfill(5)}.png', pad_label[y:y+term, x:x+term])
                    COUNT += 1
