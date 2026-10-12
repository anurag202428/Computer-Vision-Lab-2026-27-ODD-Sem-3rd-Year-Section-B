import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (180, 220), (40, 180, 240), -1)
    cv2.circle(img, (270, 130), 65, (220, 80, 80), -1)
    cv2.putText(img, "CV Lab", (120, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional path to an image")
    args = parser.parse_args()
    image = cv2.imread(args.image) if args.image else demo_image()
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"Width={w}, Height={h}, Channels={image.shape[2] if image.ndim == 3 else 1}")
    resized = cv2.resize(rgb, (w // 2, h // 2))
    flipped = cv2.flip(rgb, 1)
    matrix = cv2.getRotationMatrix2D((w/2, h/2), 25, 1.0)
    rotated = cv2.warpAffine(rgb, matrix, (w, h))
    translated = cv2.warpAffine(rgb, np.float32([[1,0,40],[0,1,25]]), (w,h))
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for a, im, title in zip(ax.ravel(), [rgb, resized, flipped, rotated, translated, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)],
                            ["Original", "Resized", "Horizontal flip", "Rotated 25°", "Translated", "Grayscale"]):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(title); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
