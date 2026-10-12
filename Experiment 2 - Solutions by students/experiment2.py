import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_image():
    rng = np.random.default_rng(7)
    base = np.tile(np.linspace(40, 150, 320, dtype=np.uint8), (220, 1))
    noise = rng.normal(0, 12, base.shape)
    return np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional path to grayscale image")
    args = parser.parse_args()
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE) if args.image else demo_image()
    if gray is None: raise FileNotFoundError(args.image)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for a, im, title in zip(ax[0], [gray, equalized, clahe], ["Original", "Histogram equalization", "CLAHE"]):
        a.imshow(im, cmap="gray"); a.set_title(title); a.axis("off")
    for a, im, title in zip(ax[1], [gray, equalized, clahe], ["Original histogram", "Equalized histogram", "CLAHE histogram"]):
        a.hist(im.ravel(), 256, range=(0,256)); a.set_title(title); a.set_xlim(0,255)
    plt.tight_layout(); plt.show()

if __name__ == "__main__": main()
