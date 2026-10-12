import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_image():
    im = np.zeros((260, 360), dtype=np.uint8)
    cv2.rectangle(im, (40,40), (150,190), 150, -1)
    cv2.circle(im, (250,125), 65, 230, -1)
    im = cv2.GaussianBlur(im, (5,5), 0)
    return im

def main():
    p = argparse.ArgumentParser(); p.add_argument("--image"); args = p.parse_args()
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE) if args.image else demo_image()
    if gray is None: raise FileNotFoundError(args.image)
    mean = cv2.blur(gray, (5,5))
    gaussian = cv2.GaussianBlur(gray, (5,5), 0)
    median = cv2.medianBlur(gray, 5)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    canny = cv2.Canny(gray, 80, 160)
    images = [gray, mean, gaussian, median, sobel, canny]
    titles = ["Original", "Mean blur", "Gaussian blur", "Median blur", "Sobel edges", "Canny edges"]
    fig, ax = plt.subplots(2,3,figsize=(11,7))
    for a, im, t in zip(ax.ravel(), images, titles): a.imshow(im,cmap="gray"); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__": main()
