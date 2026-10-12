import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create a synthetic image so the experiment runs without an external file.
    image = np.zeros((256,256), dtype=np.uint8)
    cv2.rectangle(image, (40,40), (216,216), 100, -1)
    cv2.circle(image, (128,128), 55, 230, -1)
    cv2.line(image, (0,0), (255,255), 255, 3)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = np.fft.fftshift(dft, axes=[0,1])
    magnitude = 20*np.log(cv2.magnitude(shifted[:,:,0], shifted[:,:,1]) + 1)
    rows, cols = image.shape; crow, ccol = rows//2, cols//2
    yy, xx = np.ogrid[:rows,:cols]
    radius = np.sqrt((yy-crow)**2 + (xx-ccol)**2)
    low_mask = (radius < 35).astype(np.float32)
    high_mask = 1.0 - low_mask
    def reconstruct(mask):
        filtered = shifted * mask[:,:,None]
        back = np.fft.ifftshift(filtered, axes=[0,1])
        return cv2.idft(back, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    low = reconstruct(low_mask); high = reconstruct(high_mask)
    fig, ax = plt.subplots(1,4,figsize=(13,4))
    for a, im, title in zip(ax, [image,magnitude,low,high], ["Original","Log Fourier magnitude","Low-pass result","High-pass result"]):
        a.imshow(im,cmap="gray"); a.set_title(title); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__": main()
