import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def demo_image():
    img = np.zeros((240,320), dtype=np.uint8)
    cv2.rectangle(img, (35,35), (130,190), 180, -1)
    cv2.circle(img, (225,100), 55, 255, -1)
    cv2.line(img, (20,220), (300,220), 120, 4)
    return img

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image"); args=p.parse_args()
    gray=cv2.imread(args.image, cv2.IMREAD_GRAYSCALE) if args.image else demo_image()
    if gray is None: raise FileNotFoundError(args.image)
    # SIFT availability depends on OpenCV build/version.
    sift_view=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if hasattr(cv2, "SIFT_create"):
        sift=cv2.SIFT_create(); keypoints, descriptors=sift.detectAndCompute(gray,None)
        sift_view=cv2.drawKeypoints(sift_view,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("SIFT keypoints:", len(keypoints), "| descriptor shape:", None if descriptors is None else descriptors.shape)
    else:
        print("SIFT unavailable in this OpenCV build; HOG will still run.")
    features, hog_image=hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                            visualize=True, feature_vector=True)
    hog_image=exposure.rescale_intensity(hog_image, in_range=(0,10))
    print("HOG descriptor length:", len(features))
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,im,t in zip(ax,[gray,sift_view,hog_image],["Input","SIFT keypoints","HOG visualization"]):
        a.imshow(im,cmap="gray" if im.ndim==2 else None); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__=="__main__": main()
