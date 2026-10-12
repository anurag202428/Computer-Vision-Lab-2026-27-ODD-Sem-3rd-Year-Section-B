import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_image():
    im=np.zeros((240,320,3),dtype=np.uint8)
    im[:,:]=(35,35,35)
    cv2.circle(im,(95,120),65,(30,170,230),-1)
    cv2.rectangle(im,(175,55),(280,190),(220,80,70),-1)
    return im

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image"); args=p.parse_args()
    bgr=cv2.imread(args.image) if args.image else demo_image()
    if bgr is None: raise FileNotFoundError(args.image)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    pixels=np.float32(bgr.reshape((-1,3)))
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,labels,centers=cv2.kmeans(pixels,3,None,criteria,5,cv2.KMEANS_PP_CENTERS)
    segmented=np.uint8(centers)[labels.flatten()].reshape(bgr.shape)
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,im,t in zip(ax,[rgb,binary,cv2.cvtColor(segmented,cv2.COLOR_BGR2RGB)],["Original","Otsu thresholding","K-means (k=3)"]):
        a.imshow(im,cmap="gray" if im.ndim==2 else None); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__=="__main__": main()
