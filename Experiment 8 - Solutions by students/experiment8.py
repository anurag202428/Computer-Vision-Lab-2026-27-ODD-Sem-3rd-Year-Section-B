import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_frame(shift):
    frame=np.zeros((240,320),dtype=np.uint8)
    cv2.rectangle(frame,(60+shift,70),(125+shift,135),200,-1)
    cv2.circle(frame,(220-shift,120),30,255,-1)
    return frame

def main():
    previous=make_frame(0); current=make_frame(12)
    flow=cv2.calcOpticalFlowFarneback(previous,current,None,0.5,3,15,3,5,1.2,0)
    mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
    step=12
    y,x=np.mgrid[step//2:previous.shape[0]:step,step//2:previous.shape[1]:step].reshape(2,-1).astype(int)
    fx,fy=flow[y,x].T
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    ax[0].imshow(previous,cmap="gray"); ax[0].set_title("Previous frame")
    ax[1].imshow(current,cmap="gray"); ax[1].set_title("Current frame")
    ax[2].imshow(current,cmap="gray"); ax[2].quiver(x,y,fx,fy,color="red",angles="xy",scale_units="xy",scale=1); ax[2].set_title("Dense optical flow")
    for a in ax: a.axis("off")
    print("Mean flow magnitude:", float(np.mean(mag)))
    plt.tight_layout(); plt.show()

if __name__=="__main__": main()
