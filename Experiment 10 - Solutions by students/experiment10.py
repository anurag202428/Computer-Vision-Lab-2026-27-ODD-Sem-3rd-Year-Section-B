import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",help="Optional image path; webcam is not used")
    args=p.parse_args()
    if args.image:
        image=cv2.imread(args.image)
        if image is None: raise FileNotFoundError(args.image)
        source="image"
    else:
        # Synthetic demo makes setup testable, but contains no actual face.
        image=None; source="demo"
    cascade_path=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(cascade_path)
    if detector.empty(): raise RuntimeError("Could not load OpenCV Haar cascade.")
    if image is None:
        import numpy as np
        image=np.full((300,400,3),230,dtype=np.uint8)
        cv2.putText(image,"Provide a real photo with --image", (15,150),cv2.FONT_HERSHEY_SIMPLEX,0.65,(20,20,20),2)
        print("Demo image has no face; run with --image path/to/photo.jpg to test detection.")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces: cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    print("Faces detected:",len(faces))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); plt.title(f"Face detection ({source})"); plt.axis("off"); plt.show()

if __name__=="__main__": main()
