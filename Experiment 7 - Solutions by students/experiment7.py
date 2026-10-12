"""Optional CNN experiment using scikit-learn's built-in handwritten digits dataset.
TensorFlow is required: python -m pip install tensorflow
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError as exc:
        raise SystemExit("Install TensorFlow first: python -m pip install tensorflow") from exc
    digits=load_digits()
    X=digits.images.astype("float32")/16.0
    X=X[...,np.newaxis]
    y=digits.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model=models.Sequential([
        layers.Input(shape=(8,8,1)),
        layers.Conv2D(16,(3,3),activation="relu",padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32,activation="relu"),
        layers.Dense(10,activation="softmax")
    ])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(X_train,y_train,epochs=8,batch_size=32,validation_split=0.1,verbose=1)
    loss,acc=model.evaluate(X_test,y_test,verbose=0)
    print(f"Test accuracy: {acc:.3f}")
    pred=np.argmax(model.predict(X_test,verbose=0),axis=1)
    print(classification_report(y_test,pred))
    fig,ax=plt.subplots(1,5,figsize=(10,2.5))
    for a,img,label,guess in zip(ax,X_test[:5],y_test[:5],pred[:5]):
        a.imshow(img.squeeze(),cmap="gray"); a.set_title(f"Actual {label}\\nPred {guess}"); a.axis("off")
    plt.tight_layout(); plt.show()

if __name__=="__main__": main()
