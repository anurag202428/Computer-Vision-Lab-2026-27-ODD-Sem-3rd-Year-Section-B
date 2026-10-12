"""Optional YOLO detection. Install with: python -m pip install ultralytics
Usage: python experiment9.py --image path/to/image.jpg --model yolov8n.pt
"""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True,help="Image file to detect objects in")
    p.add_argument("--model",default="yolov8n.pt",help="Ultralytics model weights")
    p.add_argument("--output",default="runs/detect",help="Output directory")
    args=p.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install YOLO dependency: python -m pip install ultralytics") from exc
    if not Path(args.image).is_file(): raise FileNotFoundError(args.image)
    model=YOLO(args.model)
    results=model.predict(source=args.image,save=True,project=args.output,name="experiment9",exist_ok=True)
    for result in results:
        names=result.names
        for box in result.boxes:
            cls=int(box.cls[0]); conf=float(box.conf[0])
            print(f"{names[cls]}: confidence={conf:.3f}")
    print(f"Annotated output saved under: {args.output}/experiment9")

if __name__=="__main__": main()
