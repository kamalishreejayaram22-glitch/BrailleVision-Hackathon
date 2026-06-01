from ultralytics import YOLO
from gtts import gTTS
import cv2
import numpy as np
import os
model = YOLO(
    "models/best.pt"
)
CLASS_NAMES = [
'a','b','c','d','e','f','g','h','i','j','k','l','m',
'n','o','p','q','r','s','t','u','v','w','x','y','z'
]
os.makedirs(
    "sample_outputs",
    exist_ok=True
)
def sort_cells(boxes):
    if len(boxes)==0:
        return []
    cells=[]
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        cells.append({
            "idx":i,
            "cx":(x1+x2)/2,
            "cy":(y1+y2)/2,
            "h":y2-y1
        })
    cells.sort(
        key=lambda x:x["cy"]
    )
    rows=[]
    while cells:
        current=cells.pop(0)
        row=[current]
        remain=[]
        threshold=current["h"]*0.60
        for c in cells:
            if abs(
                c["cy"]
                -
                current["cy"]
            )<=threshold:
                row.append(c)
            else:
                remain.append(c)
        row.sort(
            key=lambda x:x["cx"]
        )
        rows.append(row)
        cells=remain
    return [
        c["idx"]
        for r in rows
        for c in r
    ]
def preprocess(img):
    gray=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )
    blur=cv2.GaussianBlur(
        gray,
        (51,51),
        0
    )
    corrected=cv2.divide(
        gray,
        blur,
        scale=255
    )
    clahe=cv2.createCLAHE(
        clipLimit=2,
        tileGridSize=(8,8)
    )
    enhanced=clahe.apply(
        corrected
    )
    kernel=np.array([
        [0,-0.5,0],
        [-0.5,3,-0.5],
        [0,-0.5,0]
    ])
    sharp=cv2.filter2D(
        enhanced,
        -1,
        kernel
    )
    return cv2.cvtColor(
        sharp,
        cv2.COLOR_GRAY2BGR
    )
def detect(img):
    img=preprocess(img)
    best=None
    count=0
    for conf in [
        0.3,
        0.2,
        0.1,
        0.05
    ]:
        result=model(
            img,
            conf=conf,
            iou=0.3
        )[0]
        boxes=result.boxes.xyxy.cpu().numpy()
        cls=(
            result.boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )
        if len(boxes)>count:
            count=len(boxes)
            best=(
                result,
                boxes,
                cls
            )
    if best:
        result,boxes,cls=best
        order=sort_cells(
            boxes
        )
        word="".join(
            CLASS_NAMES[
                cls[i]
            ]
            for i in order
        )
        return (
            result.plot(),
            word
        )
    return None,""
def speak(text,ts):
    path=(
        f"sample_outputs/"
        f"speech_{ts}.mp3"
    )
    gTTS(
        text=text,
        lang="en"
    ).save(path)
    return path
