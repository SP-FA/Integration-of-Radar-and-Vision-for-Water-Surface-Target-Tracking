import numpy as np
from ultralytics import YOLO


class YOLOIterator:
    def __init__(self, modelPath="../weights/yolov8_best.pt"):
        self.model = YOLO(modelPath)

    def getBoxes(self, source):
        result = self.model.track(source=source, verbose=False)
        result = result[0].boxes

        if result.id is not None:
            xyxy = result.xyxy
            cls = result.cls
            id = result.id
            conf = result.conf

            sortedIndex = conf.argsort(descending=True)
            xyxy = xyxy[sortedIndex]
            cls = cls[sortedIndex]
            id = id[sortedIndex]
            conf = conf[sortedIndex]
            return xyxy, cls, id, conf   # [n, 4], [n]
        else:
            return None, None, None, None
