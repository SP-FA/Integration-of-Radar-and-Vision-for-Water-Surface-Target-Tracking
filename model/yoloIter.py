import numpy as np
from ultralytics import YOLO


class YOLOIterator:
    def __init__(self, modelPath="../weights/yolov8_best.pt"):
        self.model = YOLO(modelPath)


    def getBoxes(self, source):
        result = self.model.track(source=source, verbose=False)
        result = result[0].boxes

        if result.id is not None:
            xyxy = result.xyxy.cpu().numpy()
            cls = result.cls.cpu().numpy()
            id = result.id.cpu().numpy()
            conf = result.conf.cpu().numpy()
        else:
            return None, None, None, None

        sortedIndex = np.argsort(conf)[::-1]
        xyxy = xyxy[sortedIndex]
        cls = cls[sortedIndex]
        id = id[sortedIndex]
        conf = conf[sortedIndex]
        return xyxy, cls, id, conf   # [n, 4], [n]


if __name__ == "__main__":
    iter = YOLOIterator()
    iter.getBoxes("../yolov8/runs/detect/predict2/1669621792.47048.jpg")
