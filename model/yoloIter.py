from ultralytics import YOLO


class YOLOIterator:
    def __init__(self, modelPath="../weights/yolov8_best.pt"):
        self.model = YOLO(modelPath)


    def getBoxes(self, source):
        result = self.model.predict(source=source)
        xyxy = result[0].boxes.xyxy
        conf = result[0].boxes.conf
        return xyxy.cpu().numpy(), conf.cpu().numpy()  # [n, 4], [n]
