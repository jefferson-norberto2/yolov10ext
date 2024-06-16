from ultralytics import YOLOv10
import cv2

model = YOLOv10('best2.pt')

image = cv2.imread('bisturi.png')

if image.any() != None:
    results = model.predict(source=image, imgsz=640, conf=0.25)
    annotated_image = results[0].plot()
    annotated_image = annotated_image[:, :, ::-1]

cv2.imshow('result', annotated_image)
cv2.waitKey(0)