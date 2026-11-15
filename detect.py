from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")   # auto-downloads on first run (~5MB)

cap = cv2.VideoCapture(0)    # 0 = your laptop webcam
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    results = model.predict(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()