import cv2
import numpy as np
import time
import os

ESP32_URL = "http://192.168.137.253/cam-lo.jpg"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG = os.path.join(BASE_DIR, "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov3.weights")
YOLO_LABELS = os.path.join(BASE_DIR, "coco.names")

for file in [YOLO_CFG, YOLO_WEIGHTS, YOLO_LABELS]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Archivo {file} no encontrado. Verifica la ruta.")

with open(YOLO_LABELS, "r") as f:
    classes = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

car_count = 0  # Variable global para el contador de autos


def detect_objects(frame):
    global car_count
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    car_count = len(indices.flatten()) if len(indices) > 0 else 0
    return car_count


def capture_and_process():
    global car_count
    cap = cv2.VideoCapture(ESP32_URL)

    if not cap.isOpened():
        print("Error: No se pudo acceder al ESP32-CAM. Verifica la conexión.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la transmisión del ESP32-CAM. Intentando reconectar...")
            time.sleep(10)
            cap = cv2.VideoCapture(ESP32_URL)  # Intentar reconectar
            continue

        try:
            car_count = detect_objects(frame)

            cv2.putText(frame, f"Carros detectados: {car_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Car Detection', frame)
            print(f"Autos detectados en esta captura: {car_count}")

        except Exception as e:
            print(f"Error en detección de objetos: {e}")

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        time.sleep(10)

    cap.release()
    cv2.destroyAllWindows()

