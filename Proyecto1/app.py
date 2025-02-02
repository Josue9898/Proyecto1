from flask import Flask, jsonify, render_template, Response
import threading
import detector
import cv2

app = Flask(__name__)

detector_running = False

@app.route('/api/cars', methods=['GET'])
def get_car_count():
    """Devuelve la cantidad de autos detectados en JSON."""
    return jsonify({"car_count": detector.car_count})

@app.route('/')
def index():
    """Carga la página web."""
    return render_template("index.html")

def start_detector():
    """Inicia la detección de autos en un hilo separado solo si no está corriendo."""
    global detector_running
    if not detector_running:
        print("Iniciando detección de autos...")
        detector_running = True
        t = threading.Thread(target=detector.capture_and_process, daemon=True)
        t.start()

def generate_frames():
    """Captura frames de la cámara ESP32-CAM y los envía a la web."""
    while True:
        cap = cv2.VideoCapture(detector.ESP32_URL)
        success, frame = cap.read()
        cap.release()  # Cerrar la conexión inmediatamente después de capturar

        if not success:
            print("Error obteniendo frame para /video_feed")
            continue
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Ruta que devuelve el video en vivo como stream de imágenes."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    start_detector()
    app.run(host='0.0.0.0', port=5000, debug=True)
