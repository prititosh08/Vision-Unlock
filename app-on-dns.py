from flask import Flask, render_template, Response, send_from_directory
import cv2
import numpy as np
import os
import time
import requests
from zeroconf import Zeroconf, ServiceInfo
import socket

app = Flask(__name__)
ESP_HOST = "http://esp32.local"

# Load classifier and trained model
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Storage for intruder images
INTRUDER_FOLDER = "static/intruder_images"
os.makedirs(INTRUDER_FOLDER, exist_ok=True)

def delete_intruder_images():
    for file in os.listdir(INTRUDER_FOLDER):
        os.remove(os.path.join(INTRUDER_FOLDER, file))

last_status = None
cooldown = 7  # Cooldown in seconds
last_sent_time = time.time()

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    global last_status, last_sent_time
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    current_status = "Unknown"
    for (x, y, w, h) in features:
        box_color = color
        text_color = color

        id, pred = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 75:
            if id == 1:
                current_status = "legend"
                cv2.putText(img, "legend", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                current_status = "Manish"
                cv2.putText(img, "Manish", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            current_status = "Unknown"
            box_color = (0, 0, 255)
            text_color = (0, 0, 255)
            cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

        cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)

    # Handle access detection
    if current_status != last_status and time.time() - last_sent_time > cooldown:
        last_status = current_status
        last_sent_time = time.time()

        if current_status == "legend" or current_status == "Manish":
            print(f"✅ Access Granted: Welcome, {current_status}!")
            
        else:
            print("⛔ Access Denied: Unknown person detected!")
            timestamp = int(time.time())
            img_path = os.path.join(INTRUDER_FOLDER, f"intruder_{timestamp}.jpg")
            cv2.imwrite(img_path, img)

    return img

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        success, img = video_capture.read()
        if not success:
            break

        fps = int(1 / (time.time() - start_time))
        start_time = time.time()

        img = draw_boundary(img, faceCascade, 1.3, 6, (0, 255, 0), clf)
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/admin')
def admin():
    images = os.listdir(INTRUDER_FOLDER)
    return render_template('adminpg.html', images=images)

@app.route('/static/intruder_images/<filename>')
def get_intruder_image(filename):
    return send_from_directory(INTRUDER_FOLDER, filename)

@app.route("/lock")
def lock():
    return render_template("anim.html")

@app.route("/buzzer")
def buzzer():
    requests.get(f"{ESP_HOST}/buzzer")
    return "Buzzer Activated (Auto-Off in 4 sec)"

@app.route('/live')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


if __name__ == '__main__':
    ip = get_ip()
    
    print(" link for other devices : \n")
    print(" http://vision-unlock.local:5000 \n")
    
    
    info = ServiceInfo(
        "_http._tcp.local.",
        "vision-unlock._http._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=5000,
        properties={},
        server="vision-unlock.local."
    )

    zeroconf = Zeroconf()
    zeroconf.register_service(info)

    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()

