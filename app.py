from flask import Flask, render_template, Response, request, redirect, url_for
import face_recognition
import cv2
import pickle
import pandas as pd
from datetime import datetime
import os
import time
import threading
from playsound import playsound
import numpy as np

app = Flask(__name__)

# Load known face encodings
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

attendance = set()
last_unrecognized_time = 0

# Phát âm thanh không chặn tiến trình chính
def play_sound(file):
    threading.Thread(target=playsound, args=(file,), daemon=True).start()

# Xử lý khung hình từ camera
def gen_frames():
    global last_unrecognized_time
    cap = cv2.VideoCapture(0)

    last_thanked_time = {}
    last_seen_name = None
    seen_interval = 5  # giây chờ để phát lại âm thanh cho cùng một người

    while True:
        success, frame = cap.read()
        if not success:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)
        current_time = time.time()

        for enc, loc in zip(encs, faces):
            name = "Unknown"
            distances = face_recognition.face_distance(known_encodings, enc)

            if len(distances) > 0:
                min_distance = np.min(distances)
                best_match_index = np.argmin(distances)

                if min_distance < 0.45:
                    matched_name = known_names[best_match_index]
                    name = matched_name

                    should_thank = (
                        name != last_seen_name or
                        current_time - last_thanked_time.get(name, 0) > seen_interval
                    )

                    if should_thank:
                        play_sound("static/sounds/thanks.mp3")
                        last_thanked_time[name] = current_time

                    if name not in attendance:
                        attendance.add(name)
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        df = pd.DataFrame([[name, time_str]], columns=["Name", "Time"])
                        df.to_csv("attendance.csv", mode='a', header=not os.path.exists("attendance.csv"), index=False)
                else:
                    if current_time - last_unrecognized_time > 5:
                        play_sound("static/sounds/not_found.mp3")
                        last_unrecognized_time = current_time

            top, right, bottom, left = [v * 4 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_seen_name = name

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Giao diện điểm danh (webcam)
@app.route('/attendance')
def attendance_page():
    return render_template('attendance_page.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Trang xem bảng điểm danh
@app.route('/admin')
def admin_page():
    try:
        df = pd.read_csv('attendance.csv', header=None, names=["Name", "Time"])
        records = df.to_dict(orient='records')
    except FileNotFoundError:
        records = []
    return render_template('attendance.html', records=records)

# Xoá một dòng điểm danh
@app.route('/delete', methods=['POST'])
def delete_record():
    name = request.form['name']
    time_str = request.form['time']

    df = pd.read_csv('attendance.csv', header=None, names=["Name", "Time"])
    df = df[~((df["Name"] == name) & (df["Time"] == time_str))]
    df.to_csv('attendance.csv', index=False, header=False)

    return redirect(url_for('admin_page'))

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
