# Các thư viện cần thiết
from flask import Flask, render_template, Response, request, redirect, url_for, send_file, session
import face_recognition  # Thư viện nhận diện khuôn mặt
import cv2  # Xử lý hình ảnh từ webcam
import pickle  # Đọc file mã hóa khuôn mặt
import pandas as pd  # Xử lý file CSV
from datetime import datetime
import os
import time
import threading  # Dùng để phát âm thanh không chặn giao diện
import numpy as np
import pygame  # Thư viện phát âm thanh

# Khởi tạo ứng dụng Flask và cấu hình session
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Khóa bảo mật phiên làm việc
pygame.mixer.init()  # Khởi động pygame để phát âm thanh

# Load dữ liệu khuôn mặt đã mã hóa từ file
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

last_unrecognized_time = 0  # Thời gian lần cuối nhận diện không thành công

# Hàm phát âm thanh không đồng bộ (chạy song song)
def play_sound(file):
    def _play():
        if os.path.exists(file):
            try:
                sound = pygame.mixer.Sound(file)
                sound.play()
            except Exception as e:
                print(f"Lỗi phát âm thanh: {e}")
    threading.Thread(target=_play, daemon=True).start()

# Hàm xử lý nhận diện khuôn mặt từ webcam
def gen_frames():
    global last_unrecognized_time
    cap = cv2.VideoCapture(0)  # Mở webcam
    last_seen_name = None
    last_thanked_time = {}
    seen_interval = 5  # Khoảng cách thời gian để tránh phát âm lặp

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize và chuyển màu ảnh để xử lý nhanh hơn
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Nhận diện khuôn mặt và mã hóa
        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)
        current_time = time.time()

        for enc, loc in zip(encs, faces):
            name = "Unknown"
            distances = face_recognition.face_distance(known_encodings, enc)

            if len(distances) > 0:
                min_distance = np.min(distances)
                best_match_index = np.argmin(distances)

                # Nếu độ tương đồng tốt => nhận diện được
                if min_distance < 0.45:
                    name = known_names[best_match_index]

                    # Phát âm thanh nếu chưa vừa mới phát
                    should_thank = (
                        name != last_seen_name or
                        current_time - last_thanked_time.get(name, 0) > seen_interval
                    )
                    if should_thank:
                        play_sound("static/sounds/thanks.mp3")
                        last_thanked_time[name] = current_time

                    # Lưu thời gian điểm danh vào file CSV
                    today = datetime.now().strftime("%Y-%m-%d")
                    now_time = datetime.now().strftime("%H:%M:%S")
                    filename = f"attendance_{today}.csv"

                    if not os.path.exists(filename):
                        df = pd.DataFrame(columns=["Name", "Date", "Checkin", "Checkout"])
                    else:
                        df = pd.read_csv(filename)

                    match = df[(df['Name'] == name) & (df['Date'] == today)]

                    if match.empty:
                        # Chưa có bản ghi => Check-in
                        df.loc[len(df)] = [name, today, now_time, ""]
                        print(f"[CHECKIN] {name} tại {now_time}")
                    else:
                        # Đã có => Cập nhật Check-out (nếu muộn hơn)
                        index = match.index[0]
                        existing_checkout = match.iloc[0]['Checkout']
                        if pd.isna(existing_checkout) or existing_checkout == "" or pd.to_datetime(now_time) > pd.to_datetime(existing_checkout):
                            df.at[index, 'Checkout'] = now_time
                            print(f"[CHECKOUT] {name} cập nhật tại {now_time}")

                    df.to_csv(filename, index=False)

                else:
                    # Nhận diện thất bại
                    if current_time - last_unrecognized_time > 5:
                        play_sound("static/sounds/not_found.mp3")
                        last_unrecognized_time = current_time
                        print("[CẢNH BÁO] Không nhận diện được.")

            # Vẽ khung và tên người trên khung hình
            top, right, bottom, left = [v * 4 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_seen_name = name

        # Trả về từng khung hình dạng MJPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Trang đăng nhập quản trị
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == '1234':
            session['just_logged_in'] = True  # chỉ cho phép 1 lần truy cập
            return redirect(url_for('admin_page'))
        else:
            return render_template('login.html', error="Sai mật khẩu")
    return render_template('login.html')

# Đăng xuất
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Trang hiển thị camera điểm danh
@app.route('/attendance')
def attendance_page():
    return render_template('attendance_page.html')

# Stream video từ camera
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Trang quản trị điểm danh
@app.route('/admin')
def admin_page():
    # Yêu cầu nhập mật khẩu mỗi lần vào
    if not session.pop('just_logged_in', False):
        return redirect(url_for('login'))

    # Lọc dữ liệu theo ngày và tên
    date_filter = request.args.get("date")
    name_filter = request.args.get("name", "").strip().lower()
    all_records = []

    # Lặp qua các file attendance_*.csv để lấy dữ liệu
    for file in os.listdir():
        if file.startswith("attendance_") and file.endswith(".csv"):
            df = pd.read_csv(file)
            df['Late'] = df['Checkin'].apply(lambda x: pd.to_datetime(x, errors='coerce') > pd.to_datetime("08:00:00"))
            df['Early'] = df['Checkout'].apply(lambda x: pd.to_datetime(x, errors='coerce') < pd.to_datetime("17:00:00"))
            df['Late'] = df['Late'].fillna(False)
            df['Early'] = df['Early'].fillna(False)
            all_records.extend(df.to_dict(orient='records'))

    # Lọc theo ngày và tên nếu có
    if date_filter:
        all_records = [r for r in all_records if r['Date'] == date_filter]
    if name_filter:
        all_records = [r for r in all_records if name_filter in r['Name'].lower()]

    return render_template("attendance.html", records=all_records, date_filter=date_filter, name_filter=name_filter)

# Xoá bản ghi điểm danh
@app.route('/delete', methods=['POST'])
def delete_record():
    name = request.form['name']
    date = request.form['date']
    filename = f"attendance_{date}.csv"

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df['Name'] != name]
        df.to_csv(filename, index=False)

    return redirect(url_for('admin_page', date=date))

# Xuất file CSV điểm danh trong ngày
@app.route('/export')
def export():
    date_filter = request.args.get("date")
    filename = f"attendance_{date_filter}.csv"
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "Không có dữ liệu để xuất."

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
