import cv2
import os
import time  # ➕ THÊM

name = input("Nhập tên người dùng (không dấu): ").strip()
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)
print(f"[✔] Thư mục lưu ảnh: {save_path}")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if cap.isOpened() and not face_cascade.empty():
    print("[✔] Webcam và cascade đã sẵn sàng.")
else:
    print("[❌] Lỗi webcam hoặc cascade.")
    exit()

count = 0
last_capture_time = 0  # ➕ THÊM

while True:
    ret, frame = cap.read()
    if not ret:
        print("[❌] Không lấy được hình.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ➕ Delay chụp mỗi 0.5 giây
    current_time = time.time()
    if len(faces) > 0 and current_time - last_capture_time >= 0.5:
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            filename = f"{save_path}/{count}.jpg"
            success = cv2.imwrite(filename, face_img)
            print(f"[📸] Lưu ảnh {count}: {filename}")
            last_capture_time = current_time

            # Vẽ khung
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if count >= 20:
                break

    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
