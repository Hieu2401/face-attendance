import cv2
import os

name = input("Nhập tên người dùng: ")
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
