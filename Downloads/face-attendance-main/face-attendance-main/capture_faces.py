import cv2
import os

name = input("Nhập tên nhân viên : ").strip()
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
while count < 20:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1)
    if key == ord('c'):
        count += 1
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        print(f"Ảnh {count} đã lưu.")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()