import face_recognition
import os
import pickle

path = 'dataset'
known_encodings = []
known_names = []

for name in os.listdir(path):
    for file in os.listdir(f"{path}/{name}"):
        image = face_recognition.load_image_file(f"{path}/{name}/{file}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

with open('encodings.pickle', 'wb') as f:
    pickle.dump((known_encodings, known_names), f)
print("✅ Mã hóa hoàn tất. Đã lưu encodings.pickle")
input("Nhấn Enter để thoát...")