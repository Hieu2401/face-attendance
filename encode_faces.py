import face_recognition
import os
import pickle

dataset = 'dataset'
encodings = []
names = []

for name in os.listdir(dataset):
    folder = os.path.join(dataset, name)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(image)
        if enc:
            encodings.append(enc[0])
            names.append(name)

with open("encodings.pickle", "wb") as f:
    pickle.dump((encodings, names), f)

print("✅ Mã hóa khuôn mặt thành công.")