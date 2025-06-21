import face_recognition  
import cv2
import os
import pandas as pd
from datetime import datetime

known_encodings = []
student_names = []

for file in os.listdir('student_img'):
    img = face_recognition.load_image_file(f"student_img/{file}")
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_encodings.append(encodings[0])
        student_names.append(os.path.splitext(file)[0])
    else:
        print(f"⚠️ No face found in {file}")

input_image = face_recognition.load_image_file("group_img.jpg")
input_encodings = face_recognition.face_encodings(input_image)

if not input_encodings:
    print("⚠️ No faces found in group image.")
    exit()


present_students = []
for encoding in input_encodings:
    results = face_recognition.compare_faces(known_encodings, encoding)
    for i, match in enumerate(results):
        if match:
            present_students.append(student_names[i])


present_students = list(set(present_students))

all_students = set(student_names)
absent_students = all_students - set(present_students)

data = []
now = datetime.now().strftime('%Y-%m-%d')

for student in all_students:
    status = "Present" if student in present_students else "Absent"
    data.append({"Date": now, "Name": student, "Status": status})

df = pd.DataFrame(data)
df.to_csv("attendance.csv", index=False)
print("✅ Attendance marked successfully.")
