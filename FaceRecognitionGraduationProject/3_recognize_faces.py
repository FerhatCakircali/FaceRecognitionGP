import cv2
import dlib
import numpy as np
import pickle

# Model ve shape predictor yollarını ayarla
model_path = 'dlib_face_recognition_resnet_model_v1.dat'
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Dlib modellerini yükle
model = dlib.face_recognition_model_v1(model_path)
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_detector = dlib.get_frontal_face_detector()

# Eğitimli SVM modelini ve etiketleri yükle
with open('face_recognition_model.pkl', 'rb') as f:
    clf, labels = pickle.load(f)

# Decision function için eşik değeri belirle

# VideoCapture nesnesi oluştur
video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_image = frame[y:y+h, x:x+w]

        if face_image.size > 0:
            shape = shape_predictor(rgb_frame, face)
            face_descriptor = model.compute_face_descriptor(rgb_frame, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)

            # Tahmin yap ve olasılık değerlerini al
            probabilities = clf.predict_proba(face_descriptor)[0]
            max_prob = max(probabilities)
            prediction = clf.classes_[np.argmax(probabilities)]
            p = max_prob*100

            if  p < 75:
                name = "Bilinmiyor"
                percentage = max_prob * 100
            else:
                name = prediction
                percentage = max_prob * 100

            label = f"{name} ({percentage:.2f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
