import cv2
import dlib
import math

# 顔と笑顔の検出器を作成
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# カメラから映像を取得
cap = cv2.VideoCapture(0)

while True:
    # フレームを読み込む
    ret, frame = cap.read()
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # 顔ごとに処理
    for (x, y, w, h) in faces:
        # 顔の領域を切り出す
        face_roi = gray[y:y+h, x:x+w]
        # 笑顔を検出
        smiles = smile_detector.detectMultiScale(face_roi, 1.8, 20)
        # 笑顔があれば
        if len(smiles) > 0:
            # 顔のランドマークを検出
            landmarks = landmark_detector(gray, dlib.rectangle(x, y, x+w, y+h))
            # 口角の座標を取得
            left_mouth_corner = (landmarks.part(48).x, landmarks.part(48).y)
            right_mouth_corner = (landmarks.part(54).x, landmarks.part(54).y)
            # 目の座標を取得
            left_eye_top = (landmarks.part(37).x, landmarks.part(37).y)
            left_eye_bottom = (landmarks.part(41).x, landmarks.part(41).y)
            right_eye_top = (landmarks.part(43).x, landmarks.part(43).y)
            right_eye_bottom = (landmarks.part(47).x, landmarks.part(47).y)
            # 口角の距離を計算
            mouth_width = math.sqrt((left_mouth_corner[0] - right_mouth_corner[0])**2 + (left_mouth_corner[1] - right_mouth_corner[1])**2)
            # 目の開き具合を計算
            left_eye_height = math.sqrt((left_eye_top[0] - left_eye_bottom[0])**2 + (left_eye_top[1] - left_eye_bottom[1])**2)
            right_eye_height = math.sqrt((right_eye_top[0] - right_eye_bottom[0])**2 + (right_eye_top[1] - right_eye_bottom[1])**2)
            eye_height = (left_eye_height + right_eye_height) / 2
            # 笑顔の度合いを計算（口角の距離と目の開き具合の比）
            smile_ratio = mouth_width / eye_height
            # 笑顔の度合いに応じてメッセージを表示
            if smile_ratio > 17:
                cv2.putText(frame, 'Perfect Smile!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif smile_ratio > 12:
                cv2.putText(frame, 'Good Smile!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Please Smile!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # 顔の領域に矩形を描く
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # フレームを表示
    cv2.imshow('Smile Detector', frame)
    # キー入力を待つ
    key = cv2.waitKey(1)
    # Escキーが押されたら終了
    if key == 27:
        break

# カメラを解放
cap.release()
# ウィンドウを破棄
cv2.destroyAllWindows()
