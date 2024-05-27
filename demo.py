import streamlit as st
import cv2
import numpy as np
from PIL import Image

def resize_eyes(image, scale_factor):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Haar Cascade分類器を読み込む
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # 目の位置を検出
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5)
    for (x, y, w, h) in eyes:
        # 目の範囲を取得
        eye = image[y:y+h, x:x+w]
        # 目のリサイズ
        resized_eye = cv2.resize(eye, (int(w*2), int(h*2)))
        # リサイズした目の画像が元の画像をはみ出さないように制限
        start_x = max(x - int(w//2), 0)
        end_x = min(x + int(w*3//2), image.shape[1])
        start_y = max(y - int(h//2), 0)
        end_y = min(y + int(h*3//2), image.shape[0])
        resized_eye = resized_eye[:end_y - start_y, :end_x - start_x]
        image[start_y:end_y, start_x:end_x] = resized_eye
    return image

def main():
    st.title("画像をアップロードして目を大きくするシステム")

    # スケールファクターを調節するスライダー
    scale_factor = st.slider("スケールファクターを調節してください", min_value=1.0, max_value=2.0, step=0.1, value=1.1)

    # 画像ファイルをアップロード
    uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像ファイルをPIL形式に変換
        image = Image.open(uploaded_file)

        # OpenCV形式に変換
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 目を大きくする処理
        scaled_image = resize_eyes(cv_image, scale_factor)

        # Streamlit上で画像を表示
        st.image(scaled_image, caption='Uploaded Image with Resized Eyes', use_column_width=True)

if __name__ == "__main__":
    main()
    
