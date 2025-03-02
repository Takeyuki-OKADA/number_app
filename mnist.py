import sys
import subprocess
import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
import re

# ✅ OpenCV のインストールチェック
try:
    import cv2
except ModuleNotFoundError:
    print("OpenCV is missing. Attempting to install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
    import cv2
    print("OpenCV installed successfully:", cv2.__version__)

# ✅ ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ クラスラベル（0～9）
classes = [str(i) for i in range(10)]

# ✅ 必要なフォルダを作成
for folder in ["debug_images", "input_images", "trim-num-file"]:
    os.makedirs(folder, exist_ok=True)

# ✅ Flask アプリのセットアップ
app = Flask(__name__)

# ✅ 学習済みモデルのロード
model = load_model("./model.keras", compile=False)
logger.info("モデルロード完了")

# ✅ ファイル名の正規化
def clean_filename(filename):
    filename = re.sub(r"[^\w\d.]", "_", filename)  # 記号をアンダースコアに置換
    return filename.lower()  # 小文字化

# ✅ 黒背景の外周を削除
def remove_black_border(img, grid_size=18, threshold=20, brightness_threshold=50):
    h, w = img.shape
    block_h = h // grid_size
    block_w = w // grid_size

    # ✅ 中心座標（grid_size//2, grid_size//2）と半径7ブロック
    center_x, center_y = grid_size // 2, grid_size // 2
    radius = 7  # 半径7ブロック

    for row in range(grid_size):
        for col in range(grid_size):
            # 円形の除外条件（中心からの距離が半径7未満のブロックを除外）
            if np.sqrt((row - center_y) ** 2 + (col - center_x) ** 2) < radius:
                continue  # 円の内側は白塗りしない

            x1, x2 = col * block_w, (col + 1) * block_w
            y1, y2 = row * block_h, (row + 1) * block_h
            block = img[y1:y2, x1:x2]

            # 白ピクセルの割合を計算
            white_ratio = np.mean(block > 0) * 100
            block_brightness = cv2.mean(block)[0]

            # 黒レベルが高すぎる & 平均輝度が低すぎる場合は白塗り
            if white_ratio < threshold and block_brightness < brightness_threshold:
                img[y1:y2, x1:x2] = 255

    return img

# ✅ ノイズ除去関数
def denoise_image(img, blur_ksize=5, morph_kernel_size=3, morph_iterations=2):
    img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    return img

# ✅ 画像の前処理
def preprocess_image(img, file_name):
    if img is None:
        logger.error("preprocess_image: 入力画像が None です！")
        return None, None
    
    # 1️⃣ 外周部15%カット
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    img = img[margin_h:h-margin_h, margin_w:w-margin_w]
    
    # 2️⃣ グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3️⃣ 二値化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 4️⃣ ノイズ除去
    binary = denoise_image(binary)
    # 5️⃣ **黒背景の外周を削除**
    binary = remove_black_border(binary, grid_size=18)
    # 6️⃣ **ノイズ除去**
    binary = denoise_image(binary)
    # 7️⃣ **文字を太くする**
    kernel = np.ones((3, 3), np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=3)
    # 8️⃣ **黒以外をすべて白として除去**
    _, cleaned = cv2.threshold(thickened, 1, 255, cv2.THRESH_BINARY)
    
    # 9️⃣ **輪郭検出（最大の文字領域を抽出）**
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        logger.error("❌ preprocess_image: 輪郭が見つかりませんでした！")
        return None, None
    
    # 1️⃣0️⃣ **最大輪郭の領域を取得し、1.03を掛けて長辺とする**
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    scale_factor = 1.03
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    new_x, new_y = max(x - (new_w - w) // 2, 0), max(y - (new_h - h) // 2, 0)
    new_w = min(new_w, cleaned.shape[1] - new_x)
    new_h = min(new_h, cleaned.shape[0] - new_y)

    cropped = cleaned[new_y:new_y+new_h, new_x:new_x+new_w]
    # 1️⃣1️⃣ **アスペクト比を維持しながら長辺を28pxにリサイズ**
    target_size = 28
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    # 1️⃣2️⃣ **白黒反転**
    final_img = cv2.bitwise_not(resized)
    
    # 1️⃣3️⃣ **デバッグ用に保存**
    debug_base = os.path.join("debug_images", file_name.replace(".", "_"))
    debug_path = debug_base + "_final.png"
    cv2.imwrite(debug_path, final_img)
    
    logger.info(f"✅ preprocess_image: Processed image saved to {debug_path}")
    
    return final_img, debug_path

# ✅ ルートパス（画像アップロード・処理・推論）
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html", answer="", image_path=None)

    if request.method == "POST":
        logger.info("POSTリクエスト受信（推論開始）")

        if "file" not in request.files:
            return render_template("index.html", answer="ファイルがありません", image_path=None)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="ファイルがありません", image_path=None)

        clean_name = clean_filename(file.filename)
        file_path = os.path.join("input_images", clean_name)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            logger.error("アップロード画像の読み込みに失敗")
            return render_template("index.html", answer="画像の読み込みに失敗しました", image_path=None)

        processed_img, processed_image_name = preprocess_image(img, clean_name)
        if processed_img is None:
            return render_template("index.html", answer="画像の前処理に失敗しました", image_path=None)

        img_array = processed_img / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        try:
            result = model.predict(img_array)
            probabilities = tf.nn.softmax(result[0]).numpy()

            # 偏差値を計算
            mean_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)
            deviation_scores = (probabilities - mean_prob) / std_prob * 10 + 50
            adjusted_scores = deviation_scores * 1.25  # 1.25倍に調整

            # 上位2クラスを取得
            top1_idx = np.argmax(adjusted_scores)
            top2_idx = np.argsort(adjusted_scores)[-2]
            top1_score = adjusted_scores[top1_idx]
            top2_score = adjusted_scores[top2_idx]

            # 表記変更（確率 → 自信度）
            if top1_score > 60.0:
                pred_answer = f"これは {classes[top1_idx]} です<br>自信度: {top1_score:.2f}%"
            elif 50.0 < top1_score <= 60.0:
                pred_answer = f"もしかして {classes[top1_idx]} ですか？<br>{classes[top1_idx]} ({top1_score:.2f}%) と {classes[top2_idx]} ({top2_score:.2f}%) で悩んでいます"
            else:
                pred_answer = "ちゃんと読めませんでした<br>もう一度お願いします"

            return render_template("index.html", answer=pred_answer, image_path=f"/processed/{os.path.basename(processed_image_name)}")

        except Exception as e:
            logger.error(f"モデルの推論中にエラーが発生: {e}")
            return render_template("index.html", answer="推論に失敗しました", image_path=None)


@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("debug_images", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
