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

# ✅ ログの設定（絵文字なし）
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

# ✅ 画像の前処理関数
def preprocess_image(img, file_name):
    if img is None:
        logger.error("preprocess_image: 入力画像が None です！")
        return None, None

    # 1️⃣ **グレースケール変換**
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ **二値化処理**
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 3️⃣ **ノイズ除去**
    binary = cv2.GaussianBlur(binary, (5, 5), 0)

    # 4️⃣ **輪郭抽出**
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        logger.error("preprocess_image: 輪郭が見つかりませんでした！")
        return None, None

    # 5️⃣ **最大輪郭の領域を取得**
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # 6️⃣ **切り取り & 28x28リサイズ**
    cropped = binary[y:y+h, x:x+w]
    processed_img = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # 7️⃣ **白黒反転**
    final_img = cv2.bitwise_not(processed_img)

    # 8️⃣ **処理済み画像の保存**
    processed_image_name = file_name.replace(".", "_") + "_processed.png"
    processed_image_path = os.path.join("debug_images", processed_image_name)
    cv2.imwrite(processed_image_path, final_img)
    logger.info(f"preprocess_image: 画像処理完了 → {processed_image_path}")

    return final_img, processed_image_name  # 画像名を返す

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

        # ✅ ファイル名の正規化
        clean_name = re.sub(r"[^\w\d.]", "_", file.filename).lower()
        file_path = os.path.join("input_images", clean_name)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            logger.error("アップロード画像の読み込みに失敗")
            return render_template("index.html", answer="画像の読み込みに失敗しました", image_path=None)

        # ✅ 画像の前処理
        processed_img, processed_image_name = preprocess_image(img, clean_name)
        if processed_img is None:
            return render_template("index.html", answer="画像の前処理に失敗しました", image_path=None)

        # ✅ モデル入力用の画像処理
        img_array = processed_img / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        try:
            # ✅ モデルで予測
            result = model.predict(img_array)
            probabilities = tf.nn.softmax(result[0]).numpy()
            top1_idx = np.argmax(probabilities)
            top2_idx = np.argsort(probabilities)[-2]
            top1_prob = probabilities[top1_idx] * 100
            top2_prob = probabilities[top2_idx] * 100

            # ✅ 判定結果の表示
            if top1_prob > 20.0:
                pred_answer = f"これは {classes[top1_idx]} です<br>確率: {top1_prob:.4f}%"
            elif 15.0 < top1_prob <= 20.0:
                pred_answer = f"もしかして {classes[top1_idx]} ですか？<br>{classes[top1_idx]} ({top1_prob:.4f}%) と {classes[top2_idx]} ({top2_prob:.4f}%) で悩んでいます"
            else:
                pred_answer = "ちゃんと読めませんでした<br>もう一度お願いします"

            logger.info(f"推論結果: {pred_answer.replace('<br>', ' ')}")

            # ✅ **画像のパスをテンプレートに渡す**
            return render_template("index.html", answer=pred_answer, image_path=f"/processed/{processed_image_name}")

        except Exception as e:
            logger.error(f"モデルの推論中にエラーが発生: {e}")
            return render_template("index.html", answer="推論に失敗しました", image_path=None)

# ✅ 処理済み画像を提供するエンドポイント
@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("debug_images", filename)

# ✅ Flask アプリ起動
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"アプリ起動: ポート {port}")
        app.run(host="0.0.0.0", port=port, threaded=True)
    except Exception as e:
        logger.error(f"サーバー起動中にエラーが発生: {e}")
