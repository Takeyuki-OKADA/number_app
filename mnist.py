import sys
import subprocess
import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, render_template
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

# ✅ ファイル名の正規化
def clean_filename(filename):
    return re.sub(r"[^\w\d.]", "_", filename).lower()

# ✅ ルートパス（画像アップロード・処理・推論）
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        logger.info("GETリクエスト受信 → index.html を表示")
        return render_template("index.html", answer="", processing=False)

    if request.method == "POST":
        logger.info("POSTリクエスト受信（推論開始）")

        if "file" not in request.files:
            return render_template("index.html", answer="ファイルがありません", processing=False)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="ファイルがありません", processing=False)

        # ✅ ファイル名の正規化
        clean_name = clean_filename(file.filename)
        file_path = os.path.join("input_images", clean_name)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            logger.error("アップロード画像の読み込みに失敗")
            return render_template("index.html", answer="画像の読み込みに失敗しました", processing=False)

        # ✅ 画像の前処理
        processed_img = preprocess_image(img, clean_name)
        if processed_img is None:
            return render_template("index.html", answer="画像の前処理に失敗しました", processing=False)

        # ✅ モデル入力用の画像処理
        img_array = processed_img / 255.0  # 正規化
        img_array = np.expand_dims(img_array, axis=0)  # バッチ次元追加
        img_array = np.expand_dims(img_array, axis=-1)  # チャネル次元追加

        logger.info(f"推論対象の画像の形状: {img_array.shape}")
        logger.info(f"入力ファイル名: {clean_name}")

        try:
            # ✅ モデルで予測
            result = model.predict(img_array)
            probabilities = tf.nn.softmax(result[0]).numpy()
            predicted_idx = np.argmax(probabilities)
            predicted_class = classes[predicted_idx]
            predicted_prob = probabilities[predicted_idx] * 100  # パーセント表記

            # ✅ 上位2クラスを取得
            sorted_indices = np.argsort(probabilities)[::-1]
            top1_idx = sorted_indices[0]
            top2_idx = sorted_indices[1]
            top1_prob = probabilities[top1_idx] * 100
            top2_prob = probabilities[top2_idx] * 100

            # ✅ 判定結果の表示処理
            if top1_prob > 20.0:
                pred_answer = f"これは {classes[top1_idx]} です\n確率: {top1_prob:.4f}%"
            elif 15.0 < top1_prob <= 20.0:
                pred_answer = f"もしかして {classes[top1_idx]} ですか？\n{classes[top1_idx]} ({top1_prob:.4f}%) と {classes[top2_idx]} ({top2_prob:.4f}%) で悩んでいます"
            else:
                pred_answer = "ちゃんと読めませんでした\nもう一度お願いします"

            # ✅ 判定結果をログに出力
            logger.info(f"推論結果: {pred_answer} (確率: {top1_prob:.4f}%)")

        except Exception as e:
            logger.error(f"モデルの推論中にエラーが発生: {e}")
            return render_template("index.html", answer="推論に失敗しました", processing=False)

    return render_template("index.html", answer=pred_answer, processing=False)

# ✅ Flask アプリ起動
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"アプリ起動: ポート {port}")
        app.run(host="0.0.0.0", port=port, threaded=True)
    except Exception as e:
        logger.error(f"サーバー起動中にエラーが発生: {e}")
