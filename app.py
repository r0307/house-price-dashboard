from flask import Flask, request, render_template
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import uuid
from math import log10
import logging

#初期設定
app=Flask(__name__)

logging.basicConfig(level=logging.INFO)

model_data=None
train_data=None
try:
  model_data=joblib.load("best_california_housing_model.pkl")
  train_data=joblib.load("california_train_data.pkl")
except FileNotFoundError:
  logging.error("モデルファイルが見つかりません。train_model.pyを実行してモデルを保存してください。")

#ディレクトリ
PLOT_DIR="static"
if not os.path.exists(PLOT_DIR):
  os.makedirs(PLOT_DIR)

#データの読み込み
feature_names= train_data.get("feature_names", [])
mean_values= train_data.get("mean_values", {})
r2_score = model_data.get("r2_score", "N/A")
mae = model_data.get("mae", "N/A")
rmse = model_data.get("rmse", "N/A")
mape = model_data.get("mape", "N/A")

#フォーム項目ごとの設定
features_config = {
    "MedInc": {"label": "中間所得 (万ドル)", "min": 0, "max": 150, "step": 0.1, "initial": mean_values.get("MedInc", 0)},
    "HouseAge": {"label": "住宅の築年数", "min": 0, "max": 60, "step": 1, "initial": mean_values.get("HouseAge", 0)},
    "AveRooms": {"label": "平均部屋数", "min": 0, "max": 15, "step": 0.1, "initial": mean_values.get("AveRooms", 0)},
    "AveOccup": {"label": "平均世帯人数", "min": 0, "max": 10, "step": 0.1, "initial": mean_values.get("AveOccup", 0)},
    "Latitude": {"label": "緯度", "min": 32, "max": 42, "step": 0.01, "initial": mean_values.get("Latitude", 0)},
    "Longitude": {"label": "経度", "min": -125, "max": -114, "step": 0.01, "initial": mean_values.get("Longitude", 0)},
    "BedroomsPerRoom": {"label": "部屋数あたりの寝室数", "min": 0, "max": 1, "step": 0.01, "initial": mean_values.get("BedroomsPerRoom", 0)},
    "PopulationPerHousehold": {"label": "世帯数", "min": 0, "max": 1000, "step": 1, "initial": mean_values.get("PopulationPerHousehold", 0)},
    "Dist_SF": {"label": "サンフランシスコからの距離 (マイル)", "min": 0, "max": 500, "step": 1, "initial": mean_values.get("Dist_SF", 0)},
}
# 小数点以下の桁数を計算するヘルパー関数
def count_decimal_places(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1])
    return 0
# 平均値を計算し、stepに合わせて丸める
for key, config in features_config.items():
    step = config.get("step", 1)
    # stepが整数の場合、小数点以下は0
    if step == int(step):
        decimals = 0
    else:
        decimals = count_decimal_places(step)  
# 訓練データの平均値を取得し、stepに合わせて丸める
    mean_val = mean_values.get(key, 0)
    rounded_mean = round(mean_val, decimals)
    features_config[key]["initial"] = rounded_mean

#予測結果の可視化
def create_prediction_plot(user_df, prediction, x_feature):
  filename = f"prediction_plot_{uuid.uuid4().hex}.png"
  output_path = os.path.join(PLOT_DIR, filename)
  if not train_data:
    return None
  X_train_data=train_data["X_train"]
  t_train_data=train_data["t_train"]
  plot_x_feature = x_feature
  if plot_x_feature not in X_train_data.columns:
        plot_x_feature = "AveRooms"
    
  x_label_map={
        "MedInc": "Median Income",
        "HouseAge": "House Age",
        "AveRooms": "Average Rooms",
        "AveOccup": "Average Household Occupancy",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "BedroomsPerRoom": "Bedrooms per Room",
        "PopulationPerHousehold": "Population per Household",
        "Dist_SF": "Distance from San Francisco"
  }
  x_label=x_label_map.get(x_feature, x_feature)
  plt.style.use("seaborn-v0_8-whitegrid")
  plt.figure(figsize=(10, 6))
  mask = pd.Series(True, index=X_train_data.index)
  for col in X_train_data.columns:
    if col != x_feature:
        val = user_df[col].values[0]
        tol = (X_train_data[col].max() - X_train_data[col].min()) * 0.1
        mask &= (X_train_data[col] >= val - tol) & (X_train_data[col] <= val + tol)
  filtered_X = X_train_data[mask]
  filtered_y = t_train_data[mask]
  plt.scatter(
        filtered_X[x_feature],
        filtered_y,
        alpha=0.6,
        color="gray",
        label="Other Housing Data"
    )
  plt.scatter(
        user_df[x_feature].values[0],
        prediction,
        color="#FFD700",
        marker="*",
        s=400,
        edgecolors="white",
        linewidths=1.5,
        label=f"Your Prediction\n(Predicted: ${prediction:.2f}M)"
    )
  plt.title(f"{x_label} vs. Price", fontsize=24)
  plt.xlabel(f"{x_label}", fontsize=18)
  plt.ylabel("Price (Million USD)", fontsize=18)
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()
  return filename

@app.route("/", methods=["GET", "POST"])
def index():
  prediction=None
  plot_url=None
  error=None

  display_feature_keys = list(features_config.keys())
  
  selected_x_feature=request.form.get("x_feature_select", "AveRooms")

  if request.method=="POST" and model_data:
    try:
      # ユーザーの入力値を取得し、フォームの再表示のために保存
      user_input_dict={}
      for key in display_feature_keys:
        form_value=request.form.get(key)
        if form_value and form_value.strip():
          user_input_dict[key]=float(form_value)
        else:
          user_input_dict[key]=features_config[key]["initial"]
      # モデルに渡す特徴量の順番を揃える
      ordered_input = {col: user_input_dict.get(col) for col in feature_names}
      user_df=pd.DataFrame([ordered_input], columns=feature_names)
      #予測
      prediction=model_data["model"].predict(user_df)[0]
      #plot_url
      plot_url=create_prediction_plot(user_df, prediction, selected_x_feature)
      #再表示時にユーザーの入力値を使用
      initial_values_to_pass = user_input_dict
    except (ValueError, TypeError) as e:
            error=f"入力値が無効です。数値で入力してください。"
            logging.error(f"入力値エラー: {e}")
    except Exception as e:
            error=f"予測中に予期せぬエラーが発生しました。"
            logging.error(f"予期せぬエラー: {e}")
  else:
     #最初のページ読み込み時はfeatres_configのinitialを取得
     initial_values_to_pass={key:features_config[key]["initial"] for key in features_config.keys()}
  return render_template("index.html", 
                         prediction=prediction, 
                         plot_url=plot_url, 
                         error=error, 
                         initial_values=initial_values_to_pass,
                         features_config=features_config,
                         selected_x_feature=selected_x_feature,
                         r2_score=r2_score,
                         mae=mae,
                         rmse=rmse,
                         mape=mape)

if __name__=="__main__":
  app.run(debug=True)