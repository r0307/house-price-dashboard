from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve,KFold
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import lightgbm as lgb 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

dataset=fetch_california_housing()

x=dataset.data
t=dataset.target
df=pd.DataFrame(x, columns=dataset.feature_names)
df["Target"]=t

df["BedroomsPerRoom"]=df["AveBedrms"]/df["AveRooms"]
df["PopulationPerHousehold"]=df["Population"]/df["AveOccup"]
sf_lat, sf_lon=37.77,-122.42
df["Dist_SF"]=np.sqrt((df["Latitude"]-sf_lat)**2+(df["Longitude"]-sf_lon)**2)

mean_values = {
            "MedInc": df["MedInc"].mean(),
            "HouseAge": df["HouseAge"].mean(),
            "AveRooms": df["AveRooms"].mean(),
            "AveOccup": df["AveOccup"].mean(),
            "Latitude": df["Latitude"].mean(),
            "Longitude": df["Longitude"].mean(),
            "BedroomsPerRoom": df["BedroomsPerRoom"].mean(),
            "PopulationPerHousehold": df["PopulationPerHousehold"].mean(),
            "Dist_SF": df["Dist_SF"].mean(),
        }

t=df["Target"]
x=df.drop(columns=["Target","AveBedrms","Population"])
x_train,x_test,t_train, t_test=train_test_split(x,t,test_size=0.2, random_state=0)

best_params={
  "num_leaves": 35, 
  "n_estimators": 1000, 
  "min_child_samples": 35, 
  "max_depth": 15, 
  "learning_rate": 0.05
}
best_model=lgb.LGBMRegressor(random_state=0,num_leaves=35, n_estimators=1000, min_child_samples=35, max_depth=15, learning_rate=0.05)
best_model.fit(x_train,t_train)

#予測
t_train_pred=best_model.predict(x_train)
t_test_pred=best_model.predict(x_test)
#R2スコア
train_score=best_model.score(x_train,t_train)
test_score=best_model.score(x_test,t_test)
print(f"train_score:{train_score:.4f}")
print(f"test_score:{test_score:.4f}")
#MAE
train_MAE=mean_absolute_error(t_train,t_train_pred)
test_MAE=mean_absolute_error(t_test,t_test_pred)
print(f"train_MAE:{train_MAE:.4f}")
print(f"test_MAE:{test_MAE:.4f}")
#RMSE
train_RMSE=np.sqrt(mean_squared_error(t_train,t_train_pred))
test_RMSE=np.sqrt(mean_squared_error(t_test,t_test_pred))
print(f"train_RMSE:{train_RMSE:.4f}")
print(f"test_RMSE:{test_RMSE:.4f}")
#MAPE
t_test_clean = t_test[t_test != 0]
t_pred_clean = t_test_pred[t_test != 0]
test_MAPE=np.mean(np.abs((t_test_clean)-(t_pred_clean))/t_test_clean)*100
print(f"test_MAPE:{test_MAPE:.2f}")

#学習曲線
kf=KFold(n_splits=5, shuffle=True, random_state=0)
train_sizes, train_scores, test_scores=learning_curve(
  estimator=best_model,
  X=x_train, y=t_train,
  train_sizes=np.linspace(0.1,1.0,10),
  cv=kf,
  scoring="r2",
  n_jobs=-1
)
train_scores_mean=np.mean(train_scores, axis=1)
test_scores_mean=np.mean(test_scores, axis=1)
plt.figure(figsize=(10,6))
sns.lineplot(x=train_sizes, y=train_scores_mean, label="Training score", marker="o")
sns.lineplot(x=train_sizes, y=test_scores_mean, label="Cross-validation score", marker="o")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Score(R2)")
plt.ylim(0.0,1.05)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.legend()

model_data={
  "model":best_model,
  "feature_names":x.columns.tolist(),
  "r2_score":test_score,
  "mae":test_MAE,
  "rmse":test_RMSE,
  "mape":test_MAPE
}
model_filename="best_california_housing_model.pkl"
joblib.dump(model_data, model_filename)
train_data={
  "X_train":x_train,
  "t_train":t_train,
  "feature_names": x_train.columns.tolist(),
  "mean_values":mean_values
}
joblib.dump(train_data,"california_train_data.pkl")