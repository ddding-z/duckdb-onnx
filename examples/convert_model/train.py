# train_and_save.py
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

os.makedirs('./models', exist_ok=True)
# 加载数据
X, y = load_iris(return_X_y=True)

# 训练模型
model = RandomForestClassifier()
model.fit(X, y)

# 保存模型
joblib.dump(model, './models/decision_tree_model.pkl')
print("模型已保存到 models/decision_tree_model.pkl")
