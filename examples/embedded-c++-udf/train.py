# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# 加载Iris数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 目标标签

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集结果
y_pred = clf.predict(X_test)

# 输出模型的准确率
print("模型准确率:", metrics.accuracy_score(y_test, y_pred))

# 保存训练好的模型到文件
joblib.dump(clf, "decision_tree_model.pkl")
