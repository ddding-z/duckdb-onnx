import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 加载模型
model = joblib.load('./models/decision_tree_model.pkl')

# 定义初始类型，形状为 (None, 4)
initial_type_ = [('input', FloatTensorType([None, 4]))]

# 转换模型为 ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type_)

# 保存 ONNX 模型
with open("models/decision_tree_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("模型转换并保存成功！")
