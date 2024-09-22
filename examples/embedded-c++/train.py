'''
训练一个简单的线性回归模型用于测试
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

model = LinearRegression()
model.fit(X, y)

initial_type = [('float_input', FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
onnx.save_model(onnx_model, "simple_linear_regression.onnx")
