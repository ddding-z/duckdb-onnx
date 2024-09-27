#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>

namespace py = pybind11;
using namespace pybind11::literals;

int main() {
    try {
        // 初始化 Python 解释器
        py::scoped_interpreter guard{};

        // 导入必要的 Python 模块
        py::module sklearn = py::module::import("sklearn");
        py::module skl2onnx = py::module::import("skl2onnx");
        py::module joblib = py::module::import("joblib");
        py::module data_types = py::module::import("skl2onnx.common.data_types");
        py::module builtins = py::module::import("builtins");

        // 加载模型
        py::object model = joblib.attr("load")("models/decision_tree_model.pkl");
        std::cout << "模型加载成功。" << std::endl;

        // 动态获取特征数量
        int n_features = 0;
        if (py::hasattr(model, "n_features_in_")) {
            n_features = model.attr("n_features_in_").cast<int>();
            std::cout << "模型的特征数量: " << n_features << std::endl;
        } else {
            std::cerr << "模型没有 'n_features_in_' 属性，无法确定特征数量。" << std::endl;
            return 1;
        }

        // 构造 initial_type_
        py::list shape;
        shape.append(py::none());
        shape.append(n_features);
        py::object FloatTensorType_obj = data_types.attr("FloatTensorType")(shape);

        py::tuple input_tuple = py::make_tuple("input", FloatTensorType_obj);
        py::list initial_type_;
        initial_type_.append(input_tuple);

        // 构造 options 参数
        py::object model_id = builtins.attr("id")(model); 
        py::dict inner_options;
        inner_options["zipmap"] = false; 
        py::dict options;
        options[model_id] = inner_options;

        // 转换模型为 ONNX
        py::object convert_sklearn = skl2onnx.attr("convert_sklearn");
        // 使用关键字参数传递 initial_types 和 options
        py::object onnx_model = convert_sklearn(
            model,
            "initial_types"_a = initial_type_,
            "options"_a = options
        );
        std::cout << "模型转换为 ONNX 成功。" << std::endl;

        // 保存 ONNX 模型
        std::ofstream ofs("models/decision_tree_model.onnx", std::ios::binary);
        if (!ofs) {
            std::cerr << "无法打开文件以保存 ONNX 模型。" << std::endl;
            return 1;
        }

        py::object onnx_bytes = onnx_model.attr("SerializeToString")();
        std::string onnx_str = onnx_bytes.cast<std::string>();
        ofs.write(onnx_str.data(), onnx_str.size());
        ofs.close();
        std::cout << "ONNX 模型已保存。" << std::endl;

    } catch (py::error_already_set &e) {
        std::cerr << "Python 错误：" << e.what() << std::endl;
        return 1;
    } catch (std::exception &e) {
        std::cerr << "C++ 错误：" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
