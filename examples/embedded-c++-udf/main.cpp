#include <pybind11/embed.h>  // pybind11嵌入式API
#include "duckdb.hpp"
#include <iostream>
#include <stdexcept>

namespace py = pybind11;
using namespace duckdb;
using namespace std;

std::string convert_sklearn_model_to_onnx(const std::string &model_file, const std::string &onnx_file) {
    try {
        py::module skl2onnx = py::module::import("convert_model_to_onnx");
        py::object result = skl2onnx.attr("convert_model_to_onnx")(model_file, onnx_file);
        return result.cast<std::string>();  // 返回ONNX文件路径
    } catch (const std::exception &e) {
        throw std::runtime_error("Error in Python function call: " + std::string(e.what()));
    }
}

void ConvertModelUDF(Vector &input, Vector &output, idx_t count) {
    auto model_file = StringValue::Get(input.GetValue(0));
    auto onnx_file = StringValue::Get(input.GetValue(1));

    std::string converted_onnx_file = convert_sklearn_model_to_onnx(model_file, onnx_file);

    output.SetValue(0, Value(converted_onnx_file));
}

int main() {
    py::scoped_interpreter guard{};
    DuckDB db(nullptr);
    Connection con(db);

    con.Query("CREATE TABLE models (model_file VARCHAR, onnx_file VARCHAR);");
    con.CreateScalarFunction("convert_model_to_onnx", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
                             ConvertModelUDF);
    con.Query("INSERT INTO models VALUES ('model.pkl', 'model.onnx');");
    auto result = con.Query("SELECT convert_model_to_onnx(model_file, onnx_file) FROM models;");
    result->Print();
    return 0;
}
