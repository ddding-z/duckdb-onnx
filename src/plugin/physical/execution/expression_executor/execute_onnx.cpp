#include "duckdb/plugin/physical/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_onnx_expression.hpp"
#include "onnxruntime_cxx_api.h"

namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundOnnxExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	// result->Finalize(true);
	result->Finalize();
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundOnnxExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
    // D_ASSERT(expr.path);
    std::string weightFile = expr.path;
    state->intermediate_chunk.Reset();
    
    // 初始化 onnx runtime
    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<std::string> output_names;
    std::vector<ONNXTensorElementDataType> output_types;

    // ONNX Runtime session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "duckdb");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, weightFile.c_str(), session_options);


    // 获取模型的输入和输出信息
    size_t num_input_nodes = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        // allocator.Free(input_name);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_node_dims = tensor_info.GetShape();
        // set batch size= 1
        for (auto& dim : input_node_dims) {
            if (dim == -1) {
                dim = 1;
            }
        }
        input_shapes.push_back(input_node_dims);
        input_types.push_back(tensor_info.GetElementType());
    }

    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        // allocator.Free(output_name);
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_types.push_back(tensor_info.GetElementType());
    }
    // 准备ONNX模型的输入节点名称
    std::vector<const char*> input_node_names(input_names.size());
    for (size_t i = 0; i < input_names.size(); ++i) {
        input_node_names[i] = input_names[i].c_str();
    }
    std::vector<const char*> output_node_names(output_names.size());
    for (size_t i = 0; i < output_names.size(); ++i) {
        output_node_names[i] = output_names[i].c_str();
    }

    // 构建input tensor
    auto &chunk = *this->chunk;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    // 1.创建内存信息对象和输入张量
    // 2.根据输入数据类型（FLOAT, INTEGER, BIGINT）创建相应的输入张量。
    for (size_t input_idx = 0; input_idx < input_names.size(); input_idx++) {
        auto &input_shape = input_shapes[input_idx];
        auto input_type = input_types[input_idx];
        LogicalType column_type = chunk.data[input_idx].GetType();
        // ToDo 类型处理：FLOAT, INTEGER, BIGINT, DOUBLE, VARCHAR, CHAR
        switch (column_type.id()) {
            case LogicalTypeId::FLOAT:
                if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    std::vector<float> input_tensor_values(count);
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        input_tensor_values[i] = chunk.data[input_idx].GetValue(idx).GetValue<float>();
                    }
                    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()));
                }
                break;
            case LogicalTypeId::INTEGER:
                if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                    std::vector<int32_t> input_tensor_values(count);
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        input_tensor_values[i] = chunk.data[input_idx].GetValue(idx).GetValue<int32_t>();
                    }
                    input_tensors.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()));
                }
                break;
            case LogicalTypeId::BIGINT:
                if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    std::vector<int64_t> input_tensor_values(count);
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        input_tensor_values[i] = chunk.data[input_idx].GetValue(idx).GetValue<int64_t>();
                    }
                    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()));
                }
                break;
            // 添加更多类型处理 DOUBLE, CHAR
            default:
                throw NotImplementedException("Unsupported input type: " + column_type.ToString());
        }
        if (input_tensors.size() != input_idx + 1) {
            throw std::runtime_error("Failed to create input tensor for input " + std::to_string(input_idx));
        }
    }
    // 模型推理
    // auto output_tensors = expr.session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    try {
        // 执行模型推理: std::vector<Ort::Value> input_tensors;
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size());
        // 处理output tensors
        if (output_tensors.empty()) {
            throw std::runtime_error("No output tensors produced by the ONNX model");
        }
        for (size_t output_idx = 0; output_idx < output_tensors.size(); output_idx++) {
            // 获取输出张量的类型和形状
            auto &output_tensor = output_tensors[output_idx];
            auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
            auto output_type = tensor_info.GetElementType();
            auto output_shape = tensor_info.GetShape();
            // 检查输出张量的形状是否与预期一致
            size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
            if (output_size != count) {
                throw std::runtime_error(std::string("Output tensor size mismatch") +",output_size: " + std::to_string(output_size) +",count: " + std::to_string(count));
            }
            switch (output_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    auto *output_data = output_tensor.GetTensorData<float>();
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        result.SetValue(idx, Value::FLOAT(output_data[i]));
                    }
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    auto *output_data = output_tensor.GetTensorData<int32_t>();
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        result.SetValue(idx, Value::INTEGER(output_data[i]));
                    }
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    auto *output_data = output_tensor.GetTensorData<int64_t>();
                    for (idx_t i = 0; i < count; i++) {
                        auto idx = sel ? sel->get_index(i) : i;
                        result.SetValue(idx, Value::BIGINT(output_data[i]));
                    }
                    break;
                }
                // 添加更多类型处理...
                default:
                    throw std::runtime_error("Unsupported output type: " + std::to_string(output_type));
            }
        }
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX model execution failed: " + std::string(e.what()));
    }

} // namespace duckdb
}
/*
1.关注一下onnxruntime的c++ api 如何运行 需要哪些输入和输出
2.关注datachunk->onnxruntime的输入
3.onnxruntime的输出如何到UnifiedVectorFormat (Vector &result)
4.检查onnxruntime c++ api是否有调用错误
5.1 从chunk获取到的模型的输入是否可以匹配到模型的输入 目前应该只能处理单输入/多输入需要onnxexpression的更多表达力
5.2 测试简单的线性数据集
5.3 构造简单的树模型 case 基于红酒分类数据集 基于onnxruntime直接运行
6. 将csv数据导入duckdb，然后运行BoundOnnxExpression.execute
*/

// scirpt 
    // Ort::Value input_tensor = nullptr;
    // LogicalType column_type = chunk.data[0].GetType();
    // switch (column_type.id()) {
    //     case LogicalTypeId::FLOAT:
    //         if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    //             input_tensor = CreateInputTensor<float>(chunk, sel, count, input_shape);
    //         }
    //         break;
    //     case LogicalTypeId::INTEGER:
    //         if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    //             input_tensor = CreateInputTensor<int32_t>(chunk, sel, count, input_shape);
    //         }
    //         break;
    //     case LogicalTypeId::BIGINT:
    //         if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    //             input_tensor = CreateInputTensor<int64_t>(chunk, sel, count, input_shape);
    //         }
    //         break;
    //     default:
    //         throw NotImplementedException("Unsupported input type");
    // }
    // if (!input_tensor) {
    //     throw NotImplementedException("Failed to create input tensor due to type mismatch");
    // }

    // template <typename T>
    // Ort::Value CreateInputTensor(DataChunk &chunk, const SelectionVector *sel, idx_t count, std::vector<int64_t> &input_shape) {
    //     std::vector<T> input_tensor_values(count);
    //     auto &vec = chunk.data[0];
    //     UnifiedVectorFormat vec_data;
    //     vec.ToUnifiedFormat(count, vec_data);

    //     auto data_ptr = (T*)vec_data.data;
    //     if (sel) {
    //         for (idx_t i = 0; i < count; i++) {
    //             auto idx = sel->get_index(i);
    //             input_tensor_values[i] = data_ptr[vec_data.sel->get_index(idx)];
    //         }
    //     } else {
    //         for (idx_t i = 0; i < count; i++) {
    //             input_tensor_values[i] = data_ptr[vec_data.sel->get_index(i)];
    //         }
    //     }

    //     return Ort::Value::CreateTensor<T>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
    //                                        input_tensor_values.data(), input_tensor_values.size(),
    //                                        input_shape.data(), input_shape.size());
    // }


    // // 构建output tensor
    // auto &output_tensor = output_tensors[0];
    // auto *output_data = output_tensor.GetTensorMutableData<float>();
    // ONNXTensorElementDataType output_type = output_types[0];
    // // Convert Output
    // switch (output_type) {
    //     case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    //         auto *output_data = output_tensor.GetTensorMutableData<float>();
    //         for (idx_t i = 0; i < count; i++) {
    //             auto idx = sel ? sel->get_index(i) : i;
    //             result.SetValue(idx, Value(output_data[i]));
    //         }
    //         break;
    //     }
    //     case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    //         auto *output_data = output_tensor.GetTensorMutableData<int32_t>();
    //         for (idx_t i = 0; i < count; i++) {
    //             auto idx = sel ? sel->get_index(i) : i;
    //             result.SetValue(idx, Value(output_data[i]));
    //         }
    //         break;
    //     }
    //     case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    //         auto *output_data = output_tensor.GetTensorMutableData<int64_t>();
    //         for (idx_t i = 0; i < count; i++) {
    //             auto idx = sel ? sel->get_index(i) : i;
    //             result.SetValue(idx, Value(output_data[i]));
    //         }
    //         break;
    //     }
    //     default:
    //         throw NotImplementedException("Unsupported output type");
    // }