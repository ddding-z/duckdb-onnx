#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/type_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_type_info.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/planner/extension_callback.hpp"
#include "duckdb/sql_function/function.hpp"
#include "onnx/onnx_pb.h"
#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace duckdb;

class ModelCache {
public:
  static const size_t capacity = 10;
  static std::unordered_map<std::string, std::shared_ptr<Ort::Session>>
      modelcache;
  static std::queue<std::string> model_queue;
  static std::shared_mutex map_mutex;

  static std::shared_ptr<Ort::Session>
  getOrCreateSession(const std::string &key, const Ort::Env &env,
                     const Ort::SessionOptions &options) {
    {
      std::shared_lock<std::shared_mutex> sharedLock(map_mutex);
      auto it = modelcache.find(key);
      if (it != modelcache.end()) {
        return it->second;
      }
    }
    std::unique_lock<std::shared_mutex> lock(map_mutex);
    auto it = modelcache.find(key);
    if (it != modelcache.end()) {
      return it->second;
    }

    std::shared_ptr<Ort::Session> session;
    try {
      session = std::make_shared<Ort::Session>(env, key.c_str(), options);
    } catch (const Ort::Exception &e) {
      std::cerr << "Failed to create session: " << e.what() << std::endl;
      return nullptr;
    }

    if (modelcache.size() >= capacity) {
      const std::string &old_key = model_queue.front();
      modelcache.erase(old_key);
      model_queue.pop();
    }
    model_queue.push(key);
    modelcache[key] = session;

    return session;
  }
};
std::unordered_map<std::string, std::shared_ptr<Ort::Session>>
    ModelCache::modelcache;
std::queue<std::string> ModelCache::model_queue;
std::shared_mutex ModelCache::map_mutex;

// Function to perform inference const vector<const void *> &input_buffers
vector<double_t> InferenceModel(const std::string &model_path,
                                const vector<const void *> &input_buffers,
                                const vector<int64_t> &input_shape) {

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceModel");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  std::shared_ptr<Ort::Session> session =
      ModelCache::getOrCreateSession(model_path, env, session_options);
  if (!session) {
    std::cerr << "Failed to create session: " << std::endl;
    return {};
  }

  Ort::AllocatorWithDefaultOptions allocator;
  vector<std::string> input_node_names;
  vector<std::string> output_node_names;
  size_t numInputNodes = session->GetInputCount();
  size_t numOutputNodes = session->GetOutputCount();
  input_node_names.reserve(numInputNodes);
  output_node_names.reserve(numOutputNodes);

  vector<int64_t> adjusted_input_shape = input_shape;
  for (size_t i = 0; i < numInputNodes; i++) {
    auto input_name = session->GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());

    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> input_dims = input_tensor_info.GetShape();

    for (auto &dim : input_dims) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }
    adjusted_input_shape = input_dims;
  }

  for (size_t i = 0; i < numOutputNodes; i++) {
    auto output_name = session->GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(output_name.get());

    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();

    for (auto &dim : output_dims) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }
  }

  // Create input tensor
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor(nullptr);
  if (input_buffers.empty() || input_buffers[0] == nullptr) {
    std::cerr << "Error: Input buffers are empty or null." << std::endl;
    return {};
  }

  // 调整列存数据为行优先数据
  vector<float> row_major_data(adjusted_input_shape[0] *
                               adjusted_input_shape[1]);

  int num_rows = adjusted_input_shape[0]; // 样本数 (行数)
  int num_cols = adjusted_input_shape[1]; // 特征数 (列数)

  for (int col = 0; col < num_cols; ++col) {
    // 提前将列的 void* 转换为 float*
    const float *column_data = static_cast<const float *>(input_buffers[col]);
    // 遍历行数据，转换为行优先存储
    for (int row = 0; row < num_rows; ++row) {
      row_major_data[row * num_cols + col] = column_data[row];
    }
  }

  // Get the data type of the input tensor from the ONNX model
  size_t input_index = 0;
  Ort::TypeInfo type_info = session->GetInputTypeInfo(input_index);
  Ort::ConstTensorTypeAndShapeInfo tensor_info =
      type_info.GetTensorTypeAndShapeInfo();
  auto data_type = tensor_info.GetElementType();

  input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,           // 内存信息
                                      row_major_data.data(), // 指向行优先的数据
                                      row_major_data.size(), // 数据元素总数
                                      adjusted_input_shape.data(), // 张量形状
                                      adjusted_input_shape.size() // 维度数
      );

  // 获取输出张量
  vector<int64_t> output_shape;
  try {
    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape = output_tensor_info.GetShape();
    // 检查输出张量中是否有动态维度
    for (auto &dim : output_shape) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }

    // 创建输出张量
    Ort::Value output_tensor(nullptr);
    // 获取输出张量的数据类型
    auto output_data_type = output_tensor_info.GetElementType();
    if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      try {
        vector<float_t> output_data(output_shape[0]);
        output_tensor = Ort::Value::CreateTensor<float_t>(
            memory_info, output_data.data(), output_data.size(),
            output_shape.data(), output_shape.size());
        // run
        vector<const char *> input_node_names_c;
        vector<const char *> output_node_names_c;
        for (const auto &name : input_node_names) {
          input_node_names_c.push_back(name.c_str());
        }
        for (const auto &name : output_node_names) {
          output_node_names_c.push_back(name.c_str());
        }

        try {
          session->Run(Ort::RunOptions{nullptr}, input_node_names_c.data(),
                       &input_tensor, 1, output_node_names_c.data(),
                       &output_tensor, 1);
        } catch (const Ort::Exception &e) {
          std::cerr << "Failed to run inference: " << e.what() << std::endl;
          return {};
        }
        double *output_data_ptr = output_tensor.GetTensorMutableData<double>();
        vector<double_t> results(output_data_ptr,
                                 output_data_ptr + output_data.size());

        return results;
      } catch (const Ort::Exception &e) {
        std::cerr << "Failed to create output tensor: " << e.what()
                  << std::endl;
        return {};
      }
    } else if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      try {
        vector<int64_t> output_data(output_shape[0]);
        output_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, output_data.data(), output_data.size(),
            output_shape.data(), output_shape.size());
        // run
        vector<const char *> input_node_names_c;
        vector<const char *> output_node_names_c;
        for (const auto &name : input_node_names) {
          input_node_names_c.push_back(name.c_str());
        }
        for (const auto &name : output_node_names) {
          output_node_names_c.push_back(name.c_str());
        }

        try {
          session->Run(Ort::RunOptions{nullptr}, input_node_names_c.data(),
                       &input_tensor, 1, output_node_names_c.data(),
                       &output_tensor, 1);
        } catch (const Ort::Exception &e) {
          std::cerr << "Failed to run inference: " << e.what() << std::endl;
          return {};
        }
        int64_t *output_data_ptr =
            output_tensor.GetTensorMutableData<int64_t>();
        vector<double_t> results(output_data_ptr,
                                 output_data_ptr + output_data.size());

        return results;
      } catch (const Ort::Exception &e) {
        std::cerr << "Failed to create output tensor: " << e.what()
                  << std::endl;
        return {};
      }
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "Failed to get output shape: " << e.what() << std::endl;
    return {};
  }

  return {};
}

static void OnnxInferenceFunction(DataChunk &args, ExpressionState &state,
                                  Vector &result) {
  auto &model_path_vector = args.data[0];
  // check data type
  if (model_path_vector.GetType().id() != LogicalTypeId::VARCHAR) {
    std::cerr << "Error: Model path column must be VARCHAR type." << std::endl;
    return;
  }
  // **获取模型地址
  auto model_path_data = FlatVector::GetData<string_t>(model_path_vector);
  std::string model_path = model_path_data[0].GetString();
  //** 获取模型输入feature name
  vector<UnifiedVectorFormat> feature_data(args.ColumnCount() - 1);
  for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
    args.data[col_idx].ToUnifiedFormat(args.size(), feature_data[col_idx - 1]);
  }
  //** 构造直接指向 DuckDB 内存数据的指针数组
  vector<const void *> input_buffers;
  for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
    const void *ptr = feature_data[col_idx - 1].data;
    input_buffers.push_back(ptr);
  }

  // ** input_shape: [batch_size, num_features]
  vector<int64_t> input_shape = {static_cast<int64_t>(args.size()),
                                 static_cast<int64_t>(args.ColumnCount()) - 1};

  // Run inference
  //   auto inference_results =
  //       InferenceModel(model_path, input_buffers, input_shape);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceModel");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1); // onnxruntime单线程

  std::shared_ptr<Ort::Session> session =
      ModelCache::getOrCreateSession(model_path, env, session_options);
  if (!session) {
    std::cerr << "Failed to create session: " << std::endl;
  }

  Ort::AllocatorWithDefaultOptions allocator;
  vector<std::string> input_node_names;
  vector<std::string> output_node_names;
  size_t numInputNodes = session->GetInputCount();
  size_t numOutputNodes = session->GetOutputCount();
  input_node_names.reserve(numInputNodes);

  vector<int64_t> adjusted_input_shape = input_shape;
  for (size_t i = 0; i < numInputNodes; i++) {
    auto input_name = session->GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());

    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> input_dims = input_tensor_info.GetShape();

    for (auto &dim : input_dims) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }
    adjusted_input_shape = input_dims;
  }

  for (size_t i = 0; i < numOutputNodes; i++) {
    auto output_name = session->GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(output_name.get());

    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();

    for (auto &dim : output_dims) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }
	break;
  }

  // Create input tensor
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor(nullptr);
  if (input_buffers.empty() || input_buffers[0] == nullptr) {
    std::cerr << "Error: Input buffers are empty or null." << std::endl;
  }

  // 调整列存数据为行优先数据
  vector<float> row_major_data(adjusted_input_shape[0] *
                               adjusted_input_shape[1]);

  int num_rows = adjusted_input_shape[0]; // 样本数 (行数)
  int num_cols = adjusted_input_shape[1]; // 特征数 (列数)

  for (int col = 0; col < num_cols; ++col) {
    // 提前将列的 void* 转换为 float*
    const float *column_data = static_cast<const float *>(input_buffers[col]);
    // 遍历行数据，转换为行优先存储
    for (int row = 0; row < num_rows; ++row) {
      row_major_data[row * num_cols + col] = column_data[row];
    }
  }

  // Get the data type of the input tensor from the ONNX model
  size_t input_index = 0;
  Ort::TypeInfo type_info = session->GetInputTypeInfo(input_index);
  Ort::ConstTensorTypeAndShapeInfo tensor_info =
      type_info.GetTensorTypeAndShapeInfo();
  auto data_type = tensor_info.GetElementType();

  input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,           // 内存信息
                                      row_major_data.data(), // 指向行优先的数据
                                      row_major_data.size(), // 数据元素总数
                                      adjusted_input_shape.data(), // 张量形状
                                      adjusted_input_shape.size() // 维度数
      );

  // 获取输出张量
  vector<int64_t> output_shape;
  try {
    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape = output_tensor_info.GetShape();
    // 检查输出张量中是否有动态维度
    for (auto &dim : output_shape) {
      if (dim == -1) {
        dim = input_shape[0];
      }
    }

    // 创建输出张量
    Ort::Value output_tensor(nullptr);
    // 获取输出张量的数据类型
    auto output_data_type = output_tensor_info.GetElementType();
    if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      try {
        vector<float_t> output_data(output_shape[0]);
        output_tensor = Ort::Value::CreateTensor<float_t>(
            memory_info, output_data.data(), output_data.size(),
            output_shape.data(), output_shape.size());
        // run
        vector<const char *> input_node_names_c;
        vector<const char *> output_node_names_c;
        for (const auto &name : input_node_names) {
          input_node_names_c.push_back(name.c_str());
        }
        for (const auto &name : output_node_names) {
          output_node_names_c.push_back(name.c_str());
        }

        try {
          session->Run(Ort::RunOptions{nullptr}, input_node_names_c.data(),
                       &input_tensor, 1, output_node_names_c.data(),
                       &output_tensor, 1);
        } catch (const Ort::Exception &e) {
          std::cerr << "Failed to run inference: " << e.what() << std::endl;
        }
        float *output_data_ptr = output_tensor.GetTensorMutableData<float>();
        vector<float_t> inference_results(output_data_ptr,
                                          output_data_ptr + output_data.size());
        // Write output vector
        result.SetVectorType(VectorType::FLAT_VECTOR);

        if (inference_results.size() != args.size()) {
          std::cerr << "Error: Inference results size mismatch." << std::endl;
          return;
        }
        auto result_data = FlatVector::GetData<float_t>(result);
        for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
          result_data[row_idx] = inference_results[row_idx];
        }
        result.Verify(args.size());
      } catch (const Ort::Exception &e) {
        std::cerr << "Failed to create output tensor: " << e.what()
                  << std::endl;
      }
    } else if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      try {
        vector<int64_t> output_data(output_shape[0]);
        output_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, output_data.data(), output_data.size(),
            output_shape.data(), output_shape.size());
        // run
        vector<const char *> input_node_names_c;
        vector<const char *> output_node_names_c;
        for (const auto &name : input_node_names) {
          input_node_names_c.push_back(name.c_str());
        }
        for (const auto &name : output_node_names) {
          output_node_names_c.push_back(name.c_str());
        }

        try {
          session->Run(Ort::RunOptions{nullptr}, input_node_names_c.data(),
                       &input_tensor, 1, output_node_names_c.data(),
                       &output_tensor, 1);
        } catch (const Ort::Exception &e) {
          std::cerr << "Failed to run inference: " << e.what() << std::endl;
        }
        int64_t *output_data_ptr =
            output_tensor.GetTensorMutableData<int64_t>();
        vector<int64_t> inference_results(output_data_ptr,
                                          output_data_ptr + output_data.size());

        // Write output vector
        result.SetVectorType(VectorType::FLAT_VECTOR);

        if (inference_results.size() != args.size()) {
          std::cerr << "Error: Inference results size mismatch." << std::endl;
          return;
        }
        //   auto result_data = FlatVector::GetData<float_t>(result);
        auto result_data = FlatVector::GetData<float_t>(result);
        for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
          result_data[row_idx] = static_cast<float>(inference_results[row_idx]);
		  std::cout<<result_data[row_idx]<<" "<<std::endl;
        }
        result.Verify(args.size());

      } catch (const Ort::Exception &e) {
        std::cerr << "Failed to create output tensor: " << e.what()
                  << std::endl;
      }
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "Failed to get output shape: " << e.what() << std::endl;
  }
}

//===--------------------------------------------------------------------===//
// Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void
loadable_extension_demo_init(duckdb::DatabaseInstance &db) {
  // create a scalar function
  Connection con(db);
  auto &client_context = *con.context;
  auto &catalog = Catalog::GetSystemCatalog(client_context);
  con.BeginTransaction();

  // ** Function onnx inference
  // todo here return_type(LogicalType::FLOAT) is simplified for demo purpose.
  // todo Future will support return_type(LogicalType::STRUCT)
  ScalarFunction onnx_inference_fun("onnx",
                                    {LogicalType::VARCHAR, LogicalType::ANY},
                                    LogicalType::FLOAT, OnnxInferenceFunction);
  onnx_inference_fun.varargs = LogicalType::ANY;
  CreateScalarFunctionInfo onnx_inference_info(onnx_inference_fun);
  catalog.CreateFunction(client_context, onnx_inference_info);

  con.Commit();
}

DUCKDB_EXTENSION_API const char *loadable_extension_demo_version() {
  return DuckDB::LibraryVersion();
}
}
