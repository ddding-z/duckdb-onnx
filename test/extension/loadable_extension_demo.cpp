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
#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <unordered_map>

using namespace duckdb;

//===--------------------------------------------------------------------===//
// Scalar function
//===--------------------------------------------------------------------===//
inline int32_t hello_fun(string_t what) {
	return what.GetSize() + 5;
}

inline void TestAliasHello(DataChunk &args, ExpressionState &state, Vector &result) {
	result.Reference(Value("Hello Alias!"));
}

inline void AddPointFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &left_vector = args.data[0];
	auto &right_vector = args.data[1];
	const int count = args.size();
	auto left_vector_type = left_vector.GetVectorType();
	auto right_vector_type = right_vector.GetVectorType();

	UnifiedVectorFormat lhs_data;
	UnifiedVectorFormat rhs_data;
	left_vector.ToUnifiedFormat(count, lhs_data);
	right_vector.ToUnifiedFormat(count, rhs_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &child_entries = StructVector::GetEntries(result);
	auto &left_child_entries = StructVector::GetEntries(left_vector);
	auto &right_child_entries = StructVector::GetEntries(right_vector);
	for (int base_idx = 0; base_idx < count; base_idx++) {
		auto lhs_list_index = lhs_data.sel->get_index(base_idx);
		auto rhs_list_index = rhs_data.sel->get_index(base_idx);
		if (!lhs_data.validity.RowIsValid(lhs_list_index) || !rhs_data.validity.RowIsValid(rhs_list_index)) {
			FlatVector::SetNull(result, base_idx, true);
			continue;
		}
		for (size_t col = 0; col < child_entries.size(); ++col) {
			auto &child_entry = child_entries[col];
			auto &left_child_entry = left_child_entries[col];
			auto &right_child_entry = right_child_entries[col];
			auto pdata = ConstantVector::GetData<int32_t>(*child_entry);
			auto left_pdata = ConstantVector::GetData<int32_t>(*left_child_entry);
			auto right_pdata = ConstantVector::GetData<int32_t>(*right_child_entry);
			pdata[base_idx] = left_pdata[lhs_list_index] + right_pdata[rhs_list_index];
		}
	}
	if (left_vector_type == VectorType::CONSTANT_VECTOR && right_vector_type == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	result.Verify(count);
}

inline void SubPointFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &left_vector = args.data[0];
	auto &right_vector = args.data[1];
	const int count = args.size();
	auto left_vector_type = left_vector.GetVectorType();
	auto right_vector_type = right_vector.GetVectorType();

	UnifiedVectorFormat lhs_data;
	UnifiedVectorFormat rhs_data;
	left_vector.ToUnifiedFormat(count, lhs_data);
	right_vector.ToUnifiedFormat(count, rhs_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &child_entries = StructVector::GetEntries(result);
	auto &left_child_entries = StructVector::GetEntries(left_vector);
	auto &right_child_entries = StructVector::GetEntries(right_vector);
	for (int base_idx = 0; base_idx < count; base_idx++) {
		auto lhs_list_index = lhs_data.sel->get_index(base_idx);
		auto rhs_list_index = rhs_data.sel->get_index(base_idx);
		if (!lhs_data.validity.RowIsValid(lhs_list_index) || !rhs_data.validity.RowIsValid(rhs_list_index)) {
			FlatVector::SetNull(result, base_idx, true);
			continue;
		}
		for (size_t col = 0; col < child_entries.size(); ++col) {
			auto &child_entry = child_entries[col];
			auto &left_child_entry = left_child_entries[col];
			auto &right_child_entry = right_child_entries[col];
			auto pdata = ConstantVector::GetData<int32_t>(*child_entry);
			auto left_pdata = ConstantVector::GetData<int32_t>(*left_child_entry);
			auto right_pdata = ConstantVector::GetData<int32_t>(*right_child_entry);
			pdata[base_idx] = left_pdata[lhs_list_index] - right_pdata[rhs_list_index];
		}
	}
	if (left_vector_type == VectorType::CONSTANT_VECTOR && right_vector_type == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	result.Verify(count);
}

static std::vector<float_t> InferenceModelTest(const std::string &model_path) {
	// initialize ONNX Runtime
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceModel");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	std::cout << "Session initialized successfully." << std::endl;
	try {
		Ort::Session session(env, model_path.c_str(), session_options);
		std::cout << "Session created successfully." << std::endl;
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to create session: " << e.what() << std::endl;
	}
	// Fixed for demo test
	return {0.95f, 0.85f, 0.99f};
}

// Helper function to create tensor
template <typename T>
Ort::Value CreateTensorFromData(Ort::MemoryInfo &memory_info, const void *data, size_t size,
                                const vector<int64_t> &shape) {
	return Ort::Value::CreateTensor<T>(memory_info, (T *)data, size, shape.data(), shape.size());
}

// class ModelCache {
// public:
//     static const size_t capacity = 10;
//     static std::unordered_map<std::string, std::pair<std::shared_ptr<Ort::Session>,
//     std::list<std::string>::iterator>> modelcache; static std::list<std::string> usage_order; static
//     std::shared_mutex map_mutex;

//     static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &key, const Ort::Env &env,
//                                                             const Ort::SessionOptions &options) {
//         {
//             std::shared_lock<std::shared_mutex> sharedLock(map_mutex);
//             auto it = modelcache.find(key);
//             if (it != modelcache.end()) {
//                 sharedLock.unlock();
//                 std::unique_lock<std::shared_mutex> uniqueLock(map_mutex);
//                 it = modelcache.find(key);
//                 if (it != modelcache.end()) {
//                     usage_order.erase(it->second.second);
//                     usage_order.push_front(key);
//                     it->second.second = usage_order.begin();
//                     return it->second.first;
//                 }
//             }
//         }
//         std::unique_lock<std::shared_mutex> lock(map_mutex);
//         auto it = modelcache.find(key);
//         if (it != modelcache.end()) {
//             usage_order.erase(it->second.second);
//             usage_order.push_front(key);
//             it->second.second = usage_order.begin();
//             return it->second.first;
//         }

//         std::shared_ptr<Ort::Session> session;
//         try {
//             session = std::make_shared<Ort::Session>(env, key.c_str(), options);
//             // std::cout << key << " Session created successfully." << std::endl;
//         } catch (const Ort::Exception &e) {
//             std::cerr << "Failed to create session: " << e.what() << std::endl;
//             return nullptr;
//         }

//         if (modelcache.size() >= capacity) {
//             const std::string &lru_key = usage_order.back();
//             modelcache.erase(lru_key);
//             usage_order.pop_back();
//         }
//         usage_order.push_front(key);
//         modelcache[key] = std::make_pair(session, usage_order.begin());

//         return session;
//     }
// };
// // 在类外定义静态成员
// std::unordered_map<std::string, std::pair<std::shared_ptr<Ort::Session>, std::list<std::string>::iterator>>
// ModelCache::modelcache; std::list<std::string> ModelCache::usage_order; std::shared_mutex ModelCache::map_mutex;

class ModelCache {
public:
	static const size_t capacity = 10;
	static std::unordered_map<std::string, std::shared_ptr<Ort::Session>> modelcache;
	static std::queue<std::string> model_queue;
	static std::shared_mutex map_mutex;

	static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &key, const Ort::Env &env,
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
			// std::cout << key << " Session created successfully." << std::endl;
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
// 在类外定义静态成员
std::unordered_map<std::string, std::shared_ptr<Ort::Session>> ModelCache::modelcache;
std::queue<std::string> ModelCache::model_queue;
std::shared_mutex ModelCache::map_mutex;

// // no lrucache
// class ModelCache {
// public:
// 	// key:thread ID & model_path，value:session
// 	static std::map<std::tuple<std::thread::id, std::string>, std::shared_ptr<Ort::Session>> modelcache;
// 	static std::shared_mutex map_mutex;

// 	static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &model_path, const Ort::Env &env,
// 	                                                        const Ort::SessionOptions &options) {
// 		std::thread::id thread_id = std::this_thread::get_id();
// 		auto key = std::make_tuple(thread_id, model_path);
// 		// 读取缓存
// 		{
// 			std::shared_lock<std::shared_mutex> sharedLock(map_mutex);
// 			auto it = modelcache.find(key);
// 			if (it != modelcache.end()) {
// 				return it->second;
// 			}
// 		}
// 		// 创建新的session
// 		std::unique_lock<std::shared_mutex> uniqueLock(map_mutex);
// 		auto it = modelcache.find(key);
// 		if (it != modelcache.end()) {
// 			return it->second;
// 		}
// 		std::shared_ptr<Ort::Session> session;
// 		try {
// 			session = std::make_shared<Ort::Session>(env, model_path.c_str(), options);
// 			std::cout << model_path << " session created successfully for thread " << thread_id << std::endl;
// 		} catch (const Ort::Exception &e) {
// 			std::cerr << "Failed to create session for thread " << thread_id << ": " << e.what() << std::endl;
// 			return nullptr;
// 		}
// 		modelcache[key] = session;
// 		return session;
// 	}
// };
// std::map<std::tuple<std::thread::id, std::string>, std::shared_ptr<Ort::Session>> ModelCache::modelcache;
// std::shared_mutex ModelCache::map_mutex;

// class ModelCache {
// public:
// 	// cache capacity
// 	static const size_t capacity = 10;
// 	struct TupleHash {
// 		template <typename T1, typename T2>
// 		std::size_t operator()(const std::tuple<T1, T2> &tuple) const {
// 			std::size_t h1 = std::hash<T1> {}(std::get<0>(tuple));
// 			std::size_t h2 = std::hash<T2> {}(std::get<1>(tuple));
// 			return h1 ^ (h2 << 1);
// 		}
// 	};
// 	// key:thread ID & model_path，value: session
// 	static std::unordered_map<
// 	    std::tuple<std::thread::id, std::string>,
// 	    std::pair<std::shared_ptr<Ort::Session>, std::list<std::tuple<std::thread::id, std::string>>::iterator>,
// 	    TupleHash>
// 	    modelcache;

// 	static std::list<std::tuple<std::thread::id, std::string>> usage_order;
// 	static std::shared_mutex map_mutex; // 读写锁
// 	static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &model_path, const Ort::Env &env,
// 	                                                        const Ort::SessionOptions &options) {
// 		std::thread::id thread_id = std::this_thread::get_id();
// 		auto key = std::make_tuple(thread_id, model_path);
// 		// 加锁
// 		std::unique_lock<std::shared_mutex> lock(map_mutex);
// 		auto it = modelcache.find(key);
// 		if (it != modelcache.end()) {
// 			usage_order.erase(it->second.second);
// 			usage_order.push_front(key);
// 			it->second.second = usage_order.begin();
// 			return it->second.first;
// 		}
// 		if (modelcache.size() >= capacity) {
// 			auto lru_key = usage_order.back();
// 			usage_order.pop_back();
// 			modelcache.erase(lru_key);
// 		}
// 		std::shared_ptr<Ort::Session> session;
// 		try {
// 			session = std::make_shared<Ort::Session>(env, model_path.c_str(), options);
// 			std::cout << model_path << " session created successfully for thread " << thread_id << std::endl;
// 		} catch (const Ort::Exception &e) {
// 			std::cerr << "Failed to create session for thread " << thread_id << ": " << e.what() << std::endl;
// 			return nullptr;
// 		}
// 		usage_order.push_front(key);
// 		modelcache[key] = std::make_pair(session, usage_order.begin());

// 		return session;
// 	}
// };

// // 静态成员变量初始化
// const size_t ModelCache::capacity;
// std::unordered_map<
//     std::tuple<std::thread::id, std::string>,
//     std::pair<std::shared_ptr<Ort::Session>, std::list<std::tuple<std::thread::id, std::string>>::iterator>,
//     ModelCache::TupleHash>
//     ModelCache::modelcache;
// std::list<std::tuple<std::thread::id, std::string>> ModelCache::usage_order;
// std::shared_mutex ModelCache::map_mutex;

// Function to perform inference const vector<const void *> &input_buffers
vector<float> InferenceModel(const std::string &model_path, const vector<const void *> &input_buffers,
                             const vector<int64_t> &input_shape) {

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceModel");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	std::shared_ptr<Ort::Session> session = ModelCache::getOrCreateSession(model_path, env, session_options);
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
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor(nullptr);
	if (input_buffers.empty() || input_buffers[0] == nullptr) {
		std::cerr << "Error: Input buffers are empty or null." << std::endl;
		return {};
	}

	// 调整列存数据为行优先数据
	vector<float> row_major_data(adjusted_input_shape[0] * adjusted_input_shape[1]);

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
	Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto data_type = tensor_info.GetElementType();

	input_tensor = Ort::Value::CreateTensor<float>(memory_info,                 // 内存信息
	                                               row_major_data.data(),       // 指向行优先的数据
	                                               row_major_data.size(),       // 数据元素总数
	                                               adjusted_input_shape.data(), // 张量形状
	                                               adjusted_input_shape.size()  // 维度数
	);

	// 获取输出张量形状, 目前假设只有一个输出
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
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to get output shape: " << e.what() << std::endl;
		return {};
	}

	// 创建输出张量
	vector<float_t> output_data(output_shape[0]);
	Ort::Value output_tensor(nullptr);
	try {
		output_tensor = Ort::Value::CreateTensor<float_t>(memory_info, output_data.data(), output_data.size(),
		                                                  output_shape.data(), output_shape.size());
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to create output tensor: " << e.what() << std::endl;
		return {};
	}

	vector<const char *> input_node_names_c;
	vector<const char *> output_node_names_c;
	for (const auto &name : input_node_names) {
		input_node_names_c.push_back(name.c_str());
	}
	for (const auto &name : output_node_names) {
		output_node_names_c.push_back(name.c_str());
	}

	try {
		session->Run(Ort::RunOptions {nullptr}, input_node_names_c.data(), &input_tensor, 1, output_node_names_c.data(),
		             &output_tensor, 1);
	} catch (const Ort::Exception &e) {
		std::cerr << "Failed to run inference: " << e.what() << std::endl;
		return {};
	}
	// auto adjusted_output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
	// Extract output data
	float *output_data_ptr = output_tensor.GetTensorMutableData<float>();
	vector<float> results(output_data_ptr, output_data_ptr + output_data.size());

	// // Print for debugging
	// // 1. Print inpu
	// std::cout << "Input tensor shape: ";
	// for (const auto &dim : adjusted_input_shape) {
	// 	std::cout << dim << " ";
	// }
	// std::cout << std::endl;

	// std::cout << "Output tensor shape: ";
	// for (const auto &dim : adjusted_output_shape) {
	// 	std::cout << dim << " ";
	// }
	// std::cout << std::endl;

	// // 2. Print some of the input and output values for verification
	// auto input_data = input_tensor.GetTensorData<float>();
	// auto output_data_p = output_tensor.GetTensorData<float>();

	// std::cout << "Sample input values: ";
	// for (int i = 0; i < 11 && i < adjusted_input_shape[0] * adjusted_input_shape[1]; ++i) {
	// 	std::cout << input_data[i] << " ";
	// }
	// std::cout << std::endl;

	// std::cout << "Sample output values: ";
	// for (int i = 0; i < 10 && i < adjusted_output_shape[0] * adjusted_output_shape[1]; ++i) {
	// 	std::cout << output_data_p[i] << " ";
	// }
	// std::cout << std::endl;

	// std::cout << "Inference results (first 10 values): ";
	// for (size_t i = 0; i < std::min(results.size(), size_t(10)); ++i) {
	// 	std::cout << results[i] << " ";
	// }
	// std::cout << std::endl;

	return results;
}

static void OnnxInferenceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_path_vector = args.data[0];
	// check data type
	if (model_path_vector.GetType().id() != LogicalTypeId::VARCHAR) {
		std::cerr << "Error: Model path column must be VARCHAR type." << std::endl;
		return;
	}
	// Assume VARCHAR
	auto model_path_data = FlatVector::GetData<string_t>(model_path_vector);
	idx_t model_path_index = 0;
	// make sure index valid
	if (model_path_index >= args.size()) {
		std::cerr << "Error: Invalid model path index." << std::endl;
		return;
	}
	std::string model_path = model_path_data[model_path_index].GetString();
	vector<UnifiedVectorFormat> feature_data(args.ColumnCount() - 1);
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		// std::cout << "Processing column: " << col_idx << std::endl;
		// std::cout << "Data type: " << args.data[col_idx].GetType().ToString() << std::endl; // 打印数据类型
		args.data[col_idx].ToUnifiedFormat(args.size(), feature_data[col_idx - 1]);
	}
	// 构造直接指向 DuckDB 内存数据的指针数组，并验证指针
	vector<const void *> input_buffers;
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		const void *ptr = feature_data[col_idx - 1].data;
		input_buffers.push_back(ptr);
		// 假设数据是 float 类型
		// std::cout << "Pointer " << col_idx - 1 << " data: " << static_cast<const float *>(ptr)[0] << std::endl;
	}
	// for (const void *ptr : input_buffers) {
	// 	const float *data = static_cast<const float *>(ptr);
	// 	// std::cout << "Data: " << data[0] << std::endl;
	// }

	// For Demo Test
	// 使用 unique_ptr 管理 float 数组
	// vector<unique_ptr<float[]>> input_buffers;

	// for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
	// 	// 假设 feature_data[col_idx - 1].data 是动态分配的 float 数组
	// 	unique_ptr<float[]> ptr(feature_data[col_idx - 1].data);
	// 	input_buffers.push_back(std::move(ptr)); // 使用 std::move 转移所有权
	// 	// std::cout << "Pointer " << col_idx - 1 << " data: " << input_buffers.back()[0] << std::endl;
	// }

	// for (const auto &ptr : input_buffers) {
	// 	const float *data = ptr.get(); // 从 unique_ptr 获取原始指针
	// 	// std::cout << "Data: " << data[0] << std::endl;
	// }

	// vector<std::shared_ptr<float[]>> input_buffers; // 使用 shared_ptr 管理数据

	// for (int col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
	// 	// 创建一个 shared_ptr 来管理 feature_data[col_idx - 1].data
	// 	// std::shared_ptr<float> ptr(reinterpret_cast<float *>(feature_data[col_idx - 1].data));
	// 	// std::shared_ptr<float[]> ptr(reinterpret_cast<float *>(feature_data[col_idx - 1].data));
	// 	std::shared_ptr<float[]> ptr(reinterpret_cast<float *>(feature_data[col_idx - 1].data), [](float *p) {
	// 		free(p); // 使用 free 来释放 malloc 分配的内存
	// 	});

	// 	input_buffers.push_back(ptr);
	// }

	// input_shape: (batch_size, num_features)
	vector<int64_t> input_shape = {(int64_t)args.size(), (int64_t)args.ColumnCount() - 1};

	// Run inference
	auto inference_results = InferenceModel(model_path, input_buffers, input_shape);

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
}

//===--------------------------------------------------------------------===//
// Quack Table Function
//===--------------------------------------------------------------------===//
class QuackFunction : public TableFunction {
public:
	QuackFunction() {
		name = "quack";
		arguments.push_back(LogicalType::BIGINT);
		bind = QuackBind;
		init_global = QuackInit;
		function = QuackFunc;
	}

	struct QuackBindData : public TableFunctionData {
		QuackBindData(idx_t number_of_quacks) : number_of_quacks(number_of_quacks) {
		}

		idx_t number_of_quacks;
	};

	struct QuackGlobalData : public GlobalTableFunctionState {
		QuackGlobalData() : offset(0) {
		}

		idx_t offset;
	};

	static duckdb::unique_ptr<FunctionData> QuackBind(ClientContext &context, TableFunctionBindInput &input,
	                                                  vector<LogicalType> &return_types, vector<string> &names) {
		names.emplace_back("quack");
		return_types.emplace_back(LogicalType::VARCHAR);
		return make_uniq<QuackBindData>(BigIntValue::Get(input.inputs[0]));
	}

	static duckdb::unique_ptr<GlobalTableFunctionState> QuackInit(ClientContext &context,
	                                                              TableFunctionInitInput &input) {
		return make_uniq<QuackGlobalData>();
	}

	static void QuackFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
		auto &bind_data = data_p.bind_data->Cast<QuackBindData>();
		auto &data = (QuackGlobalData &)*data_p.global_state;
		if (data.offset >= bind_data.number_of_quacks) {
			// finished returning values
			return;
		}
		// start returning values
		// either fill up the chunk or return all the remaining columns
		idx_t count = 0;
		while (data.offset < bind_data.number_of_quacks && count < STANDARD_VECTOR_SIZE) {
			output.SetValue(0, count, Value("QUACK"));
			data.offset++;
			count++;
		}
		output.SetCardinality(count);
	}
};

//===--------------------------------------------------------------------===//
// Parser extension
//===--------------------------------------------------------------------===//
struct QuackExtensionData : public ParserExtensionParseData {
	QuackExtensionData(idx_t number_of_quacks) : number_of_quacks(number_of_quacks) {
	}

	idx_t number_of_quacks;

	duckdb::unique_ptr<ParserExtensionParseData> Copy() const override {
		return make_uniq<QuackExtensionData>(number_of_quacks);
	}
};

class QuackExtension : public ParserExtension {
public:
	QuackExtension() {
		parse_function = QuackParseFunction;
		plan_function = QuackPlanFunction;
	}

	static ParserExtensionParseResult QuackParseFunction(ParserExtensionInfo *info, const string &query) {
		auto lcase = StringUtil::Lower(StringUtil::Replace(query, ";", ""));
		if (!StringUtil::Contains(lcase, "quack")) {
			// quack not found!?
			if (StringUtil::Contains(lcase, "quac")) {
				// use our error
				return ParserExtensionParseResult("Did you mean... QUACK!?");
			}
			// use original error
			return ParserExtensionParseResult();
		}
		auto splits = StringUtil::Split(lcase, "quack");
		for (auto &split : splits) {
			StringUtil::Trim(split);
			if (!split.empty()) {
				// we only accept quacks here
				return ParserExtensionParseResult("This is not a quack: " + split);
			}
		}
		// QUACK
		return ParserExtensionParseResult(make_uniq<QuackExtensionData>(splits.size() + 1));
	}

	static ParserExtensionPlanResult QuackPlanFunction(ParserExtensionInfo *info, ClientContext &context,
	                                                   duckdb::unique_ptr<ParserExtensionParseData> parse_data) {
		auto &quack_data = (QuackExtensionData &)*parse_data;

		ParserExtensionPlanResult result;
		result.function = QuackFunction();
		result.parameters.push_back(Value::BIGINT(quack_data.number_of_quacks));
		result.requires_valid_transaction = false;
		result.return_type = StatementReturnType::QUERY_RESULT;
		return result;
	}
};

static set<string> test_loaded_extension_list;

class QuackLoadExtension : public ExtensionCallback {
	void OnExtensionLoaded(DatabaseInstance &db, const string &name) override {
		test_loaded_extension_list.insert(name);
	}
};

inline void LoadedExtensionsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	string result_str;
	for (auto &ext : test_loaded_extension_list) {
		if (!result_str.empty()) {
			result_str += ", ";
		}
		result_str += ext;
	}
	result.Reference(Value(result_str));
}

//===--------------------------------------------------------------------===//
// Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void loadable_extension_demo_init(duckdb::DatabaseInstance &db) {
	CreateScalarFunctionInfo hello_alias_info(
	    ScalarFunction("test_alias_hello", {}, LogicalType::VARCHAR, TestAliasHello));

	// create a scalar function
	Connection con(db);
	auto &client_context = *con.context;
	auto &catalog = Catalog::GetSystemCatalog(client_context);
	con.BeginTransaction();
	// con.CreateScalarFunction<int32_t, string_t>("hello", {LogicalType(LogicalTypeId::VARCHAR)},
	//                                             LogicalType(LogicalTypeId::INTEGER), &hello_fun);
	// catalog.CreateFunction(client_context, hello_alias_info);

	// // Add alias POINT type
	// string alias_name = "POINT";
	// child_list_t<LogicalType> child_types;
	// child_types.push_back(make_pair("x", LogicalType::INTEGER));
	// child_types.push_back(make_pair("y", LogicalType::INTEGER));
	// auto alias_info = make_uniq<CreateTypeInfo>();
	// alias_info->internal = true;
	// alias_info->name = alias_name;
	// LogicalType target_type = LogicalType::STRUCT(child_types);
	// target_type.SetAlias(alias_name);
	// alias_info->type = target_type;

	// catalog.CreateType(client_context, *alias_info);

	// // Function add point
	// ScalarFunction add_point_func("add_point", {target_type, target_type}, target_type, AddPointFunction);
	// CreateScalarFunctionInfo add_point_info(add_point_func);
	// catalog.CreateFunction(client_context, add_point_info);

	// // Function sub point
	// ScalarFunction sub_point_func("sub_point", {target_type, target_type}, target_type, SubPointFunction);
	// CreateScalarFunctionInfo sub_point_info(sub_point_func);
	// catalog.CreateFunction(client_context, sub_point_info);

	// Function onnx inference
	// here return_type(LogicalType::FLOAT) is simplified for demo purpose.
	// Future will support return_type(LogicalType::STRUCT)
	ScalarFunction onnx_inference_fun("onnx", {LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::FLOAT,
	                                  OnnxInferenceFunction);
	onnx_inference_fun.varargs = LogicalType::ANY; // 支持不定参数
	CreateScalarFunctionInfo onnx_inference_info(onnx_inference_fun);
	catalog.CreateFunction(client_context, onnx_inference_info);

	// // Function sub point
	// ScalarFunction loaded_extensions("loaded_extensions", {}, LogicalType::VARCHAR, LoadedExtensionsFunction);
	// CreateScalarFunctionInfo loaded_extensions_info(loaded_extensions);
	// catalog.CreateFunction(client_context, loaded_extensions_info);

	// // Quack function
	// QuackFunction quack_function;
	// CreateTableFunctionInfo quack_info(quack_function);
	// catalog.CreateTableFunction(client_context, quack_info);

	con.Commit();

	// // add a parser extension
	// auto &config = DBConfig::GetConfig(db);
	// config.parser_extensions.push_back(QuackExtension());
	// config.extension_callbacks.push_back(make_uniq<QuackLoadExtension>());
}

DUCKDB_EXTENSION_API const char *loadable_extension_demo_version() {
	return DuckDB::LibraryVersion();
}
}
