#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/parser/base_expression.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/plugin/physical/common/types/column/column_data_collection.hpp"
#include "onnx/checker.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/pass_manager.h"
#include "onnxoptimizer/pass_registry.h"

using namespace duckdb;

// whatever
#include <arpa/inet.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <netdb.h>
#include <netinet/in.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <set>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>

#ifdef __MVS__
#define _XOPEN_SOURCE_EXTENDED 1
#include <strings.h>
#endif

class WaggleExtension : public OptimizerExtension {
public:
	WaggleExtension() {
		optimize_function = WaggleOptimizeFunction;
	}

	static bool HasParquetScan(LogicalOperator &op) {
		if (op.type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = op.Cast<LogicalGet>();
			return get.function.name == "parquet_scan";
		}
		for (auto &child : op.children) {
			if (HasParquetScan(*child)) {
				return true;
			}
		}
		return false;
	}

	static void WriteChecked(int sockfd, void *data, idx_t write_size) {
		auto bytes_written = write(sockfd, data, write_size);
		if (bytes_written < 0) {
			throw InternalException("Failed to write \"%lld\" bytes to socket: %s", write_size, strerror(errno));
		}
		if (idx_t(bytes_written) != write_size) {
			throw InternalException("Failed to write \"%llu\" bytes from socket - wrote %llu instead", write_size,
			                        bytes_written);
		}
	}
	static void ReadChecked(int sockfd, void *data, idx_t read_size) {
		auto bytes_read = read(sockfd, data, read_size);
		if (bytes_read < 0) {
			throw InternalException("Failed to read \"%lld\" bytes from socket: %s", read_size, strerror(errno));
		}
		if (idx_t(bytes_read) != read_size) {
			throw InternalException("Failed to read \"%llu\" bytes from socket - read %llu instead", read_size,
			                        bytes_read);
		}
	}

	static void WaggleOptimizeFunction(ClientContext &context, OptimizerExtensionInfo *info,
	                                   duckdb::unique_ptr<LogicalOperator> &plan) {
		if (!HasParquetScan(*plan)) {
			return;
		}
		// rpc

		Value host, port;
		if (!context.TryGetCurrentSetting("waggle_location_host", host) ||
		    !context.TryGetCurrentSetting("waggle_location_port", port)) {
			throw InvalidInputException("Need the parameters damnit");
		}

		// socket create and verification
		auto sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sockfd == -1) {
			throw InternalException("Failed to create socket");
		}

		struct sockaddr_in servaddr;
		bzero(&servaddr, sizeof(servaddr));
		// assign IP, PORT
		servaddr.sin_family = AF_INET;
		auto host_string = host.ToString();
		servaddr.sin_addr.s_addr = inet_addr(host_string.c_str());
		servaddr.sin_port = htons(port.GetValue<int32_t>());

		// connect the client socket to server socket
		if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
			throw IOException("Failed to connect socket %s", string(strerror(errno)));
		}

		MemoryStream stream;
		BinarySerializer serializer(stream);
		serializer.Begin();
		plan->Serialize(serializer);
		serializer.End();
		auto data = stream.GetData();
		idx_t len = stream.GetPosition();

		WriteChecked(sockfd, &len, sizeof(idx_t));
		WriteChecked(sockfd, data, len);

		auto chunk_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator());
		idx_t n_chunks;
		ReadChecked(sockfd, &n_chunks, sizeof(idx_t));
		for (idx_t i = 0; i < n_chunks; i++) {
			idx_t chunk_len;
			ReadChecked(sockfd, &chunk_len, sizeof(idx_t));
			auto buffer = malloc(chunk_len);
			D_ASSERT(buffer);
			ReadChecked(sockfd, buffer, chunk_len);

			MemoryStream source(data_ptr_cast(buffer), chunk_len);
			DataChunk chunk;

			BinaryDeserializer deserializer(source);

			deserializer.Begin();
			chunk.Deserialize(deserializer);
			deserializer.End();
			chunk_collection->Initialize(chunk.GetTypes());
			chunk_collection->Append(chunk);
			free(buffer);
		}

		auto types = chunk_collection->Types();
		plan = make_uniq<LogicalColumnDataGet>(0, types, std::move(chunk_collection));

		len = 0;
		(void)len;
		WriteChecked(sockfd, &len, sizeof(idx_t));
		// close the socket
		close(sockfd);
	}
};

// -----------------------------------------------------------------
/**
 * ToDo:
 * node list -> tree node
 *  */
namespace onnx::optimization {

class OnnxExtension : public OptimizerExtension {
public:
	OnnxExtension() {
		optimize_function = OnnxOptimizeFunction;

		// 初始化 comparison_funcs
		comparison_funcs[ExpressionType::COMPARE_LESSTHAN] = [](float_t x, float_t y) -> bool {
			return x < y;
		};
		comparison_funcs[ExpressionType::COMPARE_GREATERTHAN] = [](float_t x, float_t y) -> bool {
			return x > y;
		};
		comparison_funcs[ExpressionType::COMPARE_LESSTHANOREQUALTO] = [](float_t x, float_t y) -> bool {
			return x <= y;
		};
		comparison_funcs[ExpressionType::COMPARE_GREATERTHANOREQUALTO] = [](float_t x, float_t y) -> bool {
			return x >= y;
		};
	}
	static std::string onnx_model_path;
	static std::string new_model_path;
	static float_t predicate;
	static vector<std::string> removed_nodes;
	// static std::vector<std::string> columns_to_remove;

	static std::vector<int64_t> left_nodes;
	static std::vector<int64_t> right_nodes;
	static std::vector<std::string> node_types;
	static std::vector<double> node_thresholds;
	static std::vector<int64_t> target_nodeids;
	static std::vector<double> target_weights;

	static ExpressionType ComparisonOperator;
	static std::unordered_map<ExpressionType, std::function<bool(float_t, float_t)>> comparison_funcs;

	struct NodeID {
		int id;
		std::string node;
	};

	// TODO: need to support nested case
	static bool HasONNXFilter(LogicalOperator &op) {
		// std::cout << "Start HasONNXFilter" << std::endl;
		for (auto &expr : op.expressions) {
			if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
				auto &comparison_expr = static_cast<BoundComparisonExpression &>(*expr);
				if (comparison_expr.left->expression_class == ExpressionClass::BOUND_FUNCTION) {
					auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
					if (func_expr.function.name == "onnx" && func_expr.children.size() > 1) {
						auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
						if (first_param.value.type().id() == LogicalTypeId::VARCHAR) {
							std::string model_path = first_param.value.ToString();
							if (comparison_expr.right->type == ExpressionType::VALUE_CONSTANT) {
								auto &constant_expr = (BoundConstantExpression &)*comparison_expr.right;
								predicate = constant_expr.value.GetValue<float_t>();
								// // for test
								// if (predicate == 0.0f) {
								// 	return false;
								// }
								onnx_model_path = model_path;
								ComparisonOperator = comparison_expr.type;
								// set comparison_expr => =1
								comparison_expr.type = ExpressionType::COMPARE_EQUAL;
								duckdb::Value value(1.0f);
								auto new_constant_expr = std::make_unique<duckdb::BoundConstantExpression>(value);
								comparison_expr.right = std::move(new_constant_expr);
								// 获取时间戳
								// std::time_t now_time = std::time(nullptr);
								// size_t pos = onnx_model_path.find(".onnx");
								// std::string model_name = onnx_model_path.substr(0, pos);
								// new_model_path = model_name + "_" + std::to_string(now_time) + ".onnx";

								boost::uuids::uuid uuid = boost::uuids::random_generator()();
								size_t pos = onnx_model_path.find(".onnx");
								std::string model_name = onnx_model_path.substr(0, pos);
								new_model_path = model_name + "_" + boost::uuids::to_string(uuid) + ".onnx";

								duckdb::Value model_path_value(new_model_path);
								first_param.value = model_path_value;
								return true;
							}
						}
					}
				}
			}
		}
		// 递归检查子节点
		for (auto &child : op.children) {
			if (HasONNXFilter(*child)) {
				return true;
			}
		}
		return false;
	}

	static int pruning(size_t node_id, size_t depth, vector<std::string> &result_nodes,
	                   onnx::graph_node_list &node_list, float_t predicate) {
		// get attribute
		for (auto node : node_list) {
			for (auto name : node->attributeNames()) {
				if (strcmp(name.toString(), "nodes_truenodeids") == 0) {
					left_nodes = node->is(name);
				}
				if (strcmp(name.toString(), "nodes_falsenodeids") == 0) {
					right_nodes = node->is(name);
				}
				if (strcmp(name.toString(), "nodes_modes") == 0) {
					node_types = node->ss(name);
				}
				if (strcmp(name.toString(), "nodes_values") == 0) {
					node_thresholds = node->fs(name);
				}
				if (strcmp(name.toString(), "target_nodeids") == 0) {
					target_nodeids = node->is(name);
				}
				if (strcmp(name.toString(), "target_weights") == 0) {
					target_weights = node->fs(name);
				}
			}
		}
		result_nodes[node_id] = node_types[node_id];
		auto is_leaf = node_types[node_id] == "LEAF";
		if (is_leaf) {
			auto target_id = -1;
			for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
				int ni = target_nodeids[ti];
				if (ni == node_id) {
					target_id = static_cast<int>(ti);
					break;
				}
			}
			// modified
			auto result = static_cast<int>(comparison_funcs[ComparisonOperator](target_weights[target_id], predicate));
			result == 1 ? result_nodes[node_id] = "LEAF_TRUE" : result_nodes[node_id] = "LEAF_FALSE";
			// std::cout << "node_id: " << node_id << ", depth: " << depth << ", is_leaf: " << (is_leaf ? "true" :
			// "false")
			//           << ", result: " << result << std::endl;
			return result;
		} else {
			auto left_node_id = left_nodes[node_id];
			auto left_result = pruning(left_node_id, depth + 1, result_nodes, node_list, predicate);
			auto right_node_id = right_nodes[node_id];
			auto right_result = pruning(right_node_id, depth + 1, result_nodes, node_list, predicate);

			if (left_result == 0 && right_result == 0) {
				// std::cout << "node_id: " << node_id << ", depth: " << depth
				//           << ", is_leaf: " << (is_leaf ? "true" : "false") << ", result: " << 0 << std::endl;
				result_nodes[node_id] = "LEAF_FALSE";
				result_nodes[left_node_id] = "REMOVED";
				result_nodes[right_node_id] = "REMOVED";
				return 0;
			}

			if (left_result == 1 && right_result == 1) {
				// std::cout << "node_id: " << node_id << ", depth: " << depth
				//           << ", is_leaf: " << (is_leaf ? "true" : "false") << ", result: " << 1 << std::endl;
				result_nodes[node_id] = "LEAF_TRUE";
				result_nodes[left_node_id] = "REMOVED";
				result_nodes[right_node_id] = "REMOVED";
				return 1;
			}
			// std::cout << "node_id: " << node_id << ", depth: " << depth + 1 << ", leaf_depth: " << depth + 1
			//           << std::endl;
			return 2;
		}
	}

	static void OnnxPruneFunction() {
		// load model
		ModelProto model;
		onnx::optimization::loadModel(&model, onnx_model_path, true);
		std::shared_ptr<Graph> graph(ImportModelProto(model));
		auto node_list = graph->nodes();
		size_t length;
		for (auto node : node_list) {
			for (auto name : node->attributeNames()) {
				if (strcmp(name.toString(), "nodes_modes") == 0) {
					length = node->ss(name).size();
					break;
				}
			}
		}
		vector<std::string> result_nodes {length, ""};
		pruning(0, 0, result_nodes, node_list, predicate);
		removed_nodes = result_nodes;
	}

	static void reg2reg(std::string &model_path, onnx::graph_node_list &node_list) {
		int64_t input_n_targets;
		std::vector<int64_t> input_nodes_falsenodeids;
		std::vector<int64_t> input_nodes_featureids;
		std::vector<double> input_nodes_hitrates;
		std::vector<int64_t> input_nodes_missing_value_tracks_true;
		std::vector<std::string> input_nodes_modes;
		std::vector<int64_t> input_nodes_nodeids;
		std::vector<int64_t> input_nodes_treeids;
		std::vector<int64_t> input_nodes_truenodeids;
		std::vector<double> input_nodes_values;
		std::string input_post_transform;
		std::vector<int64_t> input_target_ids;
		std::vector<int64_t> input_target_nodeids;
		std::vector<int64_t> input_target_treeids;
		std::vector<double> input_target_weights;

		std::unordered_map<std::string, int> attr_map = {{"n_targets", 1},
		                                                 {"nodes_falsenodeids", 2},
		                                                 {"nodes_featureids", 3},
		                                                 {"nodes_hitrates", 4},
		                                                 {"nodes_missing_value_tracks_true", 5},
		                                                 {"nodes_modes", 6},
		                                                 {"nodes_nodeids", 7},
		                                                 {"nodes_treeids", 8},
		                                                 {"nodes_truenodeids", 9},
		                                                 {"nodes_values", 10},
		                                                 {"post_transform", 11},
		                                                 {"target_ids", 12},
		                                                 {"target_nodeids", 13},
		                                                 {"target_treeids", 14},
		                                                 {"target_weights", 15}};

		for (auto node : node_list) {
			for (auto name : node->attributeNames()) {
				std::string attr_name = name.toString();
				auto it = attr_map.find(attr_name);
				if (it != attr_map.end()) {
					switch (it->second) {
					case 1:
						input_n_targets = node->i(name);
						break;
					case 2:
						input_nodes_falsenodeids = node->is(name);
						break;
					case 3:
						input_nodes_featureids = node->is(name);
						break;
					case 4:
						input_nodes_hitrates = node->fs(name);
						break;
					case 5:
						input_nodes_missing_value_tracks_true = node->is(name);
						break;
					case 6:
						input_nodes_modes = node->ss(name);
						break;
					case 7:
						input_nodes_nodeids = node->is(name);
						break;
					case 8:
						input_nodes_treeids = node->is(name);
						break;
					case 9:
						input_nodes_truenodeids = node->is(name);
						break;
					case 10:
						input_nodes_values = node->fs(name);
						break;
					case 11:
						input_post_transform = node->s(name);
						break;
					case 12:
						input_target_ids = node->is(name);
						break;
					case 13:
						input_target_nodeids = node->is(name);
						break;
					case 14:
						input_target_treeids = node->is(name);
						break;
					case 15:
						input_target_weights = node->fs(name);
						break;
					default:
						break;
					}
				}
			}
		}

		// for (auto node : node_list) {
		// 	for (auto name : node->attributeNames()) {
		// 		if (strcmp(name.toString(), "n_targets") == 0) {
		// 			input_n_targets = node->i(name);
		// 		} else if (strcmp(name.toString(), "nodes_falsenodeids") == 0) {
		// 			input_nodes_falsenodeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_featureids") == 0) {
		// 			input_nodes_featureids = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_hitrates") == 0) {
		// 			input_nodes_hitrates = node->fs(name);
		// 		} else if (strcmp(name.toString(), "nodes_missing_value_tracks_true") == 0) {
		// 			input_nodes_missing_value_tracks_true = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_modes") == 0) {
		// 			input_nodes_modes = node->ss(name);
		// 		} else if (strcmp(name.toString(), "nodes_nodeids") == 0) {
		// 			input_nodes_nodeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_treeids") == 0) {
		// 			input_nodes_treeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_truenodeids") == 0) {
		// 			input_nodes_truenodeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "nodes_values") == 0) {
		// 			input_nodes_values = node->fs(name);
		// 		} else if (strcmp(name.toString(), "post_transform") == 0) {
		// 			input_post_transform = node->s(name);
		// 		} else if (strcmp(name.toString(), "target_ids") == 0) {
		// 			input_target_ids = node->is(name);
		// 		} else if (strcmp(name.toString(), "target_nodeids") == 0) {
		// 			input_target_nodeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "target_treeids") == 0) {
		// 			input_target_treeids = node->is(name);
		// 		} else if (strcmp(name.toString(), "target_weights") == 0) {
		// 			input_target_weights = node->fs(name);
		// 		}
		// 	}
		// }

		// 1. 计算 leaf_count
		int leaf_count = std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE") +
		                 std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");

		// 2. 构建 new_ids
		vector<NodeID> new_ids;
		int id_ = 0;
		for (const auto &node : removed_nodes) {
			if (node == "LEAF_FALSE" || node == "LEAF_TRUE" || node == "BRANCH_LEQ") {
				new_ids.push_back({id_, node});
				id_++;
			} else {
				new_ids.push_back({-1, node});
			}
		}

		// 3. 赋值 n_targets
		int n_targets = input_n_targets;

		// 4. 构建 nodes_falsenodeids
		vector<int> nodes_falsenodeids;
		for (size_t i = 0; i < input_nodes_falsenodeids.size(); ++i) {
			int ii = input_nodes_falsenodeids[i];
			if (new_ids[i].node != "REMOVED") {
				int value = 0;
				if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size()) {
					int new_id_value = new_ids[ii].id;
					value = (new_id_value != -1) ? new_id_value : 0;
				}
				nodes_falsenodeids.push_back(value);
			}
		}

		// 5. 构建 nodes_featureids
		vector<int> nodes_featureids;
		for (size_t i = 0; i < input_nodes_featureids.size(); ++i) {
			int ii = input_nodes_featureids[i];
			if (new_ids[i].id != -1) {
				int value = (new_ids[i].node == "BRANCH_LEQ") ? ii : 0;
				nodes_featureids.push_back(value);
			}
		}

		// 6. 构建 nodes_hitrates
		vector<float> nodes_hitrates;
		for (size_t i = 0; i < input_nodes_hitrates.size(); ++i) {
			if (new_ids[i].id != -1) {
				nodes_hitrates.push_back(input_nodes_hitrates[i]);
			}
		}

		// 7. 构建 nodes_missing_value_tracks_true
		vector<int> nodes_missing_value_tracks_true;
		for (size_t i = 0; i < input_nodes_missing_value_tracks_true.size(); ++i) {
			if (new_ids[i].id != -1) {
				nodes_missing_value_tracks_true.push_back(input_nodes_missing_value_tracks_true[i]);
			}
		}

		// 8. 构建 nodes_modes
		vector<std::string> nodes_modes;
		for (const auto &new_id : new_ids) {
			if (new_id.id != -1) {
				std::string mode = (new_id.node == "BRANCH_LEQ") ? "BRANCH_LEQ" : "LEAF";
				nodes_modes.push_back(mode);
			}
		}

		// 9. 构建 nodes_nodeids
		vector<int> nodes_nodeids;
		for (size_t i = 0; i < input_nodes_nodeids.size(); ++i) {
			if (new_ids[i].id != -1) {
				nodes_nodeids.push_back(new_ids[i].id);
			}
		}

		// 10. 构建 nodes_treeids
		vector<int> nodes_treeids;
		for (size_t i = 0; i < input_nodes_treeids.size(); ++i) {
			if (new_ids[i].id != -1) {
				nodes_treeids.push_back(input_nodes_treeids[i]);
			}
		}

		// 11. 构建 nodes_truenodeids
		vector<int> nodes_truenodeids;
		for (size_t i = 0; i < input_nodes_truenodeids.size(); ++i) {
			int ii = input_nodes_truenodeids[i];
			if (new_ids[i].node != "REMOVED") {
				int value = 0;
				if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size()) {
					int new_id_value = new_ids[ii].id;
					value = (new_id_value != -1) ? new_id_value : 0;
				}
				nodes_truenodeids.push_back(value);
			}
		}

		// 12. 构建 nodes_values
		vector<float> nodes_values;
		for (size_t i = 0; i < input_nodes_values.size(); ++i) {
			if (new_ids[i].id != -1) {
				float value = (new_ids[i].node == "BRANCH_LEQ") ? input_nodes_values[i] : 0.0f;
				nodes_values.push_back(value);
			}
		}

		// 13. 赋值 post_transform
		string post_transform = input_post_transform;

		// 14. 构建 target_ids
		vector<int> target_ids(leaf_count, 0);

		// 15. 构建 target_nodeids
		vector<int> target_nodeids;
		for (const auto &new_id : new_ids) {
			if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
				target_nodeids.push_back(new_id.id);
			}
		}

		// 16. 构建 target_treeids
		vector<int> target_treeids(leaf_count, 0);

		// 17. 构建 target_weights
		vector<float> target_weights;
		for (const auto &new_id : new_ids) {
			if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
				float weight = (new_id.node == "LEAF_TRUE") ? 1.0f : 0.0f;
				target_weights.push_back(weight);
			}
		}

		// ------------------

		// load initial model
		ModelProto initial_model;
		onnx::optimization::loadModel(&initial_model, model_path, true);
		GraphProto *initial_graph = initial_model.mutable_graph();

		ModelProto model;
		GraphProto *graph = model.mutable_graph();
		model.set_ir_version(initial_model.ir_version());

		for (const auto &input : initial_graph->input()) {
			onnx::ValueInfoProto *new_input = graph->add_input();
			new_input->CopyFrom(input);
		}

		for (const auto &output : initial_graph->output()) {
			onnx::ValueInfoProto *new_output = graph->add_output();
			new_output->CopyFrom(output);
		}

		for (const auto &initializer : initial_graph->initializer()) {
			onnx::TensorProto *new_initializer = graph->add_initializer();
			new_initializer->CopyFrom(initializer);
		}

		// 设置新模型的opset_import
		*model.mutable_opset_import() = initial_model.opset_import();

		// 3. 添加 TreeEnsembleRegressor 节点
		NodeProto new_node;
		auto initial_node = initial_graph->node()[0];
		new_node.set_op_type(initial_node.op_type());
		new_node.set_domain(initial_node.domain());    // 设置 domain 为 ai.onnx.ml
		new_node.set_name(initial_node.name());        // 设置节点名称
		new_node.add_input(initial_node.input()[0]);   // 输入
		new_node.add_output(initial_node.output()[0]); // 输出

		// 设置节点属性
		// 1. n_targets
		AttributeProto attr_n_targets;
		attr_n_targets.set_name("n_targets");
		attr_n_targets.set_type(AttributeProto::INT);
		attr_n_targets.set_i(n_targets);
		*new_node.add_attribute() = attr_n_targets;

		// 2. nodes_falsenodeids
		AttributeProto attr_nodes_falsenodeids;
		attr_nodes_falsenodeids.set_name("nodes_falsenodeids");
		attr_nodes_falsenodeids.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_falsenodeids) {
			attr_nodes_falsenodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_falsenodeids;

		// 3. nodes_featureids
		AttributeProto attr_nodes_featureids;
		attr_nodes_featureids.set_name("nodes_featureids");
		attr_nodes_featureids.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_featureids) {
			attr_nodes_featureids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_featureids;

		// 4. nodes_hitrates
		AttributeProto attr_nodes_hitrates;
		attr_nodes_hitrates.set_name("nodes_hitrates");
		attr_nodes_hitrates.set_type(AttributeProto::FLOATS);
		for (const auto &rate : nodes_hitrates) {
			attr_nodes_hitrates.add_floats(rate);
		}
		*new_node.add_attribute() = attr_nodes_hitrates;

		// 5. nodes_missing_value_tracks_true
		AttributeProto attr_nodes_missing_value_tracks_true;
		attr_nodes_missing_value_tracks_true.set_name("nodes_missing_value_tracks_true");
		attr_nodes_missing_value_tracks_true.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_missing_value_tracks_true) {
			attr_nodes_missing_value_tracks_true.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_missing_value_tracks_true;

		// 6. nodes_modes
		AttributeProto attr_nodes_modes;
		attr_nodes_modes.set_name("nodes_modes");
		attr_nodes_modes.set_type(AttributeProto::STRINGS);
		for (const auto &mode : nodes_modes) {
			attr_nodes_modes.add_strings(mode);
		}
		*new_node.add_attribute() = attr_nodes_modes;

		// 7. nodes_nodeids
		AttributeProto attr_nodes_nodeids;
		attr_nodes_nodeids.set_name("nodes_nodeids");
		attr_nodes_nodeids.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_nodeids) {
			attr_nodes_nodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_nodeids;

		// 8. nodes_treeids
		AttributeProto attr_nodes_treeids;
		attr_nodes_treeids.set_name("nodes_treeids");
		attr_nodes_treeids.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_treeids) {
			attr_nodes_treeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_treeids;

		// 9. nodes_truenodeids
		AttributeProto attr_nodes_truenodeids;
		attr_nodes_truenodeids.set_name("nodes_truenodeids");
		attr_nodes_truenodeids.set_type(AttributeProto::INTS);
		for (const auto &id : nodes_truenodeids) {
			attr_nodes_truenodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_truenodeids;

		// 10. nodes_values
		AttributeProto attr_nodes_values;
		attr_nodes_values.set_name("nodes_values");
		attr_nodes_values.set_type(AttributeProto::FLOATS);
		for (const auto &val : nodes_values) {
			attr_nodes_values.add_floats(val);
		}
		*new_node.add_attribute() = attr_nodes_values;

		// 11. post_transform
		AttributeProto attr_post_transform;
		attr_post_transform.set_name("post_transform");
		attr_post_transform.set_type(AttributeProto::STRING);
		attr_post_transform.set_s(post_transform);
		*new_node.add_attribute() = attr_post_transform;

		// 12. target_ids
		AttributeProto attr_target_ids;
		attr_target_ids.set_name("target_ids");
		attr_target_ids.set_type(AttributeProto::INTS);
		for (const auto &id : target_ids) {
			attr_target_ids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_ids;

		// 13. target_nodeids
		AttributeProto attr_target_nodeids;
		attr_target_nodeids.set_name("target_nodeids");
		attr_target_nodeids.set_type(AttributeProto::INTS);
		for (const auto &id : target_nodeids) {
			attr_target_nodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_nodeids;

		// 14. target_treeids
		AttributeProto attr_target_treeids;
		attr_target_treeids.set_name("target_treeids");
		attr_target_treeids.set_type(AttributeProto::INTS);
		for (const auto &id : target_treeids) {
			attr_target_treeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_treeids;

		// 15. target_weights
		AttributeProto attr_target_weights;
		attr_target_weights.set_name("target_weights");
		attr_target_weights.set_type(AttributeProto::FLOATS);
		for (const auto &weight : target_weights) {
			attr_target_weights.add_floats(weight);
		}
		*new_node.add_attribute() = attr_target_weights;

		// 将新节点添加到图中
		graph->add_node()->CopyFrom(new_node);

		saveModel(&model, new_model_path);
	}

	// constrcut pruned model
	static void OnnxConstructFunction() {
		ModelProto model;
		onnx::optimization::loadModel(&model, onnx_model_path, true);
		std::shared_ptr<Graph> graph(ImportModelProto(model));
		auto node_list = graph->nodes();
		reg2reg(onnx_model_path, node_list);
	}

	// Optimizer Function
	static void OnnxOptimizeFunction(ClientContext &context, OptimizerExtensionInfo *info,
	                                 duckdb::unique_ptr<LogicalOperator> &plan) {
		// auto onnx_info = dynamic_cast<ONNXOptimizerExtensionInfo *>(info);
		if (!HasONNXFilter(*plan)) {
			return;
		}
		OnnxPruneFunction();
		OnnxConstructFunction();
	}
};

std::string OnnxExtension::onnx_model_path;
std::string OnnxExtension::new_model_path;
float_t OnnxExtension::predicate;
vector<std::string> OnnxExtension::removed_nodes;
// std::vector<std::string> OnnxExtension::columns_to_remove;

std::vector<int64_t> OnnxExtension::left_nodes;
std::vector<int64_t> OnnxExtension::right_nodes;
std::vector<std::string> OnnxExtension::node_types;
std::vector<double> OnnxExtension::node_thresholds;
std::vector<int64_t> OnnxExtension::target_nodeids;
std::vector<double> OnnxExtension::target_weights;

ExpressionType OnnxExtension::ComparisonOperator;
std::unordered_map<ExpressionType, std::function<bool(float_t, float_t)>> OnnxExtension::comparison_funcs;

// -------------------------------------------------------------------

class UnusedColumnsExtension : public OptimizerExtension {
public:
	UnusedColumnsExtension() {
		optimize_function = UnusedColumnsFunction;
	}
	static std::string onnx_model_path;
	static std::string new_model_path;
	// static std::vector<int64_t> nodes_featureids;

	static void reg2reg(std::string &model_path, onnx::graph_node_list &node_list,
	                    std::vector<int64_t> &output_nodes_featureids, std::size_t dim_value) {
		int64_t input_n_targets;
		std::vector<int64_t> input_nodes_falsenodeids;
		// std::vector<int64_t> input_nodes_featureids;
		std::vector<double> input_nodes_hitrates;
		std::vector<int64_t> input_nodes_missing_value_tracks_true;
		std::vector<std::string> input_nodes_modes;
		std::vector<int64_t> input_nodes_nodeids;
		std::vector<int64_t> input_nodes_treeids;
		std::vector<int64_t> input_nodes_truenodeids;
		std::vector<double> input_nodes_values;
		std::string input_post_transform;
		std::vector<int64_t> input_target_ids;
		std::vector<int64_t> input_target_nodeids;
		std::vector<int64_t> input_target_treeids;
		std::vector<double> input_target_weights;

		std::unordered_map<std::string, int> attr_map = {{"n_targets", 1},
		                                                 {"nodes_falsenodeids", 2},
		                                                 {"nodes_featureids", 3},
		                                                 {"nodes_hitrates", 4},
		                                                 {"nodes_missing_value_tracks_true", 5},
		                                                 {"nodes_modes", 6},
		                                                 {"nodes_nodeids", 7},
		                                                 {"nodes_treeids", 8},
		                                                 {"nodes_truenodeids", 9},
		                                                 {"nodes_values", 10},
		                                                 {"post_transform", 11},
		                                                 {"target_ids", 12},
		                                                 {"target_nodeids", 13},
		                                                 {"target_treeids", 14},
		                                                 {"target_weights", 15}};
		for (auto node : node_list) {
			for (auto name : node->attributeNames()) {
				std::string attr_name = name.toString();
				auto it = attr_map.find(attr_name);
				if (it != attr_map.end()) {
					switch (it->second) {
					case 1:
						input_n_targets = node->i(name);
						break;
					case 2:
						input_nodes_falsenodeids = node->is(name);
						break;
					// case 3:
					// 	input_nodes_featureids = node->is(name);
					// 	break;
					case 4:
						input_nodes_hitrates = node->fs(name);
						break;
					case 5:
						input_nodes_missing_value_tracks_true = node->is(name);
						break;
					case 6:
						input_nodes_modes = node->ss(name);
						break;
					case 7:
						input_nodes_nodeids = node->is(name);
						break;
					case 8:
						input_nodes_treeids = node->is(name);
						break;
					case 9:
						input_nodes_truenodeids = node->is(name);
						break;
					case 10:
						input_nodes_values = node->fs(name);
						break;
					case 11:
						input_post_transform = node->s(name);
						break;
					case 12:
						input_target_ids = node->is(name);
						break;
					case 13:
						input_target_nodeids = node->is(name);
						break;
					case 14:
						input_target_treeids = node->is(name);
						break;
					case 15:
						input_target_weights = node->fs(name);
						break;
					default:
						break;
					}
				}
			}
		}
		// load initial model
		ModelProto initial_model;
		onnx::optimization::loadModel(&initial_model, model_path, true);
		GraphProto *initial_graph = initial_model.mutable_graph();

		ModelProto model;
		GraphProto *graph = model.mutable_graph();
		model.set_ir_version(initial_model.ir_version());

		for (const auto &input : initial_graph->input()) {
			onnx::ValueInfoProto *new_input = graph->add_input();
			new_input->set_name(input.name());

			// 设置类型 (TypeProto)
			onnx::TypeProto *new_type = new_input->mutable_type();
			new_type->CopyFrom(input.type());

			// 修改Tensor维度
			if (new_type->has_tensor_type()) {
				auto *tensor_type = new_type->mutable_tensor_type();
				onnx::TensorShapeProto temp_shape = tensor_type->shape();
				tensor_type->clear_shape();
				for (int i = 0; i < temp_shape.dim_size(); ++i) {
					const auto &dim = temp_shape.dim(i);
					auto *new_dim = tensor_type->mutable_shape()->add_dim();
					if (dim.has_dim_value()) {
						new_dim->set_dim_value(dim_value);
					} else if (dim.has_dim_param()) {
						new_dim->set_dim_param(dim.dim_param());
					}
				}
			}
		}

		for (const auto &output : initial_graph->output()) {
			onnx::ValueInfoProto *new_output = graph->add_output();
			new_output->CopyFrom(output);
		}

		for (const auto &initializer : initial_graph->initializer()) {
			onnx::TensorProto *new_initializer = graph->add_initializer();
			new_initializer->CopyFrom(initializer);
		}

		// 设置新模型的opset_import
		*model.mutable_opset_import() = initial_model.opset_import();

		// 3. 添加 TreeEnsembleRegressor 节点
		NodeProto new_node;
		auto initial_node = initial_graph->node()[0];
		new_node.set_op_type(initial_node.op_type());
		new_node.set_domain(initial_node.domain());    // 设置 domain 为 ai.onnx.ml
		new_node.set_name(initial_node.name());        // 设置节点名称
		new_node.add_input(initial_node.input()[0]);   // 输入
		new_node.add_output(initial_node.output()[0]); // 输出

		// 设置节点属性
		// 1. n_targets
		AttributeProto attr_n_targets;
		attr_n_targets.set_name("n_targets");
		attr_n_targets.set_type(AttributeProto::INT);
		attr_n_targets.set_i(input_n_targets);
		*new_node.add_attribute() = attr_n_targets;

		// 2. nodes_falsenodeids
		AttributeProto attr_nodes_falsenodeids;
		attr_nodes_falsenodeids.set_name("nodes_falsenodeids");
		attr_nodes_falsenodeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_nodes_falsenodeids) {
			attr_nodes_falsenodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_falsenodeids;

		// 3. nodes_featureids
		AttributeProto attr_nodes_featureids;
		attr_nodes_featureids.set_name("nodes_featureids");
		attr_nodes_featureids.set_type(AttributeProto::INTS);
		for (const auto &id : output_nodes_featureids) {
			attr_nodes_featureids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_featureids;

		// 4. nodes_hitrates
		AttributeProto attr_nodes_hitrates;
		attr_nodes_hitrates.set_name("nodes_hitrates");
		attr_nodes_hitrates.set_type(AttributeProto::FLOATS);
		for (const auto &rate : input_nodes_hitrates) {
			attr_nodes_hitrates.add_floats(rate);
		}
		*new_node.add_attribute() = attr_nodes_hitrates;

		// 5. nodes_missing_value_tracks_true
		AttributeProto attr_nodes_missing_value_tracks_true;
		attr_nodes_missing_value_tracks_true.set_name("nodes_missing_value_tracks_true");
		attr_nodes_missing_value_tracks_true.set_type(AttributeProto::INTS);
		for (const auto &id : input_nodes_missing_value_tracks_true) {
			attr_nodes_missing_value_tracks_true.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_missing_value_tracks_true;

		// 6. nodes_modes
		AttributeProto attr_nodes_modes;
		attr_nodes_modes.set_name("nodes_modes");
		attr_nodes_modes.set_type(AttributeProto::STRINGS);
		for (const auto &mode : input_nodes_modes) {
			attr_nodes_modes.add_strings(mode);
		}
		*new_node.add_attribute() = attr_nodes_modes;

		// 7. nodes_nodeids
		AttributeProto attr_nodes_nodeids;
		attr_nodes_nodeids.set_name("nodes_nodeids");
		attr_nodes_nodeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_nodes_nodeids) {
			attr_nodes_nodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_nodeids;

		// 8. nodes_treeids
		AttributeProto attr_nodes_treeids;
		attr_nodes_treeids.set_name("nodes_treeids");
		attr_nodes_treeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_nodes_treeids) {
			attr_nodes_treeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_treeids;

		// 9. nodes_truenodeids
		AttributeProto attr_nodes_truenodeids;
		attr_nodes_truenodeids.set_name("nodes_truenodeids");
		attr_nodes_truenodeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_nodes_truenodeids) {
			attr_nodes_truenodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_nodes_truenodeids;

		// 10. nodes_values
		AttributeProto attr_nodes_values;
		attr_nodes_values.set_name("nodes_values");
		attr_nodes_values.set_type(AttributeProto::FLOATS);
		for (const auto &val : input_nodes_values) {
			attr_nodes_values.add_floats(val);
		}
		*new_node.add_attribute() = attr_nodes_values;

		// 11. post_transform
		AttributeProto attr_post_transform;
		attr_post_transform.set_name("post_transform");
		attr_post_transform.set_type(AttributeProto::STRING);
		attr_post_transform.set_s(input_post_transform);
		*new_node.add_attribute() = attr_post_transform;

		// 12. target_ids
		AttributeProto attr_target_ids;
		attr_target_ids.set_name("target_ids");
		attr_target_ids.set_type(AttributeProto::INTS);
		for (const auto &id : input_target_ids) {
			attr_target_ids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_ids;

		// 13. target_nodeids
		AttributeProto attr_target_nodeids;
		attr_target_nodeids.set_name("target_nodeids");
		attr_target_nodeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_target_nodeids) {
			attr_target_nodeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_nodeids;

		// 14. target_treeids
		AttributeProto attr_target_treeids;
		attr_target_treeids.set_name("target_treeids");
		attr_target_treeids.set_type(AttributeProto::INTS);
		for (const auto &id : input_target_treeids) {
			attr_target_treeids.add_ints(id);
		}
		*new_node.add_attribute() = attr_target_treeids;

		// 15. target_weights
		AttributeProto attr_target_weights;
		attr_target_weights.set_name("target_weights");
		attr_target_weights.set_type(AttributeProto::FLOATS);
		for (const auto &weight : input_target_weights) {
			attr_target_weights.add_floats(weight);
		}
		*new_node.add_attribute() = attr_target_weights;
		// 将新节点添加到图中
		graph->add_node()->CopyFrom(new_node);

		// 获取时间戳
		// std::time_t now_time = std::time(nullptr);
		boost::uuids::uuid uuid = boost::uuids::random_generator()();
		size_t pos = onnx_model_path.find(".onnx");
		std::string model_name = onnx_model_path.substr(0, pos);
		new_model_path = model_name + "_" + boost::uuids::to_string(uuid) + ".onnx";
		// new_model_path = model_name + "_" + std::to_string(now_time) + ".onnx";

		saveModel(&model, new_model_path);
	}

	static std::set<idx_t> removeUnusedColumns(std::string &model_path, const std::vector<idx_t> &column_indexs,
	                                           onnx::graph_node_list &node_list) {
		std::vector<int64_t> input_nodes_featureids;
		std::vector<int64_t> output_nodes_featureids;
		// std::vector<idx_t> filtered_column_indexs;,
		for (auto node : node_list) {
			for (auto name : node->attributeNames()) {
				if (strcmp(name.toString(), "nodes_featureids") == 0) {
					input_nodes_featureids = node->is(name);
				}
			}
		}
		std::set<idx_t> unique_values(input_nodes_featureids.begin(), input_nodes_featureids.end());
		std::vector<idx_t> used_nodes_featureids(unique_values.begin(), unique_values.end());
		// for (int64_t id : used_nodes_featureids) {
		// 	filtered_column_indexs.push_back(feature_names[id]);
		// }
		for (size_t i = 0; i < input_nodes_featureids.size(); i++) {
			for (size_t j = 0; j < used_nodes_featureids.size(); j++) {
				if (input_nodes_featureids[i] == used_nodes_featureids[j]) {
					output_nodes_featureids.push_back(j);
				}
			}
		}
		reg2reg(model_path, node_list, output_nodes_featureids, unique_values.size());
		return unique_values;
	}

	static std::set<idx_t> OnnxConstructFunction(const std::vector<idx_t> &column_indexs) {
		ModelProto model;
		onnx::optimization::loadModel(&model, onnx_model_path, true);
		std::shared_ptr<Graph> graph(ImportModelProto(model));
		auto node_list = graph->nodes();
		return removeUnusedColumns(onnx_model_path, column_indexs, node_list);
	}
	//
	static bool HasONNXExpressionScan(Expression &expr) {
		if (expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &func_expr = (BoundFunctionExpression &)expr;
			if (func_expr.function.name == "onnx") {
				auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
				onnx_model_path = first_param.value.ToString();
				std::vector<idx_t> column_indexs;
				for (size_t i = 1; i < func_expr.children.size(); i++) {
					auto &param_expr = func_expr.children[i];
					auto &col_expr = (BoundColumnRefExpression &)*param_expr;
					column_indexs.push_back(col_expr.binding.column_index);
					// std::cout << "Parameter " << i << ": " << col_expr.GetName() << std::endl;
					// std::cout << "Parameter " << i << ": " << col_expr.binding.table_index << std::endl;
					// std::cout << "Parameter " << i << ": " << col_expr.binding.column_index << std::endl;
				}
				std::set<idx_t> filtered_column_indexs = OnnxConstructFunction(column_indexs);
				duckdb::Value model_path_value(new_model_path);
				first_param.value = model_path_value;
				for (int i = func_expr.children.size() - 1; i > 0; i--) {
					auto &param_expr = func_expr.children[i];
					auto &col_expr = (BoundColumnRefExpression &)*param_expr;
					if (filtered_column_indexs.find(col_expr.binding.column_index) == filtered_column_indexs.end()) {
						// std::cout << "erase : " << i << std::endl;
						func_expr.children.erase(func_expr.children.begin() + i);
					}
				}
				return true;
			}
		}
		bool found_onnx = false;
		ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
			if (HasONNXExpressionScan(child)) {
				found_onnx = true;
			}
		});
		return found_onnx;
	}

	static bool HasONNXScan(LogicalOperator &op) {
		for (auto &expr : op.expressions) {
			if (HasONNXExpressionScan(*expr)) {
				return true;
			}
		}
		for (auto &child : op.children) {
			if (HasONNXScan(*child)) {
				return true;
			}
		}
		return false;
	}

	// Optimizer Function
	static void UnusedColumnsFunction(ClientContext &context, OptimizerExtensionInfo *info,
	                                  duckdb::unique_ptr<LogicalOperator> &plan) {
		if (!HasONNXScan(*plan)) {
			return;
		}
		// OnnxConstructFunction();
	}
};
std::string UnusedColumnsExtension::onnx_model_path;
std::string UnusedColumnsExtension::new_model_path;

} // namespace onnx::optimization

namespace py = pybind11;
namespace pybind11::literals {
class Skl2OnnxExtension : public OptimizerExtension {
public:
	Skl2OnnxExtension() {
		optimize_function = Skl2OnnxFunction;
	}

	static std::string convertModel(std::string &model_path) {
		try {
			// 初始化 Python 解释器
			py::scoped_interpreter guard {};

			// 导入必要的 Python 模块
			py::module sklearn = py::module::import("sklearn");
			py::module skl2onnx = py::module::import("skl2onnx");
			py::module joblib = py::module::import("joblib");
			py::module data_types = py::module::import("skl2onnx.common.data_types");
			py::module builtins = py::module::import("builtins");

			// 加载模型
			py::object model = joblib.attr("load")(model_path);
			std::cout << "模型加载成功。" << std::endl;

			// 动态获取特征数量
			int n_features = 0;
			if (py::hasattr(model, "n_features_in_")) {
				n_features = model.attr("n_features_in_").cast<int>();
				std::cout << "模型的特征数量: " << n_features << std::endl;
			} else {
				std::cerr << "模型没有 'n_features_in_' 属性，无法确定特征数量。" << std::endl;
				return model_path;
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
			py::object onnx_model = convert_sklearn(model, "initial_types"_a = initial_type_, "options"_a = options);
			std::cout << "模型转换为 ONNX 成功。" << std::endl;

			// 保存 ONNX 模型
			boost::uuids::uuid uuid = boost::uuids::random_generator()();
			size_t pos = model_path.find(".skl");
			std::string model_name = model_path.substr(0, pos);
			std::string new_model_path = model_name + "_" + boost::uuids::to_string(uuid) + ".onnx";

			std::ofstream ofs(new_model_path, std::ios::binary);
			if (!ofs) {
				std::cerr << "无法打开文件以保存 ONNX 模型。" << std::endl;
				return model_path;
			}

			py::object onnx_bytes = onnx_model.attr("SerializeToString")();
			std::string onnx_str = onnx_bytes.cast<std::string>();
			ofs.write(onnx_str.data(), onnx_str.size());
			ofs.close();
			std::cout << "ONNX 模型已保存。" << std::endl;

		} catch (py::error_already_set &e) {
			std::cerr << "Python 错误：" << e.what() << std::endl;
			return model_path;
		} catch (std::exception &e) {
			std::cerr << "C++ 错误：" << e.what() << std::endl;
			return model_path;
		}
		return model_path;
	}

	static bool HasONNXExpressionScan(Expression &expr) {
		if (expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &func_expr = (BoundFunctionExpression &)expr;
			if (func_expr.function.name == "onnx") {
				auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
				std::string onnx_model_path = first_param.value.ToString();
				size_t pos = onnx_model_path.find(".joblib");
				if (pos != std::string::npos) {
					std::string new_model_path = convertModel(onnx_model_path);
					if (new_model_path == onnx_model_path)
					{
						return false;
					}
					
					duckdb::Value model_path_value(new_model_path);
					first_param.value = model_path_value;
					return true;
				}
				return false;
			}
		}
		bool found_onnx = false;
		ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
			if (HasONNXExpressionScan(child)) {
				found_onnx = true;
			}
		});
		return found_onnx;
	}

	static bool HasONNXScan(LogicalOperator &op) {
		for (auto &expr : op.expressions) {
			if (HasONNXExpressionScan(*expr)) {
				return true;
			}
		}
		for (auto &child : op.children) {
			if (HasONNXScan(*child)) {
				return true;
			}
		}
		return false;
	}

	static void Skl2OnnxFunction(ClientContext &context, OptimizerExtensionInfo *info,
	                             duckdb::unique_ptr<LogicalOperator> &plan) {
		if (!HasONNXScan(*plan)) {
			return;
		}
	}
};
} // namespace pybind11::literals

//===--------------------------------------------------------------------===//
// Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void loadable_extension_optimizer_demo_init(duckdb::DatabaseInstance &db) {
	Connection con(db);

	// add a parser extension
	auto &config = DBConfig::GetConfig(db);
	// config.optimizer_extensions.push_back(WaggleExtension());
	// config.AddExtensionOption("waggle_location_host", "host for remote callback", LogicalType::VARCHAR);
	// config.AddExtensionOption("waggle_location_port", "port for remote callback", LogicalType::INTEGER);

	// add a parser extension
	config.optimizer_extensions.push_back(pybind11::literals::Skl2OnnxExtension());
	config.AddExtensionOption("skl2onnx", "convert sklearn to onnx model", LogicalType::INVALID);

	// // add a parser extension
	// config.optimizer_extensions.push_back(onnx::optimization::OnnxExtension());
	// config.AddExtensionOption("pruning", "pruning onnx model", LogicalType::INVALID);

	// // add a parser extension
	// config.optimizer_extensions.push_back(onnx::optimization::UnusedColumnsExtension());
	// config.AddExtensionOption("cuting", "cuting unused columns", LogicalType::INVALID);
}

DUCKDB_EXTENSION_API const char *loadable_extension_optimizer_demo_version() {
	return DuckDB::LibraryVersion();
}
}
