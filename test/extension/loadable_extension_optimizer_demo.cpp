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
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
// #include <pybind11/embed.h>
// #include <pybind11/stl.h>
#include <optional>
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

// -----------------------------------------------------------------
/**
 * ToDo:
 * node list -> tree node
 *  */
namespace onnx::optimization {

// TODO：判断模型是否是分类模型，是则进行转换
// ** clf2reg

class Clf2regExtension : public OptimizerExtension {
public:
  Clf2regExtension() { optimize_function = Clf2regOptimizeFunction; }
  static std::string onnx_model_path;
  static std::string new_model_path;

  static bool HasONNXFilter(LogicalOperator &op) {
    for (auto &expr : op.expressions) {
      if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
        auto &comparison_expr =
            dynamic_cast<BoundComparisonExpression &>(*expr);
        if (comparison_expr.left->expression_class ==
            ExpressionClass::BOUND_FUNCTION) {
          auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
          if (func_expr.function.name == "onnx" &&
              func_expr.children.size() > 1) {
            auto &first_param =
                (BoundConstantExpression &)*func_expr.children[0];
            if (first_param.value.type().id() == LogicalTypeId::VARCHAR) {
              std::string model_path = first_param.value.ToString();
              if (comparison_expr.right->type ==
                  ExpressionType::VALUE_CONSTANT) {
                auto &constant_expr =
                    (BoundConstantExpression &)*comparison_expr.right;
                onnx_model_path = model_path;

                boost::uuids::uuid uuid = boost::uuids::random_generator()();
                size_t pos = onnx_model_path.find(".onnx");
                std::string model_name = onnx_model_path.substr(0, pos);
                new_model_path = model_name + "_" +
                                 boost::uuids::to_string(uuid) + "_clf2reg" +
                                 ".onnx";
                std::string test_model_path = "./../data/model/house_16H_d10_l281_n561_20240922063836.onnx";
                duckdb::Value model_path_value(test_model_path);
                first_param.value = test_model_path;
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

  static void clf2reg(std::string &model_path, onnx::graph_node_list &node_list) {
    int64_t input_n_targets = 1;
    // different to reg
    std::vector<int64_t> input_class_ids;
    std::vector<int64_t> input_class_nodeids;
    std::vector<int64_t> input_class_treeids;
    std::vector<double> input_class_weights;
    // same
    std::vector<double> input_classlabels_int64s;
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

    std::unordered_map<std::string, int> attr_map = {
        {"n_targets", 1},
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
        {"class_ids", 12},
        {"class_nodeids", 13},
        {"class_treeids", 14},
        {"class_weights", 15},
        {"classlabels_int64s", 16}};

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
            input_class_ids = node->is(name);
            break;
          case 13:
            input_class_nodeids = node->is(name);
            break;
          case 14:
            input_class_treeids = node->is(name);
            break;
          case 15:
            input_class_weights = node->fs(name);
            break;
          case 16:
            input_classlabels_int64s = node->fs(name);
            break;
          default:
            break;
          }
        }
      }
	  }
	
    int stride = input_classlabels_int64s.size() == 2
                      ? 1
                      : input_classlabels_int64s.size();
    int nleaf = input_class_weights.size() / stride;
    // 构建 target_treeids
    vector<int> target_treeids;
    for (size_t i = 0; i < nleaf; ++i) {
      target_treeids.push_back(input_class_treeids[i * stride]);
    }
    // 构建 target_ids
    vector<int> target_ids;
    for (size_t i = 0; i < nleaf; ++i) {
      target_ids.push_back(input_class_ids[i * stride]);
    }
    // 构建 target_nodeids
    vector<int> target_nodeids;
    for (size_t i = 0; i < nleaf; ++i) {
      target_nodeids.push_back(input_class_nodeids[i * stride]);
    }

    // 构建 target_weights
    vector<float> target_weights;
    if (stride == 1) {
      for (auto w : input_class_weights) {
        w > 0.5 ? target_weights.push_back(1.0) : target_weights.push_back(0.0);
      }
    } else {
      for (int i = 0; i < nleaf; ++i) {
        auto start_it = input_class_weights.begin() + (i * stride);
        auto end_it = input_class_weights.begin() + ((i + 1) * stride);

        auto max_it = std::max_element(start_it, end_it);
        int index = std::distance(start_it, max_it);

        target_weights.push_back(static_cast<float>(index));
      }
    }

    // --------------------------------------------------------------

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

    // 设置输出
    ValueInfoProto *new_output = graph->add_output();
    new_output->CopyFrom(initial_graph->output()[0]);
    // 设置新的输出类型和形状
    auto typeProto = new_output->mutable_type();
    auto tensorType = typeProto->mutable_tensor_type();
    tensorType->set_elem_type(TensorProto::FLOAT);

    auto shapeProto = tensorType->mutable_shape();
    auto dim1 = shapeProto->add_dim();
    dim1->set_dim_value(1);
   
    for (const auto &initializer : initial_graph->initializer()) {
      onnx::TensorProto *new_initializer = graph->add_initializer();
      new_initializer->CopyFrom(initializer);
    }

    // 设置新模型的opset_import
    *model.mutable_opset_import() = initial_model.opset_import();

    // 3. 添加 TreeEnsembleRegressor 节点
    NodeProto new_node;
    auto initial_node = initial_graph->node()[0];
    new_node.set_op_type("TreeEnsembleRegressor");
    new_node.set_domain(initial_node.domain());  // 设置 domain 为 ai.onnx.ml
    new_node.set_name("TreeEnsembleRegressor");  // 设置节点名称
    new_node.add_input(initial_node.input()[0]); // 输入
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
    for (const auto &id : input_nodes_featureids) {
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
    attr_nodes_missing_value_tracks_true.set_name(
        "nodes_missing_value_tracks_true");
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

  // constrcut reg model
  static void OnnxConstructFunction() {
    ModelProto model;
    onnx::optimization::loadModel(&model, onnx_model_path, true);
    std::shared_ptr<Graph> graph(ImportModelProto(model));
    auto node_list = graph->nodes();
    clf2reg(onnx_model_path, node_list);
  }

  // Optimizer Function
  static void
  Clf2regOptimizeFunction(ClientContext &context, OptimizerExtensionInfo *info,
                          duckdb::unique_ptr<LogicalOperator> &plan) {
    if (!HasONNXFilter(*plan)) {
      return;
    }
    OnnxConstructFunction();
  }
};

std::string Clf2regExtension::onnx_model_path;
std::string Clf2regExtension::new_model_path;

// -------------------------------------------------------------------------
// ** Pruning
class OnnxExtension : public OptimizerExtension {
public:
  OnnxExtension() {
    optimize_function = OnnxOptimizeFunction;

    // 初始化 comparison_funcs
    comparison_funcs[ExpressionType::COMPARE_LESSTHAN] =
        [](float_t x, float_t y) -> bool { return x < y; };
    comparison_funcs[ExpressionType::COMPARE_GREATERTHAN] =
        [](float_t x, float_t y) -> bool { return x > y; };
    comparison_funcs[ExpressionType::COMPARE_LESSTHANOREQUALTO] =
        [](float_t x, float_t y) -> bool { return x <= y; };
    comparison_funcs[ExpressionType::COMPARE_GREATERTHANOREQUALTO] =
        [](float_t x, float_t y) -> bool { return x >= y; };
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
  static std::unordered_map<ExpressionType,
                            std::function<bool(float_t, float_t)>>
      comparison_funcs;

  struct NodeID {
    int id;
    std::string node;
  };

  // TODO: need to support nested case
  static bool HasONNXFilter(LogicalOperator &op) {
    // std::cout << "Start HasONNXFilter" << std::endl;
    for (auto &expr : op.expressions) {
      if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
        auto &comparison_expr =
            dynamic_cast<BoundComparisonExpression &>(*expr);
        if (comparison_expr.left->expression_class ==
            ExpressionClass::BOUND_FUNCTION) {
          auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
          if (func_expr.function.name == "onnx" &&
              func_expr.children.size() > 1) {
            auto &first_param =
                (BoundConstantExpression &)*func_expr.children[0];
            if (first_param.value.type().id() == LogicalTypeId::VARCHAR) {
              std::string model_path = first_param.value.ToString();
              if (comparison_expr.right->type ==
                  ExpressionType::VALUE_CONSTANT) {
                auto &constant_expr =
                    (BoundConstantExpression &)*comparison_expr.right;
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
                auto new_constant_expr =
                    std::make_unique<duckdb::BoundConstantExpression>(value);
                comparison_expr.right = std::move(new_constant_expr);
                // 获取时间戳
                // std::time_t now_time = std::time(nullptr);
                // size_t pos = onnx_model_path.find(".onnx");
                // std::string model_name = onnx_model_path.substr(0, pos);
                // new_model_path = model_name + "_" + std::to_string(now_time)
                // + ".onnx";

                boost::uuids::uuid uuid = boost::uuids::random_generator()();
                size_t pos = onnx_model_path.find(".onnx");
                std::string model_name = onnx_model_path.substr(0, pos);
                new_model_path = model_name + "_" +
                                 boost::uuids::to_string(uuid) + "_pruning" +
                                 ".onnx";

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

  static int pruning(size_t node_id, size_t depth,
                     vector<std::string> &result_nodes,
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
      auto result = static_cast<int>(comparison_funcs[ComparisonOperator](
          target_weights[target_id], predicate));
      result == 1 ? result_nodes[node_id] = "LEAF_TRUE"
                  : result_nodes[node_id] = "LEAF_FALSE";
      // std::cout << "node_id: " << node_id << ", depth: " << depth << ",
      // is_leaf: " << (is_leaf ? "true" : "false")
      //           << ", result: " << result << std::endl;
      return result;
    } else {
      auto left_node_id = left_nodes[node_id];
      auto left_result =
          pruning(left_node_id, depth + 1, result_nodes, node_list, predicate);
      auto right_node_id = right_nodes[node_id];
      auto right_result =
          pruning(right_node_id, depth + 1, result_nodes, node_list, predicate);

      if (left_result == 0 && right_result == 0) {
        // std::cout << "node_id: " << node_id << ", depth: " << depth
        //           << ", is_leaf: " << (is_leaf ? "true" : "false") << ",
        //           result: " << 0 << std::endl;
        result_nodes[node_id] = "LEAF_FALSE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 0;
      }

      if (left_result == 1 && right_result == 1) {
        // std::cout << "node_id: " << node_id << ", depth: " << depth
        //           << ", is_leaf: " << (is_leaf ? "true" : "false") << ",
        //           result: " << 1 << std::endl;
        result_nodes[node_id] = "LEAF_TRUE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 1;
      }
      // std::cout << "node_id: " << node_id << ", depth: " << depth + 1 << ",
      // leaf_depth: " << depth + 1
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
    vector<std::string> result_nodes{length, ""};
    pruning(0, 0, result_nodes, node_list, predicate);
    removed_nodes = result_nodes;
  }

  static void reg2reg(std::string &model_path,
                      onnx::graph_node_list &node_list) {
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

    std::unordered_map<std::string, int> attr_map = {
        {"n_targets", 1},
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
    // 		} else if (strcmp(name.toString(),
    // "nodes_missing_value_tracks_true")
    // == 0) { 			input_nodes_missing_value_tracks_true =
    // node->is(name); } else if (strcmp(name.toString(), "nodes_modes") == 0) {
    // input_nodes_modes = node->ss(name); 		} else if
    // (strcmp(name.toString(), "nodes_nodeids") == 0)
    // { 			input_nodes_nodeids = node->is(name); } else if
    // (strcmp(name.toString(), "nodes_treeids") == 0) { input_nodes_treeids =
    // node->is(name); 		} else if (strcmp(name.toString(),
    // "nodes_truenodeids")
    // == 0) { 			input_nodes_truenodeids = node->is(name);
    // } else if (strcmp(name.toString(), "nodes_values") == 0) {
    // input_nodes_values = node->fs(name); 		} else if
    // (strcmp(name.toString(), "post_transform") == 0) {
    // input_post_transform = node->s(name); 		} else if
    // (strcmp(name.toString(), "target_ids") == 0) {
    // input_target_ids = node->is(name); 		} else if
    // (strcmp(name.toString(), "target_nodeids") == 0) {
    // input_target_nodeids = node->is(name); 		} else if
    // (strcmp(name.toString(), "target_treeids") == 0) {
    // input_target_treeids = node->is(name); 		} else if
    // (strcmp(name.toString(), "target_weights")
    // == 0) { 			input_target_weights = node->fs(name);
    // 		}
    // 	}
    // }

    // 1. 计算 leaf_count
    int leaf_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE") +
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
        nodes_missing_value_tracks_true.push_back(
            input_nodes_missing_value_tracks_true[i]);
      }
    }

    // 8. 构建 nodes_modes
    vector<std::string> nodes_modes;
    for (const auto &new_id : new_ids) {
      if (new_id.id != -1) {
        std::string mode =
            (new_id.node == "BRANCH_LEQ") ? "BRANCH_LEQ" : "LEAF";
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
        float value =
            (new_ids[i].node == "BRANCH_LEQ") ? input_nodes_values[i] : 0.0f;
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
    new_node.set_domain(initial_node.domain());  // 设置 domain 为 ai.onnx.ml
    new_node.set_name(initial_node.name());      // 设置节点名称
    new_node.add_input(initial_node.input()[0]); // 输入
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
    attr_nodes_missing_value_tracks_true.set_name(
        "nodes_missing_value_tracks_true");
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
  static void OnnxOptimizeFunction(ClientContext &context,
                                   OptimizerExtensionInfo *info,
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
std::unordered_map<ExpressionType, std::function<bool(float_t, float_t)>>
    OnnxExtension::comparison_funcs;

// -------------------------------------------------------------------
// ** 下推
class UnusedColumnsExtension : public OptimizerExtension {
public:
  UnusedColumnsExtension() { optimize_function = UnusedColumnsFunction; }
  static std::string onnx_model_path;
  static std::string new_model_path;
  // static std::vector<int64_t> nodes_featureids;

  static void reg2reg(std::string &model_path, onnx::graph_node_list &node_list,
                      std::vector<int64_t> &output_nodes_featureids,
                      std::size_t dim_value) {
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

    std::unordered_map<std::string, int> attr_map = {
        {"n_targets", 1},
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
    new_node.set_domain(initial_node.domain());  // 设置 domain 为 ai.onnx.ml
    new_node.set_name(initial_node.name());      // 设置节点名称
    new_node.add_input(initial_node.input()[0]); // 输入
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
    attr_nodes_missing_value_tracks_true.set_name(
        "nodes_missing_value_tracks_true");
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
    new_model_path =
        model_name + "_" + boost::uuids::to_string(uuid) + "_remove" + ".onnx";
    // new_model_path = model_name + "_" + std::to_string(now_time) + ".onnx";

    saveModel(&model, new_model_path);
  }

  static std::set<idx_t>
  removeUnusedColumns(std::string &model_path,
                      const std::vector<idx_t> &column_indexs,
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
    std::set<idx_t> unique_values(input_nodes_featureids.begin(),
                                  input_nodes_featureids.end());
    std::vector<idx_t> used_nodes_featureids(unique_values.begin(),
                                             unique_values.end());
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
    reg2reg(model_path, node_list, output_nodes_featureids,
            unique_values.size());
    return unique_values;
  }

  static std::set<idx_t>
  OnnxConstructFunction(const std::vector<idx_t> &column_indexs) {
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
          // std::cout << "Parameter " << i << ": " << col_expr.GetName() <<
          // std::endl; std::cout << "Parameter " << i << ": " <<
          // col_expr.binding.table_index << std::endl; std::cout << "Parameter
          // " << i << ": " << col_expr.binding.column_index << std::endl;
        }
        std::set<idx_t> filtered_column_indexs =
            OnnxConstructFunction(column_indexs);
        duckdb::Value model_path_value(new_model_path);
        first_param.value = model_path_value;
        for (int i = func_expr.children.size() - 1; i > 0; i--) {
          auto &param_expr = func_expr.children[i];
          auto &col_expr = (BoundColumnRefExpression &)*param_expr;
          if (filtered_column_indexs.find(col_expr.binding.column_index) ==
              filtered_column_indexs.end()) {
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
  static void UnusedColumnsFunction(ClientContext &context,
                                    OptimizerExtensionInfo *info,
                                    duckdb::unique_ptr<LogicalOperator> &plan) {
    if (!HasONNXScan(*plan)) {
      return;
    }
    // OnnxConstructFunction();
  }
};
std::string UnusedColumnsExtension::onnx_model_path;
std::string UnusedColumnsExtension::new_model_path;

// -------------------------------------------------------------------
// ** merge
class Node {
public:
  // 成员变量
  int id;         // 节点ID
  int feature_id; // 特征ID
  std::string
      mode; // 节点类型，"LEAF" 表示叶子节点，"BRANCH_LEQ" 表示非叶子节点
  double value;                 // 阈值，叶子节点的值为0
  std::optional<int> target_id; // 叶子节点的 target ID，可能为空
  std::optional<double> target_weight; // 叶子节点的权重，即预测值，可能为空
  int samples;                         // 节点的样本数

  Node *parent; // 父节点指针，可能为空
  Node *left;   // 左子节点指针，可能为空
  Node *right;  // 右子节点指针，可能为空

  // 构造函数
  Node(int _id, int _feature_id, const std::string &_mode, double _value,
       std::optional<int> _target_id, std::optional<double> _target_weight,
       int _samples)
      : id(_id), feature_id(_feature_id), mode(_mode), value(_value),
        target_id(_target_id), target_weight(_target_weight), samples(_samples),
        parent(nullptr), left(nullptr), right(nullptr) {}
};

struct ModelData {
  int64_t input_n_targets;
  std::vector<int64_t> input_nodes_falsenodeids;
  std::vector<int64_t> input_nodes_featureids;
  std::vector<double> input_nodes_hitrates;
  std::vector<int64_t> input_nodes_missing_value_tracks_true;
  std::vector<std::string> input_node_modes;
  std::vector<int64_t> input_nodes_nodeids;
  std::vector<int64_t> input_nodes_treeids;
  std::vector<int64_t> input_nodes_truenodeids;
  std::vector<double> input_nodes_values;
  std::string input_post_transform;
  std::vector<int64_t> input_target_ids;
  std::vector<int64_t> input_target_nodeids;
  std::vector<int64_t> input_target_treeids;
  std::vector<double> input_target_weights;

  // 映射：node_id -> 数组索引
  std::unordered_map<int64_t, size_t> node_id_to_index;
  std::unordered_map<int64_t, size_t> target_nodeid_to_index;

  // 初始化映射
  void initialize_maps() {
    for (size_t i = 0; i < input_nodes_nodeids.size(); ++i) {
      node_id_to_index[input_nodes_nodeids[i]] = i;
    }
    for (size_t i = 0; i < input_target_nodeids.size(); ++i) {
      target_nodeid_to_index[input_target_nodeids[i]] = i;
    }
  }
};

Node *model2tree(int64_t node_id, Node *parent, const ModelData &model_data,
                 const std::unordered_map<int64_t, int> &samples_map = {}) {
  // 获取 node_id 对应的索引
  auto it = model_data.node_id_to_index.find(node_id);
  if (it == model_data.node_id_to_index.end()) {
    std::cerr << "Node id " << node_id << " not found in node_id_to_index map."
              << std::endl;
    return nullptr;
  }
  size_t index = it->second;

  // 提取节点属性
  int64_t feature_id = model_data.input_nodes_featureids[index];
  std::string mode = model_data.input_node_modes[index];
  double value = model_data.input_nodes_values[index];
  int samples = static_cast<int>(model_data.input_nodes_hitrates[index]);

  // 获取 target_id 和 target_weight（如果存在）
  std::optional<int64_t> target_id;
  std::optional<double> target_weight;
  auto it_target = model_data.target_nodeid_to_index.find(node_id);
  if (it_target != model_data.target_nodeid_to_index.end()) {
    size_t target_index = it_target->second;
    target_id = model_data.input_target_ids[target_index];
    target_weight = model_data.input_target_weights[target_index];
  }

  // 仅用于调试，检查 samples 是否匹配
  if (!samples_map.empty()) {
    auto it_samples = samples_map.find(node_id);
    if (it_samples != samples_map.end()) {
      if (samples != it_samples->second) {
        std::cerr << "Samples not match: " << samples
                  << " != " << it_samples->second << std::endl;
        throw std::runtime_error("Samples not match");
      }
    }
  }

  // 创建节点
  Node *node = new Node(node_id, feature_id, mode, value, target_id,
                        target_weight, samples);
  node->parent = parent;

  // 如果不是叶子节点，递归创建子节点
  if (mode != "LEAF") {
    int64_t left_node_id = model_data.input_nodes_truenodeids[index];
    Node *left_node = model2tree(left_node_id, node, model_data, samples_map);
    node->left = left_node;

    int64_t right_node_id = model_data.input_nodes_falsenodeids[index];
    Node *right_node = model2tree(right_node_id, node, model_data, samples_map);
    node->right = right_node;
  }

  return node;
}

void delete_tree(Node *node) {
  if (node == nullptr) {
    return;
  }

  // 递归删除左子树和右子树
  delete_tree(node->left);
  delete_tree(node->right);

  // 删除当前节点
  delete node;
}

class TreeEnsembleRegressor {
public:
  int n_targets;
  std::vector<int> nodes_falsenodeids;
  std::vector<int> nodes_featureids;
  std::vector<double> nodes_hitrates;
  std::vector<int> nodes_missing_value_tracks_true;
  std::vector<std::string> nodes_modes;
  std::vector<int> nodes_nodeids;
  std::vector<int> nodes_treeids;
  std::vector<int> nodes_truenodeids;
  std::vector<double> nodes_values;
  std::string post_transform;
  std::vector<int> target_ids;
  std::vector<int> target_nodeids;
  std::vector<int> target_treeids;
  std::vector<double> target_weights;

  // 构造函数
  TreeEnsembleRegressor();
  onnx::ModelProto to_model(const onnx::ModelProto &input_model);
  static TreeEnsembleRegressor from_tree(Node *root);

private:
  static void from_tree_internal(TreeEnsembleRegressor &regressor, Node *node);
};

onnx::ModelProto
TreeEnsembleRegressor::to_model(const onnx::ModelProto &input_model) {
  // 创建新模型
  onnx::ModelProto model;
  model.set_ir_version(input_model.ir_version());

  // 设置 opset_import
  *model.mutable_opset_import() = input_model.opset_import();

  // 获取或创建图形
  onnx::GraphProto *graph = model.mutable_graph();

  // 设置图名称：如果输入模型的图名称为空，设置为默认名称
  std::string graph_name = input_model.graph().name();
  if (graph_name.empty()) {
    graph_name = "TreeEnsembleGraph"; // 默认名称
  }
  graph->set_name(graph_name);

  // 复制输入信息
  for (const auto &input : input_model.graph().input()) {
    onnx::ValueInfoProto *new_input = graph->add_input();
    new_input->CopyFrom(input);
  }

  // 复制输出信息
  for (const auto &output : input_model.graph().output()) {
    onnx::ValueInfoProto *new_output = graph->add_output();
    new_output->CopyFrom(output);
  }

  // 复制初始值（如有需要）
  for (const auto &initializer : input_model.graph().initializer()) {
    onnx::TensorProto *new_initializer = graph->add_initializer();
    new_initializer->CopyFrom(initializer);
  }

  // 创建新的节点
  onnx::NodeProto *new_node = graph->add_node();
  new_node->set_op_type("TreeEnsembleRegressor");
  new_node->set_domain("ai.onnx.ml");          // 设置 domain 为 ai.onnx.ml
  new_node->set_name("TreeEnsembleRegressor"); // 设置节点名称

  // 设置节点的输入和输出（假设使用第一个输入和输出）
  if (input_model.graph().input_size() > 0) {
    new_node->add_input(input_model.graph().input(0).name());
  } else {
    // 处理没有输入的情况，可能需要抛出异常或错误提示
    throw std::runtime_error("Input model has no inputs.");
  }

  if (input_model.graph().output_size() > 0) {
    new_node->add_output(input_model.graph().output(0).name());
  } else {
    // 处理没有输出的情况，可能需要抛出异常或错误提示
    throw std::runtime_error("Input model has no outputs.");
  }

  // 设置节点属性

  // 1. n_targets
  onnx::AttributeProto *attr_n_targets = new_node->add_attribute();
  attr_n_targets->set_name("n_targets");
  attr_n_targets->set_type(onnx::AttributeProto::INT);
  attr_n_targets->set_i(n_targets);

  // 2. nodes_falsenodeids
  onnx::AttributeProto *attr_nodes_falsenodeids = new_node->add_attribute();
  attr_nodes_falsenodeids->set_name("nodes_falsenodeids");
  attr_nodes_falsenodeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : nodes_falsenodeids) {
    attr_nodes_falsenodeids->add_ints(id);
  }

  // 3. nodes_featureids
  onnx::AttributeProto *attr_nodes_featureids = new_node->add_attribute();
  attr_nodes_featureids->set_name("nodes_featureids");
  attr_nodes_featureids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : nodes_featureids) {
    attr_nodes_featureids->add_ints(id);
  }

  // 4. nodes_hitrates
  onnx::AttributeProto *attr_nodes_hitrates = new_node->add_attribute();
  attr_nodes_hitrates->set_name("nodes_hitrates");
  attr_nodes_hitrates->set_type(onnx::AttributeProto::FLOATS);
  for (const auto &rate : nodes_hitrates) {
    attr_nodes_hitrates->add_floats(rate);
  }

  // 5. nodes_missing_value_tracks_true
  onnx::AttributeProto *attr_nodes_missing_value_tracks_true =
      new_node->add_attribute();
  attr_nodes_missing_value_tracks_true->set_name(
      "nodes_missing_value_tracks_true");
  attr_nodes_missing_value_tracks_true->set_type(onnx::AttributeProto::INTS);
  for (const auto &val : nodes_missing_value_tracks_true) {
    attr_nodes_missing_value_tracks_true->add_ints(val);
  }

  // 6. nodes_modes
  onnx::AttributeProto *attr_nodes_modes = new_node->add_attribute();
  attr_nodes_modes->set_name("nodes_modes");
  attr_nodes_modes->set_type(onnx::AttributeProto::STRINGS);
  for (const auto &mode : nodes_modes) {
    attr_nodes_modes->add_strings(mode);
  }

  // 7. nodes_nodeids
  onnx::AttributeProto *attr_nodes_nodeids = new_node->add_attribute();
  attr_nodes_nodeids->set_name("nodes_nodeids");
  attr_nodes_nodeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : nodes_nodeids) {
    attr_nodes_nodeids->add_ints(id);
  }

  // 8. nodes_treeids
  onnx::AttributeProto *attr_nodes_treeids = new_node->add_attribute();
  attr_nodes_treeids->set_name("nodes_treeids");
  attr_nodes_treeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : nodes_treeids) {
    attr_nodes_treeids->add_ints(id);
  }

  // 9. nodes_truenodeids
  onnx::AttributeProto *attr_nodes_truenodeids = new_node->add_attribute();
  attr_nodes_truenodeids->set_name("nodes_truenodeids");
  attr_nodes_truenodeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : nodes_truenodeids) {
    attr_nodes_truenodeids->add_ints(id);
  }

  // 10. nodes_values
  onnx::AttributeProto *attr_nodes_values = new_node->add_attribute();
  attr_nodes_values->set_name("nodes_values");
  attr_nodes_values->set_type(onnx::AttributeProto::FLOATS);
  for (const auto &val : nodes_values) {
    attr_nodes_values->add_floats(val);
  }

  // 11. post_transform
  onnx::AttributeProto *attr_post_transform = new_node->add_attribute();
  attr_post_transform->set_name("post_transform");
  attr_post_transform->set_type(onnx::AttributeProto::STRING);
  attr_post_transform->set_s(post_transform);

  // 12. target_ids
  onnx::AttributeProto *attr_target_ids = new_node->add_attribute();
  attr_target_ids->set_name("target_ids");
  attr_target_ids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : target_ids) {
    attr_target_ids->add_ints(id);
  }

  // 13. target_nodeids
  onnx::AttributeProto *attr_target_nodeids = new_node->add_attribute();
  attr_target_nodeids->set_name("target_nodeids");
  attr_target_nodeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : target_nodeids) {
    attr_target_nodeids->add_ints(id);
  }

  // 14. target_treeids
  onnx::AttributeProto *attr_target_treeids = new_node->add_attribute();
  attr_target_treeids->set_name("target_treeids");
  attr_target_treeids->set_type(onnx::AttributeProto::INTS);
  for (const auto &id : target_treeids) {
    attr_target_treeids->add_ints(id);
  }

  // 15. target_weights
  onnx::AttributeProto *attr_target_weights = new_node->add_attribute();
  attr_target_weights->set_name("target_weights");
  attr_target_weights->set_type(onnx::AttributeProto::FLOATS);
  for (const auto &weight : target_weights) {
    attr_target_weights->add_floats(weight);
  }

  // 返回构建好的模型
  return model;
}

TreeEnsembleRegressor::TreeEnsembleRegressor()
    : n_targets(1), post_transform("NONE") {
  // 各个向量默认初始化为空
}

TreeEnsembleRegressor TreeEnsembleRegressor::from_tree(Node *root) {
  TreeEnsembleRegressor regressor;
  from_tree_internal(regressor, root);

  // 创建 id 映射：old_id -> 新索引
  std::unordered_map<int, int> id_map;
  for (size_t i = 0; i < regressor.nodes_nodeids.size(); ++i) {
    int old_id = regressor.nodes_nodeids[i];
    id_map[old_id] = static_cast<int>(i);
  }

  // 判断节点是否为叶子节点
  std::vector<bool> is_leaf;
  for (const auto &mode : regressor.nodes_modes) {
    is_leaf.push_back(mode == "LEAF");
  }

  // 更新 nodes_falsenodeids 和 nodes_truenodeids
  for (size_t i = 0; i < regressor.nodes_falsenodeids.size(); ++i) {
    if (is_leaf[i]) {
      regressor.nodes_falsenodeids[i] = 0;
      regressor.nodes_truenodeids[i] = 0;
    } else {
      regressor.nodes_falsenodeids[i] = id_map[regressor.nodes_falsenodeids[i]];
      regressor.nodes_truenodeids[i] = id_map[regressor.nodes_truenodeids[i]];
    }
    regressor.nodes_nodeids[i] = id_map[regressor.nodes_nodeids[i]];
  }

  // 更新 target_nodeids
  for (size_t i = 0; i < regressor.target_nodeids.size(); ++i) {
    regressor.target_nodeids[i] = id_map[regressor.target_nodeids[i]];
  }

  return regressor;
}

void TreeEnsembleRegressor::from_tree_internal(TreeEnsembleRegressor &regressor,
                                               Node *node) {
  bool is_leaf = node->mode == "LEAF";

  // 处理 falsenodeids 和 truenodeids，对于叶子节点设为 0
  int falsenodeid = (!is_leaf && node->right) ? node->right->id : 0;
  int truenodeid = (!is_leaf && node->left) ? node->left->id : 0;

  // 添加节点信息到回归器
  regressor.nodes_falsenodeids.push_back(falsenodeid);
  regressor.nodes_featureids.push_back(node->feature_id);
  regressor.nodes_hitrates.push_back(static_cast<double>(node->samples));
  regressor.nodes_missing_value_tracks_true.push_back(0);
  regressor.nodes_modes.push_back(node->mode);
  regressor.nodes_nodeids.push_back(node->id);
  regressor.nodes_treeids.push_back(0);
  regressor.nodes_truenodeids.push_back(truenodeid);
  regressor.nodes_values.push_back(node->value);

  if (is_leaf) {
    regressor.target_ids.push_back(0);
    regressor.target_nodeids.push_back(node->id);
    regressor.target_treeids.push_back(0);
    regressor.target_weights.push_back(node->target_weight.value_or(0.0));
  }

  // 递归处理子节点
  if (!is_leaf) {
    if (node->left) {
      from_tree_internal(regressor, node->left);
    }
    if (node->right) {
      from_tree_internal(regressor, node->right);
    }
  }
}

class MergeExtension : public OptimizerExtension {
public:
  MergeExtension() { optimize_function = MergeFunction; }
  static std::string onnx_model_path;
  static std::string new_model_path;

  using NodeDepthPair = std::pair<Node *, int>;
  using LeafNodesList = std::vector<NodeDepthPair>;
  using MergeNodesList = std::vector<std::vector<NodeDepthPair>>;

  static void get_leaf_nodes(Node *node, int depth, LeafNodesList &leaf_nodes) {
    if (node == nullptr) {
      return;
    }

    // 递增深度
    int current_depth = depth + 1;

    if (node->mode == "LEAF") {
      leaf_nodes.emplace_back(node, current_depth);
      //   std::cout << "get it " << std::endl;
    } else {
      // 递归遍历左子树和右子树
      get_leaf_nodes(node->left, current_depth, leaf_nodes);
      get_leaf_nodes(node->right, current_depth, leaf_nodes);
    }
  }

  static MergeNodesList get_can_merge_nodes(const LeafNodesList &leaf_nodes) {
    MergeNodesList merge_nodes_list;

    for (size_t i = 0; i < leaf_nodes.size() - 1; ++i) {
      const NodeDepthPair &l1 = leaf_nodes[i];
      const NodeDepthPair &l2 = leaf_nodes[i + 1];

      // 如果两个叶子节点的 target_weight 不同，则跳过
      if (l1.first->target_weight != l2.first->target_weight) {
        continue;
      }

      // 获取两个叶子节点的父节点及其深度
      Node *p1_node = l1.first->parent;
      int p1_depth = l1.second - 1;
      Node *p2_node = l2.first->parent;
      int p2_depth = l2.second - 1;

      int feature_id = p1_node->feature_id;
      int d1 = 1;
      int d2 = 1;

      // 寻找共同的父节点
      while (p1_node != p2_node) {
        if (feature_id != p1_node->feature_id ||
            feature_id != p2_node->feature_id) {
          // 无法合并
          //   std::cout << "fail to merge" << std::endl;
          break;
        }

        if (p1_depth > p2_depth) {
          p1_node = p1_node->parent;
          p1_depth -= 1;
          d1 += 1;
        } else if (p2_depth > p1_depth) {
          p2_node = p2_node->parent;
          p2_depth -= 1;
          d2 += 1;
        } else {
          p1_node = p1_node->parent;
          p2_node = p2_node->parent;
          p1_depth -= 1;
          p2_depth -= 1;
          d1 += 1;
          d2 += 1;
        }

        if (p1_node == nullptr || p2_node == nullptr) {
          // 到达根节点，未找到共同父节点
          break;
          //   std::cout << "fail to find same father node" << std::endl;
        }
      }

      // 检查是否找到了可以合并的共同父节点
      if (p1_node == p2_node && feature_id == p1_node->feature_id) {
        // std::cout << "can merge: " << i << " " << (i + 1)
        //           << ", n_nodes: " << (2 * leaf_nodes.size() - 1) <<
        //           std::endl;

        // 根据距离确定顺序
        if (d1 <= d2) {
          merge_nodes_list.push_back(
              {{p1_node, p1_depth}, {l1.first, d1}, {l2.first, d2}});
        } else {
          merge_nodes_list.push_back(
              {{p1_node, p1_depth}, {l2.first, d2}, {l1.first, d1}});
        }
      }
    }

    return merge_nodes_list;
  }

  static int merge(std::vector<NodeDepthPair> &nodes) {
    int reduced_cost = 0;

    Node *common_parent = nodes[0].first;

    // 较长路径的节点
    Node *node = nodes[2].first;
    Node *parent = node->parent;
    // reduced_cost += node->samples + parent->samples;

    // std::cout << "common_parent.value " << common_parent->value
    //           << " shorter_node_parent.value " <<
    //           nodes[1].first->parent->value
    //           << " longer_node_parent.value " <<
    //           nodes[2].first->parent->value
    //           << std::endl;

    // 修改共同父节点的阈值
    common_parent->value = parent->value;
    // std::cout << "common_parent.value_ " << common_parent->value <<
    // std::endl;

    // 移除较长路径节点的父节点的对应子节点
    // std::cout<<"1715 correct"<<std::endl;
    Node *another = nullptr;
    if (parent->left == node) {
      another = parent->right;
      parent->right = nullptr;
    } else {
      another = parent->left;
      parent->left = nullptr;
    }
    // std::cout<<"1724 correct"<<std::endl;
    // 修改 parent 的父节点指向
    if (parent->parent->left == parent) {
      parent->parent->left = another;
    } else {
      parent->parent->right = another;
    }
    // std::cout<<"1731 correct"<<std::endl;
    another->parent = parent->parent;
    parent->parent = nullptr;
    // std::cout<<"1733 correct"<<std::endl;
    // 修改较长路径节点的样本数
    parent = another->parent;
    // int merge_samples = node->samples;
    // while (parent != common_parent) {
    //   parent->samples -= merge_samples;
    //   parent = parent->parent;

    //   reduced_cost += merge_samples;
    // }
    // std::cout<<"1743 correct"<<std::endl;
    // // 修改较短路径节点的样本数
    // node = nodes[1].first;
    // while (node != common_parent) {
    //   node->samples += merge_samples;
    //   node = node->parent;

    //   reduced_cost -= merge_samples;
    // }

    return reduced_cost;
  }

  static void ONNXModelConstruct() {
    onnx::ModelProto model;
    onnx::optimization::loadModel(&model, onnx_model_path, true);
    // std::cout << "1741 ONNXModelConstruct: " << onnx_model_path << std::endl;
    std::shared_ptr<onnx::Graph> graph(onnx::ImportModelProto(model));
    auto node_list = graph->nodes();

    ModelData model_data;

    for (auto node : node_list) {
      for (auto name : node->attributeNames()) {
        std::string attr_name = name.toString();
        if (attr_name == "n_targets") {
          model_data.input_n_targets = node->i(name);
        } else if (attr_name == "nodes_falsenodeids") {
          model_data.input_nodes_falsenodeids = node->is(name);
        } else if (attr_name == "nodes_featureids") {
          model_data.input_nodes_featureids = node->is(name);
        } else if (attr_name == "nodes_hitrates") {
          model_data.input_nodes_hitrates = node->fs(name);
        } else if (attr_name == "nodes_missing_value_tracks_true") {
          model_data.input_nodes_missing_value_tracks_true = node->is(name);
        } else if (attr_name == "nodes_modes") {
          model_data.input_node_modes = node->ss(name);
        } else if (attr_name == "nodes_nodeids") {
          model_data.input_nodes_nodeids = node->is(name);
        } else if (attr_name == "nodes_treeids") {
          model_data.input_nodes_treeids = node->is(name);
        } else if (attr_name == "nodes_truenodeids") {
          model_data.input_nodes_truenodeids = node->is(name);
        } else if (attr_name == "nodes_values") {
          model_data.input_nodes_values = node->fs(name);
        } else if (attr_name == "post_transform") {
          model_data.input_post_transform = node->s(name);
        } else if (attr_name == "target_ids") {
          model_data.input_target_ids = node->is(name);
        } else if (attr_name == "target_nodeids") {
          model_data.input_target_nodeids = node->is(name);
        } else if (attr_name == "target_treeids") {
          model_data.input_target_treeids = node->is(name);
        } else if (attr_name == "target_weights") {
          model_data.input_target_weights = node->fs(name);
        }
      }
    }
    model_data.initialize_maps();

    Node *root = model2tree(0, nullptr, model_data);
    LeafNodesList leaf_nodes;
    get_leaf_nodes(root, 0, leaf_nodes);

    MergeNodesList merge_nodes_list = get_can_merge_nodes(leaf_nodes);
    std::sort(merge_nodes_list.begin(), merge_nodes_list.end(),
              [](const std::vector<NodeDepthPair> &a,
                 const std::vector<NodeDepthPair> &b) {
                return a[0].second > b[0].second;
              });

    int total_reduced_cost = 0;
    for (auto &merge_nodes : merge_nodes_list) {
      total_reduced_cost += merge(merge_nodes);
    }
    // std::cout << "total_reduced_cost: " << total_reduced_cost << std::endl;
    auto regressor = TreeEnsembleRegressor::from_tree(root);
    auto output_model = regressor.to_model(model);

    saveModel(&output_model, new_model_path);
    // 释放树的内存
    delete_tree(root);
  }

  static bool HasONNXExpressionScan(Expression &expr) {
    if (expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
      auto &func_expr = (BoundFunctionExpression &)expr;
      if (func_expr.function.name == "onnx") {
        auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
        onnx_model_path = first_param.value.ToString();
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        size_t pos = onnx_model_path.find(".onnx");
        std::string model_name = onnx_model_path.substr(0, pos);
        new_model_path = model_name + "_" + boost::uuids::to_string(uuid) +
                         "_merge" + ".onnx";
        ONNXModelConstruct();
        duckdb::Value model_path_value(new_model_path);
        first_param.value = model_path_value;
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
  static void MergeFunction(ClientContext &context,
                            OptimizerExtensionInfo *info,
                            duckdb::unique_ptr<LogicalOperator> &plan) {
    if (!HasONNXScan(*plan)) {
      return;
    }
  }
}; // namespace onnx::optimization
std::string MergeExtension::onnx_model_path;
std::string MergeExtension::new_model_path;

} // namespace onnx::optimization

// ** sklearn模型转onnx模型
class Skl2OnnxExtension : public OptimizerExtension {
public:
  Skl2OnnxExtension() { optimize_function = Skl2OnnxFunction; }

  static std::string convertModel(std::string &model_path) {
    boost::uuids::uuid uuid = boost::uuids::random_generator()();

    size_t pos = model_path.find_last_of("/");
    std::string model_name = model_path.substr(pos + 1);
    std::string prefix = model_path.substr(0, pos);

    pos = model_name.find(".joblib");
    std::string new_model_name = model_name.substr(0, pos) + "_" +
                                 boost::uuids::to_string(uuid) + ".onnx";
    std::string command =
        std::string("./../data/exe/exe.linux-x86_64-3.9/convert ") + prefix +
        "/" + model_name + " " + prefix + "/" + new_model_name;
    int ret = system(command.c_str());
    if (ret != 0) {
      std::cerr << "convert failed!" << std::endl;
      return model_path;
    }

    return prefix + "/" + new_model_name;
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
          if (new_model_path == onnx_model_path) {
            std::cerr << "convert failed!" << std::endl;
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

  static void Skl2OnnxFunction(ClientContext &context,
                               OptimizerExtensionInfo *info,
                               duckdb::unique_ptr<LogicalOperator> &plan) {
    if (!HasONNXScan(*plan)) {
      return;
    }
  }
};

//===--------------------------------------------------------------------===//
// Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void
loadable_extension_optimizer_demo_init(duckdb::DatabaseInstance &db) {
  Connection con(db);

  // add a parser extension
  auto &config = DBConfig::GetConfig(db);
  // add a parser extension
  //   config.optimizer_extensions.push_back(Skl2OnnxExtension());
  //   config.AddExtensionOption("skl2onnx", "convert sklearn to onnx model",
  //                             LogicalType::INVALID);

  // add a parser extension
  config.optimizer_extensions.push_back(onnx::optimization::Clf2regExtension());
  config.AddExtensionOption("clf2reg", "convert clf model to reg model",
                            LogicalType::INVALID);

  // // add a parser extension
  // config.optimizer_extensions.push_back(onnx::optimization::OnnxExtension());
  // config.AddExtensionOption("pruning", "pruning onnx model",
  //                           LogicalType::INVALID);

  // // add a parser extension
  // config.optimizer_extensions.push_back(
  //     onnx::optimization::UnusedColumnsExtension());
  // config.AddExtensionOption("cuting", "cuting unused columns",
  //                           LogicalType::INVALID);

  // // add a parser extension
  // config.optimizer_extensions.push_back(onnx::optimization::MergeExtension());
  // config.AddExtensionOption("merge", "merge nodes", LogicalType::INVALID);
}

DUCKDB_EXTENSION_API const char *loadable_extension_optimizer_demo_version() {
  return DuckDB::LibraryVersion();
}
}
