//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/plugin/physical/execution/operator/join/physical_positional_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/plugin/physical/execution/physical_operator.hpp"
#include "duckdb/plugin/physical/common/types/column/column_data_collection.hpp"

namespace duckdb {

//! PhysicalPositionalJoin represents a cross product between two tables
class PhysicalPositionalJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::POSITIONAL_JOIN;

public:
	PhysicalPositionalJoin(vector<LogicalType> types, unique_ptr<PhysicalOperator> left,
	                       unique_ptr<PhysicalOperator> right, idx_t estimated_cardinality);

public:
	// Operator Interface
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

public:
	// Source interface
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

public:
	// Sink Interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;

	bool IsSink() const override {
		return true;
	}

public:
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;
};
} // namespace duckdb
