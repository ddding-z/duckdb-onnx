#include "duckdb/plugin/physical/execution/operator/helper/physical_execute.hpp"

#include "duckdb/plugin/physical/parallel/meta_pipeline.hpp"

namespace duckdb {

PhysicalExecute::PhysicalExecute(PhysicalOperator &plan)
    : PhysicalOperator(PhysicalOperatorType::EXECUTE, plan.types, -1), plan(plan) {
}

vector<const_reference<PhysicalOperator>> PhysicalExecute::GetChildren() const {
	return {plan};
}

void PhysicalExecute::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// EXECUTE statement: build pipeline on child
	meta_pipeline.Build(plan);
}

} // namespace duckdb
