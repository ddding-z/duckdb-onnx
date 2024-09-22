#include "duckdb/plugin/physical/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_reset.hpp"
#include "duckdb/plugin/physical/execution/operator/helper/physical_reset.hpp"

namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalReset &op) {
	return make_uniq<PhysicalReset>(op.name, op.scope, op.estimated_cardinality);
}

} // namespace duckdb
