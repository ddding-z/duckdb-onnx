//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/plugin/physical/parallel/pipeline_complete_event.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/plugin/physical/parallel/event.hpp"

namespace duckdb {
class Executor;

class PipelineCompleteEvent : public Event {
public:
	PipelineCompleteEvent(Executor &executor, bool complete_pipeline_p);

	bool complete_pipeline;

public:
	void Schedule() override;
	void FinalizeFinish() override;
};

} // namespace duckdb
