#include "duckdb/plugin/physical/parallel/thread_context.hpp"
#include "duckdb/plugin/physical/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

ThreadContext::ThreadContext(ClientContext &context) : profiler(QueryProfiler::Get(context).IsEnabled()) {
}

} // namespace duckdb
