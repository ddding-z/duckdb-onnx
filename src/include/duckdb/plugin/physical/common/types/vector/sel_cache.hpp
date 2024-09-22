//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/plugin/physical/common/types/vector/sel_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/plugin/physical/common/types/vector/vector_buffer.hpp"
#include "duckdb/common/unordered_map.hpp"

namespace duckdb {

//! Selection vector cache used for caching vector slices
struct SelCache {
	unordered_map<sel_t *, buffer_ptr<VectorBuffer>> cache;
};

} // namespace duckdb
