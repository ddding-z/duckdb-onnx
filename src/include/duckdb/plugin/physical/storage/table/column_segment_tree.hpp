//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/plugin/physical/storage/table/column_segment_tree.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/plugin/physical/storage/table/segment_tree.hpp"
#include "duckdb/plugin/physical/storage/table/column_segment.hpp"

namespace duckdb {

class ColumnSegmentTree : public SegmentTree<ColumnSegment> {};

} // namespace duckdb
