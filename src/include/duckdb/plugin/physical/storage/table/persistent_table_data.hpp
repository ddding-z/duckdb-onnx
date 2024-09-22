//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/plugin/physical/storage/table/persistent_table_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/plugin/physical/storage/data_pointer.hpp"
#include "duckdb/plugin/physical/storage/table/table_statistics.hpp"
#include "duckdb/plugin/physical/storage/metadata/metadata_manager.hpp"

namespace duckdb {
class BaseStatistics;

class PersistentTableData {
public:
	explicit PersistentTableData(idx_t column_count);
	~PersistentTableData();

	TableStatistics table_stats;
	idx_t total_rows;
	idx_t row_group_count;
	MetaBlockPointer block_pointer;
};

} // namespace duckdb
