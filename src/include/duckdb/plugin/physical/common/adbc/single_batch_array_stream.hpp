#pragma once

#include "duckdb/plugin/physical/common/arrow/arrow.hpp"
#include "duckdb/plugin/physical/common/adbc/adbc.h"

namespace duckdb_adbc {

struct SingleBatchArrayStream {
	struct ArrowSchema schema;
	struct ArrowArray batch;
};

AdbcStatusCode BatchToArrayStream(struct ArrowArray *values, struct ArrowSchema *schema,
                                  struct ArrowArrayStream *stream, struct AdbcError *error);

} // namespace duckdb_adbc
