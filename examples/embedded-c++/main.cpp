#include "duckdb.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

// preset
std::vector<std::string> predicate = {"> 0.420043", "> 0.830135", "> 1.240227", "> 1.65032", "> 2.060412",
                                      ">2.470504", "> 2.880597", "> 3.290689", "> 3.700781",  "> 4.110874"};
std::vector<std::string> threads = {"2"};
// int times = 1;

// SQL
// Set Threads
std::string setThreads = "set threads = ?;";

// Load Extensions
std::string loadOnnxRunime = "LOAD './../../../build/test/extension/loadable_extension_demo.duckdb_extension';";
std::string loadRule = "LOAD './../../../build/test/extension/loadable_extension_optimizer_demo.duckdb_extension';";

// Load csv
std::string loadCSV =
    "CREATE TABLE table1 AS  SELECT * FROM read_csv('./../../../data/csv/NASA_d10_l264_n527_20241106064301.csv', delim=',', "
    "header=True, columns={ "
    "'passenger_count': 'FLOAT', 'tolls_amount': 'FLOAT', 'total_amount': 'FLOAT','lpep_pickup_datetime_day': 'FLOAT', "
    "'lpep_pickup_datetime_hour': 'FLOAT', 'lpep_pickup_datetime_minute': 'FLOAT', 'lpep_dropoff_datetime_day': 'FLOAT', 'lpep_dropoff_datetime_hour': 'FLOAT', 'lpep_dropoff_datetime_minute': 'FLOAT'});";

// Query 1 直接run原模型
std::string query1 = "SELECT * FROM table1 where "
                                 "onnx('./../../../data/model/NASA_d10_l264_n527_20241106064301.onnx', "
                                 "passenger_count,tolls_amount,total_amount,lpep_pickup_datetime_day,lpep_pickup_datetime_hour,lpep_pickup_datetime_minute,"
                                 "lpep_dropoff_datetime_day,lpep_dropoff_datetime_hour,lpep_dropoff_datetime_minute) ?;";

// Query 2 直接run优化后模型
std::string queryPrunedModel = "SELECT * FROM table1 where "
                               "onnx('./output_model.onnx', "
                               "P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, "
                               "P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA,H40p4) > 0;";

std::string queryOptimizedModel = "SELECT * FROM table1 where "
                                  "onnx('./../../../build/output_model2.onnx', "
                                  "P1,P5p1,P11p4,P14p9,P15p1,P27p4,H2p2,H10p1,H13p1,H18pA "
                                  ") > 0;";

std::string replaceFirst(std::string str, const std::string &from, const std::string &to) {
	size_t start_pos = str.find(from);
	if (start_pos != std::string::npos) {
		str.replace(start_pos, from.length(), to);
	}
	return str;
}

void testNoOp() {
	// presets
	std::vector<double> records;
	// 启动 duckdb
	duckdb::DBConfig config;
	config.options.allow_unsigned_extensions = true;
	duckdb::DuckDB db(nullptr, &config);
	duckdb::Connection con(db);
	con.Query("PRAGMA disable_verification;");
	con.Query("PRAGMA enable_optimizer;");  // 启用优化器
	con.Query("PRAGMA force_parallelism;"); // 强制并行执行

	std::ofstream file("./../../../data/output/testNoOp_nyc.csv");

	// load extension
	con.Query(loadOnnxRunime);
	// load csv
	con.Query(loadCSV);
	for (size_t i = 0; i < threads.size(); i++) {
		for (size_t j = 0; j < predicate.size(); j++) {
			std::string set = replaceFirst(setThreads, "?", threads[i]);
			std::string querysql = replaceFirst(query1, "?", predicate[j]);
			// set threads
			con.Query(set);
			// run
			int count = 10;
			while (count--) {
				auto start = std::chrono::high_resolution_clock::now();
				con.Query(querysql);
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> duration = end - start;
				records.push_back(duration.count());
			}
			// average
			double sum = std::accumulate(records.begin(), records.end(), 0.0);
			double average = sum / records.size();
			records.push_back(average);
			std::cout << average << std::endl;
			file << predicate[j] << ",";
			for (size_t i = 0; i < records.size(); ++i) {
				file << records[i];
				if (i != records.size() - 1) {
					file << ",";
				}
			}
			file << "\n";
			records.clear();
		}
	}
	file.close();
}

void testWithOps() {
	// presets
	std::vector<double> records;
	// 启动 duckdb
	duckdb::DBConfig config;
	config.options.allow_unsigned_extensions = true;
	// config.options.unsafe_optimizations = true;
	duckdb::DuckDB db(nullptr, &config);
	duckdb::Connection con(db);
	con.Query("PRAGMA disable_verification;");
	con.Query("PRAGMA enable_optimizer;");  // 启用优化器
	con.Query("PRAGMA force_parallelism;"); // 强制并行执行

	std::ofstream file("./../../../data/output/testWithOpsMerge_nyc.csv");

	// load extension
	con.Query(loadOnnxRunime);
	con.Query(loadRule);
	// load csv
	con.Query(loadCSV);
	for (size_t i = 0; i < threads.size(); i++) {
		for (size_t j = 0; j < predicate.size(); j++) {
			std::string set = replaceFirst(setThreads, "?", threads[i]);
			std::string querysql = replaceFirst(query1, "?", predicate[j]);
			// set threads
			con.Query(set);
			int count = 10;
			while (count--) {
				auto start = std::chrono::high_resolution_clock::now();
				con.Query(querysql);
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> duration = end - start;
				// con.Query(querysql);
				// result->Print();
				records.push_back(duration.count());
			}
			// average
			double sum = std::accumulate(records.begin(), records.end(), 0.0);
			double average = sum / records.size();
			records.push_back(average);
			std::cout << average << std::endl;
			file << predicate[j] << ",";
			for (size_t i = 0; i < records.size(); ++i) {
				file << records[i];
				if (i != records.size() - 1) {
					file << ",";
				}
			}
			file << "\n";
			records.clear();
		}
	}
	file.close();
}

int main() {
	std::cout<<"start no op"<<std::endl;
	// testNoOp();
	std::cout<<"start with ops"<<std::endl;
	testWithOps();
}
