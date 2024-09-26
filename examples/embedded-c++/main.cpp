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
// std::string input_model_path = "./../../data/model/house_16H_d5_l25_n49_20240905072107.onnx";
// std::string output_model_path = "./../../data/model/output_model.onnx";
// std::vector<std::string> predicate = {"> 10.9", "> 10.7", "> 10.6", "> 11.3", "> 9.0",
//                                       "> 9.8",  "> 9.7",  "> 10.2", "> 11.6", "> 12.0"};
// std::vector<std::string> predicate = {"> 9.962608"};
std::vector<std::string> predicate = {"> 9.962608", "> 10.309411", "> 10.656214", "> 11.003016", "> 11.349819",
                                      ">11.696622", "> 12.043425", "> 12.390227", "> 12.73703",  "> 13.083833"};
// std::vector<std::string> threads = {"1", "2", "5", "10", "20"};
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
    "CREATE TABLE table1 AS  SELECT * FROM read_csv('./../../../data/csv/house_16H_1G.csv', delim=',', "
    "header=True, columns={ "
    "'P1': 'FLOAT', 'P5p1': 'FLOAT', 'P6p2': 'FLOAT','P11p4': 'FLOAT', "
    "'P14p9': 'FLOAT', 'P15p1': 'FLOAT', 'P15p3': 'FLOAT', 'P16p2': 'FLOAT', 'P18p2': 'FLOAT', "
    "'P27p4': 'FLOAT', 'H2p2': 'FLOAT', "
    "'H8p2': 'FLOAT', 'H10p1': 'FLOAT', 'H13p1': 'FLOAT', 'H18pA': 'FLOAT', 'H40p4': 'FLOAT' });";

// Query 1 直接run原模型
std::string query1 = "SELECT * FROM table1 where "
                                 "onnx('./../../../data/model/house_16H_d10_l281_n561_20240922063836.onnx', "
                                 "P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, "
                                 "P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA,H40p4) ?;";

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

	std::ofstream file("./../../../data/output/testNoOp.csv");

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
				// auto result = con.Query(querysql);
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

	std::ofstream file("./../../../data/output/testWithOps3.csv");

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



// void testPruneRules() {
// 	// presets
// 	std::vector<double> records;
// 	// 启动 duckdb
// 	duckdb::DBConfig config;
// 	config.options.allow_unsigned_extensions = true;
// 	// config.options.unsafe_optimizations = true;
// 	duckdb::DuckDB db(nullptr, &config);
// 	duckdb::Connection con(db);
// 	con.Query("PRAGMA disable_verification;");
// 	con.Query("PRAGMA enable_optimizer;");  // 启用优化器
// 	con.Query("PRAGMA force_parallelism;"); // 强制并行执行

// 	std::ofstream file("testPrunedModelModelOutput2.csv");

// 	// load extension
// 	con.Query(loadOnnxRunime);
// 	con.Query(loadRule);
// 	// load csv
// 	con.Query(loadCSV);
// 	// auto result = con.Query(loadCSV);
// 	// result->Print();
// 	for (size_t i = 0; i < threads.size(); i++) {
// 		for (size_t j = 0; j < predicate.size(); j++) {
// 			std::string set = replaceFirst(setThreads, "?", threads[i]);
// 			std::string querysql = replaceFirst(queryOriginalModel, "?", predicate[j]);
// 			// set threads
// 			con.Query(set);
// 			// run
// 			// std::cout << "start:" << set << " " << predicate[j] << std::endl;
// 			int count = 20;
// 			while (count--) {
// 				std::cout << "start:" << count << std::endl;
// 				auto start = std::chrono::high_resolution_clock::now();
// 				// auto result = con.Query(querysql);
// 				con.Query(querysql);
// 				auto end = std::chrono::high_resolution_clock::now();
// 				std::chrono::duration<double, std::milli> duration = end - start;
// 				// result->Print();
// 				records.push_back(duration.count());
// 			}
// 			// average
// 			double sum = std::accumulate(records.begin(), records.end(), 0.0);
// 			double average = sum / records.size();
// 			records.push_back(average);
// 			std::cout << average << std::endl;
// 			file << predicate[j] << ",";
// 			// file << set << " " << predicate[j] << "\n";
// 			for (size_t i = 0; i < records.size(); ++i) {
// 				file << records[i];
// 				if (i != records.size() - 1) {
// 					file << ",";
// 				}
// 			}
// 			file << "\n";
// 			records.clear();
// 		}
// 	}
// 	file.close();
// }

// void testOptimizedModel() {
	// presets
	// std::vector<double> records1;
// 	std::vector<double> records2;
// 	std::vector<double> records3;
// 	// 启动 duckdb
// 	duckdb::DBConfig config;
// 	config.options.allow_unsigned_extensions = true;
// 	// config.options.unsafe_optimizations = true;
// 	duckdb::DuckDB db(nullptr, &config);
// 	duckdb::Connection con(db);
// 	con.Query("PRAGMA disable_verification;");
// 	con.Query("PRAGMA enable_optimizer;");  // 启用优化器
// 	con.Query("PRAGMA force_parallelism;"); // 强制并行执行

// 	// std::ofstream file1("testOriginalModelOutput1.csv");
// 	std::ofstream file2("testOptimizedModelOutput.csv");

// 	// load extension
// 	con.Query(loadOnnxRunime);
// 	con.Query(loadRule);
// 	// load csv
// 	con.Query(loadCSV);
// 	// auto result = con.Query(loadCSV);
// 	// result->Print();
// 	for (size_t i = 0; i < threads.size(); i++) {
// 		for (size_t j = 0; j < predicate.size(); j++) {
// 			std::string set = replaceFirst(setThreads, "?", threads[i]);
// 			std::string querysql = replaceFirst(queryWithRules, "?", predicate[j]);
// 			// set threads
// 			con.Query(set);
// 			// run
// 			// std::cout << "start:" << set << " " << predicate[j] << std::endl;
// 			int count = 10;
// 			// auto start = std::chrono::high_resolution_clock::now();
// 			con.Query(querysql);
// 			// auto end = std::chrono::high_resolution_clock::now();
// 			// std::chrono::duration<double, std::milli> duration1 = end - start;
// 			while (count--) {
// 				std::cout << "start:" << count << std::endl;
// 				// result->Print();
// 				// records1.push_back(duration1.count());
// 				auto start2 = std::chrono::high_resolution_clock::now();
// 				con.Query(queryPrunedModel);
// 				auto end2 = std::chrono::high_resolution_clock::now();
// 				std::chrono::duration<double, std::milli> duration2 = end2 - start2;
// 				records2.push_back(duration2.count());
// 				// result->Print();
// 			}
// 			// average
// 			// double sum1 = std::accumulate(records1.begin(), records1.end(), 0.0);
// 			// double average1 = sum1 / records1.size();
// 			// records1.push_back(average1);
// 			// std::cout <<"1:" << average1 << std::endl;

// 			// file1 << set << " "<< predicate[j]<< "\n";
// 			// for (size_t i = 0; i < records1.size(); ++i) {
// 			// 	file1 << records1[i];
// 			// 	if (i != records1.size() - 1) {
// 			// 		file1 << ",";
// 			// 	}
// 			// }
// 			// file1 << "\n";
// 			// records1.clear();

// 			double sum2 = std::accumulate(records2.begin(), records2.end(), 0.0);
// 			double average2 = sum2 / records2.size();
// 			records2.push_back(average2);
// 			// std::cout << "2:" << average2 << std::endl;
// 			file2 << predicate[j] << ",";
// 			// file2 << set << " " << predicate[j] << "\n";
// 			for (size_t i = 0; i < records2.size(); ++i) {
// 				file2 << records2[i];
// 				if (i != records2.size() - 1) {
// 					file2 << ",";
// 				}
// 			}
// 			// file2 << average2;
// 			file2 << "\n";
// 			records2.clear();
// 		}
// 	}
// 	// file1.close();
// 	file2.close();
// }

int main() {
	std::cout<<"start no op"<<std::endl;
	testNoOp();
	std::cout<<"start with ops"<<std::endl;
	testWithOps();
}
