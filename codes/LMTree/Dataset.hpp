

#ifndef DATSET_H
#define DATSET_H
#include <string>
#include <iostream>
#include <vector>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <fstream>
#include <array>
#include <cstring>
#include <functional>
#include "Parameters.hpp"

namespace process_files {

    using FileHandler = std::function<bool(const std::filesystem::path &)>;
     void process_files_in_directory(const std::filesystem::path &directory, const FileHandler &handler);
}


 std::vector<std::vector<float> > read_fvecs(const std::string &filename);

 std::vector<std::vector<uint8_t> > read_bvecs(const std::string &filename);

 std::vector<int>  readTinyMetadata(const std::string &filename = father_path + "tiny_images/tiny_metadata.bin", std::int64_t num = -1);




#endif
