
#include "Dataset.hpp"
#include "Parameters.hpp"
namespace process_files {

    using FileHandler = std::function<bool(const std::filesystem::path &)>;

    void process_files_in_directory(const std::filesystem::path &directory, const FileHandler &handler) {
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
            throw std::invalid_argument("Invalid directory path:"+directory.string());
            return;
        }

        for (const auto &entry: std::filesystem::recursive_directory_iterator(directory)) {
            if (std::filesystem::is_regular_file(entry.path())) {
                if(auto result = handler(entry.path()); !result) {
                    return;
                }
            }
        }
    }
}

 std::vector<std::vector<float> > read_fvecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::vector<std::vector<float> > data;
    std::int64_t count = 0;
    while (file) {
        int dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!file) break;
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!file) break;

        data.push_back(vec);
        ++count;
    }
    return data;
}


 std::vector<std::vector<uint8_t> > read_bvecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<uint8_t> > data;

    while (file) {
        int dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!file) break;

        std::vector<uint8_t> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim);
        if (!file) break;

        data.push_back(vec);
    }
    return data;
}



 std::vector<int>  readTinyMetadata(const std::string &filename,const std::int64_t num) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    constexpr int RECORD_SIZE = sizeof(int);
    file.seekg(0, std::ios::end);
    const std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::int64_t numRecords = fileSize / RECORD_SIZE;
    std::vector<int> labels(numRecords);
    file.read(reinterpret_cast<char *>(labels.data()), fileSize);
    for (std::int64_t i = 0; i < 10; ++i) {
        std::cout << "Label " << i << ": " << labels[i] << std::endl;
    }
    return labels;
}
