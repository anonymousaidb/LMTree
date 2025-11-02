

#ifndef DEFINES_H
#define DEFINES_H

#include <string>
#include <stdexcept>
#include <cstdint>
#include <vector>

void* operator new(std::size_t size);

void operator delete(void* ptr) noexcept;

#define PRINT_LOCATION() \
std::cout << "File: " << __FILE__ << "\n" \
<< "Line: " << __LINE__ << "\n" \
<< "Function: " << __FUNCTION__ << "\n"

#define PRINT(i) std::cout << (i) << " "
#define PRINTLN(i) std::cout << (i) << std::endl
#define ERR(i) std::cerr << (i) <<  " "
#define ERRLN(i) std::cerr << (i) << std::endl


std::int64_t MemoryInfo();

class MemoryLog {
    std::int64_t memory_size;
public:
    MemoryLog();
    void tick();
    [[nodiscard]] std::int64_t get_memory() const;
};

void printStackTrace();


std::vector<std::int64_t> parse_k_values(const std::string& k_values_str);

#include <random>

extern std::mt19937 global_gen;
extern std::string source_path;
extern std::string father_path;
extern std::string data_buffer_path;

bool IsFileExist(const char *path);




#endif
