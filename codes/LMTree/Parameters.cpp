

#include "vector_calculation.hpp"
#include <sstream>
#include "Parameters.hpp"


#ifdef MACBOOK
void* operator new(std::size_t size) {
    void* ptr = nullptr;
    std::size_t alignment = AlignedAllocator<float>::alignment;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    throw std::bad_alloc();
}
#else
void* operator new(std::size_t size) {
    if (void* ptr = std::aligned_alloc(AlignedAllocator<float>::alignment, size)) {
        return ptr;
    }
    throw std::bad_alloc();
}
#endif

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

std::mt19937 global_gen(424);

std::string source_path = "../dataset/";
std::string father_path = "../dataset/";
std::string data_buffer_path = "../dataset/data_buffer_path/";

std::vector<std::int64_t> parse_k_values(const std::string& k_values_str) {
    std::vector<std::int64_t> k_values;
    std::stringstream ss(k_values_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        k_values.push_back(std::stoll(item));
    }
    return k_values;
}

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#else
#include <execinfo.h>
#endif
void printStackTrace() {
#ifdef _WIN32
    const HANDLE process = GetCurrentProcess();
    SymInitialize(process, nullptr, true);
    void* stack[100];
    WORD frames = CaptureStackBackTrace(0, 100, stack, nullptr);
    const auto symbol = static_cast<SYMBOL_INFO *>(malloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char)));
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    for (WORD i = 0; i < frames; i++) {
        SymFromAddr(process, reinterpret_cast<DWORD64>(stack[i]), 0, symbol);
        printf("%i: %s - 0x%0llu\n", frames - i - 1, symbol->Name, symbol->Address);
    }
    free(symbol);
#else
    void* callstack[128];
    int frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack, frames);
    for (int i = 0; i < frames; ++i) {
        printf("%s\n", strs[i]);
    }
    free(strs);
#endif
}

#include <cstdlib>

#include <fstream>
#include <cerrno>
#include <string>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <cerrno>
#include <string>
#include <sstream>
#include <iostream>

std::int64_t MemoryInfo() {
#if defined(__unix__) || defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    int64_t memory = -1;
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line.substr(6));
                std::string memStr;
                iss >> memStr;
                try {
                    memory = std::stoll(memStr);
                } catch (const std::invalid_argument& e) {
                    memory = -1;
                }
                break;
            }
        }
        status_file.close();
    }
    return memory >= 0 ? memory * 1024 : memory;

#elif defined(_WIN32) || defined(_WIN64)
    HANDLE handle = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(handle, &pmc, sizeof(pmc))) {
        return static_cast<int64_t>(pmc.WorkingSetSize);
    }
    return -1;

#else
    return -1;
#endif
}

MemoryLog::MemoryLog() : memory_size(0) {
    tick();
}

void MemoryLog::tick() {
    memory_size = MemoryInfo();
}

std::int64_t MemoryLog::get_memory() const {
    return MemoryInfo() - memory_size;
}



