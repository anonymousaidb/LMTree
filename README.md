# LM-Tree

This repository contains all experimental code and example data used in our project.

---

## Download

**Please download ALL files** including:

- C++ project code files
- Example Data files (`.txt` files)

---

## Build & Run

All C++ projects can be built with **cmake 3.28 & gcc-13**:

1. Open the project with clion, vscode or open in terminal.
2. Compile the project using **CMake** in **Release** mode, follow these steps in your terminal or command prompt:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```
3. Each project will generate an executable file in its output directory. Run with command
```bash
./run_LMTree
```
---

## 📑 Data Format

**Data file (example for 2‑dimensional data):**

The three values in the first line represent the data dimensionality (d), the number of data records (n), and a distance type identifier (customizable). The following n lines each contain one data record.
```csv
2,1000,2
6032.63,5585.62
6033.83,5585.80
6026.04,5580.84
...
```
