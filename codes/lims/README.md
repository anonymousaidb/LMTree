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
3. Each project will generate an executable file in its output directory. Run with command.
12 parameters are needed when starting the program.
```bash
./run_lims ../result/LIMS/lims-cost-la-0.txt 5 3 2 0 ../dataset/LA_example 0 1 20 50 100 150 200 473, 692, 989, 1409, 1875, 2314, 3096 0.02 0.02 
```

---
