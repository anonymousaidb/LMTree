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
   Four parameters are needed when starting the program.
   The Four parameters are as follows:
   dataset filename, cost filename, query filename, update filename
```bash
./run_mvptree ../dataset/LA_example.txt ../result/CO/MTree-la-cost-1.txt ../dataset/LA_query.txt ../dataset/LA_example.txt
```

---
