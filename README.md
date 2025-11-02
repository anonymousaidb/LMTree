## Introduction

This repo holds the source code and scripts for "LM-Tree: A Hybrid Learned Index for Similarity Search in Metric Spaces”.

## Codes
This file contains the source code files for M-Tree, PM-tree, MVP-Tree, EGNAT, LIMS, Flood, Qd-Tree, and our LM-Tree. All C++ projects can be built with **cmake 3.28 & gcc-13**, 

1. Open the project with clion, vscode or open in terminal.
2. Compile the project using **CMake** in **Release** mode.
3. For detailed steps and examples, please refer to the README file corresponding to each algorithm in codes folder.





## Datasets

Each dataset can be obtained from the following links, while the synthetic dataset can be generalized following the decriptions in our paper.

| __Dataset__ | __link__                                             | __Metric__ |
|-------------|------------------------------------------------------|------------|
| LA          | http://www.dbs.informatik.uni-muenchen.de/$\sim$seidl | L2         |
| Color       | http://cophir.isti.cnr.it/                         | L1         |
| Words       | http://icon.shef.ac.uk/Moby/                 | Edit       |


Two sample datasets are included in the folder 'dataset'.
