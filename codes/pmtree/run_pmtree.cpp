#include <iostream>
#include<vector>
#include<algorithm>
#include<random>
#include"pmtree/PM_Tree.h"
#include"pmtree/MyFunc.h"
#include<set>
#include"pmtree/DataMetric.h"
#include <chrono>
#include <map>

using namespace std;

int dim = 2;
int func = 2;
int objNum;

double distance_Between_Piovt_Vector(vector<vector<double>> &pivot_vector_) {
    double all_dist = 0;
    for (int i = 0; i < pivot_vector_.size(); ++i) {
        for (int j = i + 1; j < pivot_vector_.size(); ++j) {
            all_dist += MyFunc::Cal_Euclidean_distance(pivot_vector_[i], pivot_vector_[j]);
        }
    }
    return all_dist;
}

vector<vector<double>> Random_pivot(DataMetric &raw_data_, int pivot_num_, int N_) {
    default_random_engine rng(0);
    uniform_int_distribution<int> u_random_int(0, raw_data_.size() - 1);
    vector<vector<vector<double>>> N_pivot_label(N_);
    for (int i = 0; i < N_; ++i) {
        set<int> repeat_flag_set;
        vector<vector<double>> pivot_label(pivot_num_);
        for (int j = 0; j < pivot_num_; ++j) {
            int label = u_random_int(rng);
            if (repeat_flag_set.count(label) == 0) {
                pivot_label[j] = raw_data_[label];
                repeat_flag_set.insert(label);
            } else
                --j;
        }
        N_pivot_label.push_back(pivot_label);
    }


    pair<double, int> res_label(INT16_MIN, -1);
    for (int i = 0; i < N_pivot_label.size(); ++i) {
        double dist = distance_Between_Piovt_Vector(N_pivot_label[i]);
        if (dist > res_label.first)
            res_label.second = i;
    }
    return N_pivot_label[res_label.second];
}


map<string, vector<double>> get_dataset_radii_map() {
    return {
            {"LA", {473,   692,   989,   1409,  1875,  2314,  3096}},
            {"SY", {2321,  2733,  3229,  3843,  4614,  5613,  7090}},
            {"CO", {0.328, 0.485, 0.558, 0.622, 0.768, 0.812, 0.962}}
    };
}

vector<double> get_radii_for_dataset(const string &dataset_name) {
    static const auto radii_map = get_dataset_radii_map();


    string upper_name = dataset_name;
    transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    for (const auto &[key, radii]: radii_map) {
        if (upper_name.find(key) != string::npos) {
            return radii;
        }
    }


    return {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0};
}

int main(int argc, char **argv) {
    int M = 1024;
    int Pivot_Num = 32;
    int N = 500;
    const string outFileName = argv[1];
    const string data_file = argv[2];
    const string query_file = argv[3];
    const string update_file = argv[4];

    ofstream outFile(outFileName, ios::app);
    if (!outFile.is_open()) {
        cerr << "Cannot open output file: " << outFileName << endl;
        return 1;
    }


    auto range_buff = get_radii_for_dataset(data_file);

    DataMetric rawData;
    DataMetric queryData;
    rawData.load_Raw_Data(data_file);
    queryData.load_Query_Data(query_file, rawData[0].size());



    PM_Tree my_pm_tree(M);
    my_pm_tree.Cal_Distance_Num = 0;
    vector<vector<double>> pivot_vec = Random_pivot(rawData, Pivot_Num, N);
    my_pm_tree.Set_Pivot(Pivot_Num, Pivot_Num, Pivot_Num, pivot_vec);
    for (int i = 0; i < rawData.size(); ++i) {
        my_pm_tree.Insert(rawData[i], i);
    }
    M_Node_St *tmp_root = my_pm_tree.Get_Root();
    tmp_root->level = 0;
    my_pm_tree.Update_Level(tmp_root, 1);




    for (double range_id: range_buff) {
        vector<vector<pair<double, int>>> mtree_res(queryData.size());
        my_pm_tree.Cal_Distance_Num = 0;
        for (int i = 0; i < queryData.size(); ++i) {
            my_pm_tree.Range_Search(queryData[i], range_id, mtree_res[i]);
        }
    }


    vector<int> k_values = {20, 50, 100, 150, 200};
    for (int k: k_values) {
        vector<pair<double, int>> results;
        my_pm_tree.Cal_Distance_Num = 0;
        for (int i = 0; i < queryData.size(); ++i) {
            my_pm_tree.k_NN_Search(queryData[i], k, results); // 查找5个最近邻
        }
    }
    outFile.close();
    return 0;
}
