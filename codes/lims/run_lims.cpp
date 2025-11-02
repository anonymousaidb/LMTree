#include "lims/fileIO/FileIO.h"
#include "lims/caculateRef/ChooseRef.h"
#include "lims/caculateRef/CaculateIValue.h"
#include "lims/entities/Reference.h"
#include "lims/entities/Point.h"
#include "lims/queryProcessing/RangeQuery.h"
#include "lims/queryProcessing/KNN.h"
#include "lims/common/CalCirclePos.h"
#include "lims/common/config.h"
#include <random>
#include "lims/dis.h"
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <queue>
#include <algorithm>

#define rand() ((rand()%10000)*10000+rand()%10000)

using namespace std;

long disOpts = 0L;

int dim = 2;
int func = 2;
int objNum;

double CaculateEuclideanDis(Point &point_a, Point &point_b) {
    disOpts++;
    double total = 0.0;
    unsigned size = point_a.coordinate.size();
    switch (ptype) {
        case 2:
            for (unsigned i = 0; i < size; i++) {
                total += std::pow(point_a.coordinate[i] - point_b.coordinate[i], 2);
            }
            return std::sqrt(total);
        case 1:
            for (unsigned i = 0; i < size; i++) {
                total += std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
            }
            return total;
        case 0:
            for (unsigned i = 0; i < size; i++) {
                double diff = std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
                if (diff > total) {
                    total = diff;
                }
            }
            return total;
        default:
            std::cerr << "not support" << std::endl;
            return -1.0;
    }
}

double CaculateEuclideanDis2(const vector<double> &point_a, const Point &point_b) {
    disOpts++;
    double total = 0.0;
    unsigned size = point_a.size();
    switch (ptype) {
        case 2:
            for (unsigned i = 0; i < size; i++) {
                total += std::pow(point_a[i] - point_b.coordinate[i], 2);
            }
            return std::sqrt(total);
        case 1:
            for (unsigned i = 0; i < size; i++) {
                total += std::abs(point_a[i] - point_b.coordinate[i]);
            }
            return total;
        case 0:
            for (unsigned i = 0; i < size; i++) {
                double diff = std::abs(point_a[i] - point_b.coordinate[i]);
                if (diff > total) {
                    total = diff;
                }
            }
            return total;
        default:
            std::cerr << "not support" << std::endl;
            return -1.0;
    }
}

vector<Point> LoadPointForQuery(string filename) {
    ifstream fin;
    vector<Point> queryPoints;
    fin.open(filename.c_str());
    if (!fin) {
        cout << filename << " file could not be opened\n";
        exit(0);
    }
    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> coordinate;
        while (getline(ss, value, ','))
            coordinate.push_back(atof(value.c_str()));

        Point point = Point(coordinate);
        queryPoints.push_back(point);
    }
    fin.close();
    return queryPoints;
}

static bool AscendingSort(const InsertPt &point_a, const InsertPt &point_b) {
    return point_a.i_value < point_b.i_value;
}

int BinarySearch(vector<InsertPt> &disList, double target, int low, int high) {
    int middle = 0;
    while (low <= high) {
        middle = (high - low) / 2 + low;
        if (target < disList[middle].i_value) {
            high = middle - 1;
        } else if (target > disList[middle].i_value) {
            low = middle + 1;
        } else {
            if ((middle > 0 && disList[middle - 1].i_value < target) || (middle == 0))
            {
                break;
            } else
                high = middle - 1;
        }
    }
    return middle;
}


double dist[NUUM];
int fgg[NUUM];
int stack1[NUUM], top = 0;

double sm(double a, double b) {
    if (a > b) return b;
    return a;
}

void work(int dim) {
    top = 1;
    stack1[0] = rand() % num;
    fgg[stack1[0]] = 1;
    for (int j = 0; j < num; j++) dist[j] = dis(j, stack1[0], dim);
    for (int i = 1; i < pnum; i++) {
        double max = 0;
        int k = -1;
        for (int j = 0; j < num; j++)
            if (fgg[j] == 0) {
                if (dist[j] > max) {
                    max = dist[j];
                    k = j;
                }
            }
        stack1[top++] = k;
        fgg[k] = 1;
        for (int j = 0; j < num; j++)dist[j] = sm(dist[j], dis(j, k, dim));
    }
}

void clu(int dim, string path) {
    vector<vector<int>> cluster;
    cluster.resize(pnum);
    for (int index = 0; index < num; ++index) {
        double dist = dis(index, stack1[0], dim);
        int pivot_index = 0;
        for (int i = 1; i < pnum; ++i) {
            double dis_o = dis(index, stack1[i], dim);
            if (dis_o < dist) {
                dist = dis_o;
                pivot_index = i;
            }
        }
        cluster[pivot_index].push_back(index);
    }
    ofstream fout;

    fout.setf(ios::fixed);
    if (ptype == 2) {
        fout.precision(3);
    }

    if (ptype == 0) {
        fout.precision(2);
    }

    if (ptype == 1) {
        fout.precision(3);
    }


    for (int i = 0; i < pnum; ++i) {
        int len = cluster[i].size();
        if (len < 3) {
            continue;
        }
        string filename = path + "/clu_" + to_string(i) + ".txt";
        fout.open(filename);
        for (int j = 0; j < len; ++j) {
            fout << cluster[i][j];
            for (int d = 0; d < dim; ++d)
                fout << "," << loc[cluster[i][j]][d];
            fout << endl;
        }

        fout.close();
    }
}


void op(string filename, int dim) {
    ofstream out1;
    out1.open(filename);
    out1.setf(ios::fixed);
    if (ptype == 2) {
        out1.precision(3);
    }

    if (ptype == 0) {
        out1.precision(2);
    }

    if (ptype == 1) {
        out1.precision(3);
    }

    for (int i = 0; i < top; i++) {
        out1 << loc[stack1[i]][0];
        for (int d = 1; d < dim; ++d)
            out1 << "," << loc[stack1[i]][d];
        out1 << endl;
    }
    out1.close();
}


double
CalErrAvg(vector<Clu_Point> &data, vector<vector<Point>> &ref_data, vector<double> &dist_max, int k, int num_ref,
          int num_alldata) {
    double avg_err = 0;
    for (int i = 0; i < k; ++i) {
        int len = data[i].clu_point.size();
        double clu_err = 0;
        for (int r = 0; r < num_ref; ++r) {
            double err = 0;
            vector<double> arr_dis;
            arr_dis.reserve(len);
            for (int l = 0; l < len; ++l)
                arr_dis.push_back(CaculateEuclideanDis(ref_data[i][r], data[i].clu_point[l]));
            sort(arr_dis.begin(), arr_dis.end());
            if (!r)
                dist_max[i] = arr_dis[len - 1];

            double sum_xy = 0, sum_x = 0, sum_y = 0, sum_pow2x = 0;
            for (int y = 0; y < len; ++y) {
                sum_x += arr_dis[y];
                sum_y += y;
                sum_pow2x += arr_dis[y] * arr_dis[y];
                sum_xy += arr_dis[y] * y;
            }
            double avg_x = sum_x / len;
            double avg_y = sum_y / len;
            double avg_xy = sum_xy / len;
            double avg_pow2x = sum_pow2x / len;
            double a = (avg_xy - avg_x * avg_y) / (avg_pow2x - avg_x * avg_x);
            double b = avg_y - a * avg_x;
            for (int c = 0; c < len; ++c) {
                int cal_y = (int) (a * arr_dis[c] + b);
                err += abs(cal_y - c);
            }
            clu_err = clu_err + err;
        }
        avg_err += clu_err;
    }

    return avg_err / (num_ref * num_alldata);
}


double CalSihCoe(vector<Clu_Point> &data, int k) {
    double s = 0;
    double num_data = 0;
    for (int i = 0; i < k; ++i) {
        int len = data[i].clu_point.size();
        num_data += len;

        for (int j = 0; j < len; ++j) {
            double a = 0, b = 0x3f3f3f;

            for (int k = 0; k < len; ++k)
                a += CaculateEuclideanDis(data[i].clu_point[j], data[i].clu_point[k]);

            a = a / (len - 1);

            for (int l = 0; l < k; ++l) {
                if (l == i)
                    continue;
                double bi = 0;
                int oth_len = data[l].clu_point.size();
                for (int o = 0; o < oth_len; ++o)
                    bi += CaculateEuclideanDis(data[i].clu_point[j], data[l].clu_point[o]);
                bi /= oth_len;
                b = b < bi ? b : bi;
            }
            s = s + (b - a) / (a < b ? b : a);
        }
    }
    return s / num_data;
}

double CalOverlapRate(vector<vector<Point>> &ref_data, vector<double> &dist_max, int k) {
    double overlap_rate = 0;
    for (int i = 0; i < k; ++i) {
        double rate = 0;
        for (int j = 0; j < k; ++j) {
            if (i == j)
                continue;
            double dist_cl1_cl2 = CaculateEuclideanDis(ref_data[i][0], ref_data[j][0]);

            if (dist_cl1_cl2 > dist_max[i] + dist_max[j])
                rate += 0;
            else
                rate = rate + (min(dist_cl1_cl2 + dist_max[j], dist_max[i]) - max(dist_cl1_cl2 - dist_max[j], 0.0)) /
                              dist_max[i];

        }
        overlap_rate = overlap_rate + rate / (k - 1);
    }
    return overlap_rate / k;
}

double CalR2(vector<Clu_Point> &data, vector<vector<Point>> &ref_data, int k, int num_ref) {
    double r2 = 0;
    for (int i = 0; i < k; ++i) {
        int len = data[i].clu_point.size();
        double clu_r2 = 0;
        for (int r = 0; r < num_ref; ++r) {
            vector<double> arr_dis;
            arr_dis.reserve(len);
            for (int l = 0; l < len; ++l)
                arr_dis.push_back(CaculateEuclideanDis(ref_data[i][r], data[i].clu_point[l]));
            sort(arr_dis.begin(), arr_dis.end());


            double sum_xy = 0, sum_x = 0, sum_y = 0, sum_pow2x = 0;
            for (int y = 0; y < len; ++y) {
                sum_x += arr_dis[y];
                sum_y += y;
                sum_pow2x += arr_dis[y] * arr_dis[y];
                sum_xy += arr_dis[y] * y;
            }
            double avg_x = sum_x / len;
            double avg_y = sum_y / len;
            double avg_xy = sum_xy / len;
            double avg_pow2x = sum_pow2x / len;
            double a = (avg_xy - avg_x * avg_y) / (avg_pow2x - avg_x * avg_x);
            double b = avg_y - a * avg_x;


            double sum = 0, sum_var = 0;

            for (int c = 0; c < len; ++c) {
                int cal_y = (int) (a * arr_dis[c] + b);
                sum += pow(c - cal_y, 2);
                sum_var += pow(c - avg_y, 2);
            }
            sum /= len;
            sum_var /= len;
            clu_r2 = clu_r2 + 1 - sum / sum_var;
        }
        clu_r2 /= num_ref;

        r2 += clu_r2;
    }
    r2 /= k;

    return r2;
}

std::vector<unsigned> parseArrayK(const std::string &str) {
    std::vector<unsigned> result;
    std::istringstream iss(str);
    unsigned num;
    while (iss >> num) {
        result.push_back(num);
    }
    return result;
}

std::vector<double> parseArrayR(const std::string &str) {
    std::vector<double> result;
    std::istringstream iss(str);
    double num;
    while (iss >> num) {
        result.push_back(num);
    }
    return result;
}


int main(int argc, const char *argv[]) {

    string filename_main = argv[1];
    std::ofstream outFile(filename_main, std::ios::app);
    if (outFile.is_open()) {
        srand((int) time(0));

        if (argc < 6) {
            return 0;
        }

        unsigned num_ref = stoi(argv[2]);
        unsigned num_clu = stoi(argv[3]);
        unsigned dim = stoi(argv[4]);
        unsigned flag_eva = stoi(argv[5]);

        if (dim == 2)ptype = 2;

        if (dim == 20)ptype = 0;

        if (dim == 32)ptype = 1;

        if (flag_eva) {
            unsigned min_clu = stoi(argv[6]);
            unsigned max_clu = stoi(argv[7]);
            unsigned step = stoi(argv[8]);

            string data_path = argv[9];
            string clu_path = argv[10];

            for (unsigned i = min_clu; i < max_clu + 1; i += step) {

                pnum = i;
                readm(data_path, dim, 0);

                int num_alldata = num;
                outFile << num_alldata << endl;
                for (int k = 0; k <= num; k++)fgg[k] = 0;
                work(dim);

                string clu_data_path = clu_path + to_string(i);
                struct stat info{};
                if (stat(clu_data_path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
                    std::string deleteCommand = "rm -rf " + clu_data_path;
                    if (std::system(deleteCommand.c_str()) != 0) {
                        outFile << "Failed to delete existing directory: " << clu_data_path << std::endl;
                        return 1;
                    }
                }

                int isCreate = mkdir(clu_data_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
                if (!isCreate)
                    outFile << "create path success : " << clu_data_path << endl;
                else {
                    outFile << "create path failed ! error code :  " << isCreate << clu_data_path << endl;
                    return 0;
                }

                op(clu_data_path + "/ref.txt", dim);
                clu(dim, clu_data_path);
                pnum = num_ref - 1;
                for (unsigned j = 0; j < i; ++j) {
                    readm(clu_data_path + "/clu_" + to_string(j) + ".txt", dim, 1);
                    for (int k = 0; k <= num; k++)fgg[k] = 0;
                    work(dim);
                    op(clu_data_path + "/ref_" + to_string(j) + ".txt", dim);
                }
                vector<Clu_Point> data;
                data.reserve(i);
                vector<vector<Point>> ref_data;
                ref_data.resize(i);

                string ref_path = clu_data_path + "/ref.txt";
                ifstream fin;
                fin.open(ref_path);
                if (!fin) {
                    outFile << ref_path << " file could not be opened\n";
                    exit(0);
                }
                string line;
                int index = 0;
                while (getline(fin, line)) {
                    stringstream ss(line);
                    string value;
                    vector<double> coor;
                    while (getline(ss, value, ',')) {
                        coor.push_back(atof(value.c_str()));
                    }
                    Point main_ref = Point(coor);
                    ref_data[index++].push_back(main_ref);
                }
                fin.close();

                for (unsigned j = 0; j < i; ++j) {
                    InputReader inputReader(clu_data_path + "/clu_" + to_string(j) + ".txt");
                    data.push_back(inputReader.getCluster());

                    string oth_ref_path = clu_data_path + "/ref_" + to_string(j) + ".txt";
                    fin.open(oth_ref_path);
                    if (!fin) {
                        outFile << oth_ref_path << " file could not be opened\n";
                        exit(0);
                    }
                    while (getline(fin, line)) {
                        stringstream ss(line);
                        string value;
                        vector<double> coor;
                        while (getline(ss, value, ',')) {
                            coor.push_back(atof(value.c_str()));
                        }
                        Point oth_ref = Point(coor);
                        ref_data[j].push_back(oth_ref);
                    }
                    fin.close();
                }

                vector<double> dist_max;
                dist_max.resize(i);

                double avg_err = CalErrAvg(data, ref_data, dist_max, i, num_ref, num_alldata);
                double rate = CalOverlapRate(ref_data, dist_max, i);
                double r2 = CalR2(data, ref_data, i, num_ref);

                outFile << "K is : " << i << " average err of RP model : "
                        << avg_err << " overlap rate is : " << rate << " r2 is :" << r2 << endl;

            }
            return 0;
        }

        vector<Clu_Point> all_data;
        vector<mainRef_Point> all_refSet;

        vector<Point> pivots;
        pivots.reserve(num_clu);
        vector<vector<Point>> oth_pivots;
        oth_pivots.reserve(num_clu);

        string data_path = argv[6];

        unsigned filetype = stoi(argv[7]);

        disType = stoi(argv[8]);

        string filename = data_path + "/" + to_string(filetype) + "/ref/ref.txt";
        ifstream fin;
        fin.open(filename);
        if (!fin) {
            outFile << filename << " file could not be opened\n";
            exit(0);
        }
        string line;
        while (getline(fin, line)) {
            stringstream ss(line);
            string value;
            vector<double> coordinate;
            while (getline(ss, value, ',')) {
                coordinate.push_back(atof(value.c_str()));
            }
            Point pivot_pt = Point(coordinate);
            pivots.push_back(pivot_pt);
        }
        fin.close();

        for (unsigned p = 0; p < num_clu; ++p) {
            string filename = data_path + "/" + to_string(filetype) + "/ref/ref_" + to_string(p) + ".txt";
            ifstream fin;
            fin.open(filename);
            if (!fin) {
                outFile << filename << " file could not be opened\n";
                exit(0);
            }
            string line;

            vector<Point> other_pivot;
            while (getline(fin, line)) {
                stringstream ss(line);
                string value;
                vector<double> coordinate;
                while (getline(ss, value, ',')) {
                    coordinate.push_back(atof(value.c_str()));
                }
                Point pivot_pt = Point(coordinate);
                other_pivot.push_back(pivot_pt);
            }
            oth_pivots.push_back(other_pivot);
            fin.close();
        }

        double build_time = 0.0;
        disOpts = 0;
        for (unsigned i = 0; i < num_clu; i++) {
            InputReader inputReader(data_path + "/" + to_string(filetype) + "/clu/clu_" + to_string(i) + ".txt");
            all_data.push_back(inputReader.getCluster());

            if (all_data[i].clu_point.empty()) {
                outFile << "Plz do not load a null file" << endl;
                return 0;
            }

            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            ChooseRef ref_point(num_ref - 1, all_data[i], pivots[i], oth_pivots[i], 1);
            CaculateIValue calIValue(all_data[i], ref_point.getMainRefPoint());
            chrono::steady_clock::time_point end = chrono::steady_clock::now();

            build_time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

            all_data[i] = calIValue.getCluster();
            all_refSet.push_back(calIValue.getMainRef_Point());
        }

        outFile << "build time : " << build_time << "[ns]" << endl;
        outFile << "build dis opt : " << disOpts << endl;

        disOpts = 0;
        size_t total_memory = 0;


        for (const auto &clu: all_data) {
            total_memory += clu.memoryUsage();
        }


        for (const auto &ref: all_refSet) {
            total_memory += ref.memoryUsage();
        }


        total_memory += pivots.size() * sizeof(Point);
        for (const auto &p: pivots) {
            total_memory += p.memoryUsage();
        }


        for (const auto &vec: oth_pivots) {
            total_memory += vec.size() * sizeof(Point);
            for (const auto &p: vec) {
                total_memory += p.memoryUsage();
            }
        }

        outFile << "Memory usage: " << total_memory << " B" << endl;


        std::vector<unsigned> kArray = parseArrayK(argv[9]);
        std::vector<double> rArray = parseArrayR(argv[10]);

        double time = 0.0;

        filename = data_path + "/range.txt";
        vector<Point> list_KNN = LoadPointForQuery(filename);

        for (unsigned K: kArray) {
            time = 0.0;
            disOpts = 0;

            for (auto &m: list_KNN) {


                double init_r = stod(argv[11]);
                double delta_r = stod(argv[12]);

                int8_t arr[num_clu][1000000] = {0};
                chrono::steady_clock::time_point begin = chrono::steady_clock::now();

                priority_queue<pair<double, unsigned int>> KNNRes_queue;
                while (KNNRes_queue.size() < K || KNNRes_queue.top().first > init_r + delta_r) {

                    for (unsigned i = 0; i < num_clu; ++i) {
                        CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, m, init_r);
                        if (mainRefPtCircle.label == 1) {
                            continue;
                        }
                        if (mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) {
                            continue;
                        }
                        if (mainRefPtCircle.label == 3 && init_r > all_refSet[i].r) {
                            for (unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j) {
                                double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], m);
                                disOpts++;
                                if (KNNRes_queue.size() < K) {
                                    arr[i][j] = 1;
                                    pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[j].id);
                                    KNNRes_queue.push(elem);
                                } else {
                                    if (KNNRes_queue.top().first > dis_pt_qrpt && arr[i][j] == 0) {
                                        arr[i][j] = 1;
                                        KNNRes_queue.pop();
                                        pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[j].id);
                                        KNNRes_queue.push(elem);
                                    }
                                }
                            }
                            continue;
                        }

                        bool flag = true;
                        vector<CalCirclePos> ref_query;
                        ref_query.reserve(num_ref - 1);
                        for (unsigned j = 0; j < num_ref - 1; ++j) {
                            CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r,
                                                     m, init_r);
                            if (RefPtCircle.label == 1) {
                                flag = false;
                                break;
                            }
                            if (RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low) {
                                flag = false;
                                break;
                            }
                            if (RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r) {
                                for (unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l) {
                                    double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], m);
                                    disOpts++;
                                    if (KNNRes_queue.size() < K) {
                                        arr[i][l] = 1;
                                        pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[l].id);
                                        KNNRes_queue.push(elem);
                                    } else {
                                        if (KNNRes_queue.top().first > dis_pt_qrpt && arr[i][l] == 0) {
                                            arr[i][l] = 1;
                                            KNNRes_queue.pop();
                                            pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[l].id);
                                            KNNRes_queue.push(elem);
                                        }
                                    }
                                }
                                break;
                            }
                            ref_query.push_back(RefPtCircle);
                        }

                        if (!flag)
                            continue;

                        KNN KNNQuery(m, all_refSet[i], KNNRes_queue, mainRefPtCircle, ref_query, K, arr, i);
                    }

                    init_r += delta_r;
                }

                for (unsigned i = 0; i < num_clu; ++i) {
                    CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, m, init_r);
                    if (mainRefPtCircle.label == 1) {
                        continue;
                    }
                    if (mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) {
                        continue;
                    }
                    if (mainRefPtCircle.label == 3 && init_r > all_refSet[i].r) {
                        for (unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j) {
                            double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], m);
                            disOpts++;
                            if (KNNRes_queue.size() < K) {
                                arr[i][j] = 1;
                                pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[j].id);
                                KNNRes_queue.push(elem);
                            } else {
                                if (KNNRes_queue.top().first > dis_pt_qrpt && arr[i][j] == 0) {
                                    KNNRes_queue.pop();
                                    pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[j].id);
                                    KNNRes_queue.push(elem);
                                }
                            }
                        }
                        continue;
                    }

                    bool flag = true;
                    vector<CalCirclePos> ref_query;
                    ref_query.reserve(num_ref - 1);
                    for (unsigned j = 0; j < num_ref - 1; ++j) {
                        CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r,
                                                 m, init_r);
                        if (RefPtCircle.label == 1) {
                            flag = false;
                            break;
                        }
                        if (RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low) {
                            flag = false;
                            break;
                        }
                        if (RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r) {
                            for (unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l) {
                                double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], m);
                                disOpts++;
                                if (KNNRes_queue.size() < K) {
                                    arr[i][l] = 1;
                                    pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[l].id);
                                    KNNRes_queue.push(elem);
                                } else {
                                    if (KNNRes_queue.top().first > dis_pt_qrpt && arr[i][l] == 0) {
                                        arr[i][l] = 1;
                                        KNNRes_queue.pop();
                                        pair<double, unsigned int> elem(dis_pt_qrpt, all_refSet[i].iValuePts[l].id);
                                        KNNRes_queue.push(elem);
                                    }
                                }
                            }
                            break;
                        }
                        ref_query.push_back(RefPtCircle);
                    }

                    if (!flag)
                        continue;

                    KNN KNNQuery(m, all_refSet[i], KNNRes_queue, mainRefPtCircle, ref_query, K, arr, i);
                }

                chrono::steady_clock::time_point end = chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            }

        }

        string rangeQuery_filename = data_path + "/range.txt";
        vector<Point> list_rangeQry = LoadPointForQuery(rangeQuery_filename);

        for (double r: rArray) {
            time = 0.0;
            disOpts = 0;


            if (list_rangeQry.empty()) {
                outFile << "Warning: No range queries to process for r = " << r << endl;
                continue;
            }

            for (auto &m: list_rangeQry) {
                vector<int> rangeQueryRes;
                rangeQueryRes.reserve(10000);

                chrono::steady_clock::time_point begin = chrono::steady_clock::now();

                for (unsigned i = 0; i < num_clu; ++i) {
                    CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, m, r);
                    if (mainRefPtCircle.label == 1) {
                        continue;
                    }
                    if (mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) {
                        continue;
                    }
                    if (mainRefPtCircle.label == 3 && r > all_refSet[i].r) {
                        for (const auto &pt: all_refSet[i].iValuePts) {
                            rangeQueryRes.push_back(pt.id);
                        }
                        continue;
                    }

                    bool flag = true;
                    vector<CalCirclePos> ref_query;
                    ref_query.reserve(num_ref - 1);
                    for (unsigned j = 0; j < num_ref - 1; ++j) {
                        CalCirclePos RefPtCircle(
                                all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r, m, r
                        );
                        if (RefPtCircle.label == 1) {
                            flag = false;
                            break;
                        }
                        ref_query.push_back(RefPtCircle);
                    }

                    if (!flag)
                        continue;

                    RangeQuery rangeQuery(m, r, all_refSet[i], rangeQueryRes, mainRefPtCircle, ref_query);
                }

                chrono::steady_clock::time_point end = chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::microseconds>(end - begin).count();
            }

        }

        return 0;
    }
}