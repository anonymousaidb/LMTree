#include <string>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <sys/time.h>

#define NUUM 200000

#define CandA 300
using namespace std;

string loc[NUUM];
int dim;
int pnum = 2, num, PairA = 0;
int ptype = 0;
double mm = 0;
int **table1;


int minDistance(string word1, string word2) {
    int l1 = word1.length() + 1;
    int l2 = word2.length() + 1;
    int md[l1][l2];
    for (int j = 0; j < l2; j++) { md[0][j] = j; }
    for (int i = 0; i < l1; i++) { md[i][0] = i; }
    for (int i = 1; i < l1; i++) {
        for (int j = 1; j < l2; j++) {
            md[i][j] = min(min(md[i - 1][j] + 1, md[i][j - 1] + 1),
                           md[i - 1][j - 1] + (word1[i - 1] == word2[j - 1] ? 0 : 1));
        }
    }
    return md[l1 - 1][l2 - 1];
}

void readm(int p) {
    ifstream ain1;
    string filename = "dataset path" + to_string(p) + ".txt";
    ain1.open(filename, ifstream::in);
    num = 0;
    string str;
    while (getline(ain1, str)) {
        stringstream ss(str);
        string value;
        getline(ss, value, ',');
        string id = value;

        getline(ss, value, ',');
        loc[num] = value;
        num++;
    }
    PairA = num / 100;
    ain1.close();
    mm = 0;
    for (int i = 0; i < num; i++) if (loc[i].length() > mm) mm = loc[i].length();
}
