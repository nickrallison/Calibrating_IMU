// reading a text file
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "./Eigen/Dense"
#include "main.h"

int main() {
    const static int datasize = 570;
    Eigen::MatrixXd InputData =  InitData(datasize);
    //std::cout << InputData(2, 0) << '\n';

//   ###########################
//   Initialzation and file reading, Reads to MatrixXd InputData
//   Linear Regression #1 Fitting data to bisecting Plane
//   ##########################

    Eigen::VectorXd regressOut = regress(InputData, datasize);

    double a = regressOut(0);
    double b = regressOut(1);
    double c = regressOut(2);
    //std::cout << a << ' ' << b << ' ' << c << '\n';

    for (int i = 0; i < datasize; i++) {
        InputData(2, i) -= c;
    }


//   ###########################
//   Squishing data onto the plane
//   ###########################


//   ###########################
//   Finding Rotation of that plane compared to xY plane
//   &
//   Rotating plane back to xY plane
//   ###########################
    double norm = pow((a * a + b * b + 1), 0.5);
    double theta = acos(-1/norm);
    double phi = asin(a/(pow(1 - 1/(a * a + b * b + 1), 0.5)*norm));

    Eigen::MatrixXd Aout {
            {            cos(phi),             sin(phi),          0},
            {-cos(theta)*sin(phi),  cos(phi)*cos(theta), sin(theta)},
            { sin(phi)*sin(theta), -cos(phi)*sin(theta), cos(theta)}
    };

    Eigen::MatrixXd FlatData = Aout * InputData;

    for (int i = 0; i < datasize; i++) {
        FlatData(2, i) = 0;
    }

    Eigen::VectorXd sums = calcSums(FlatData);

    //std::cout << sums << "  <-- Here are the sums\n";

    std::cout << calcCircle(sums) << '\n';


//   ###########################
//   The Hair tearing out part to do in CPP, optimization
//   Finding critical points of squared error function and solving for
//   translation and radius values of the fit circle
//   ###########################

//   ###########################
//   Chose Real valued answer with r > 0
//   ###########################
    return 1;
}

std::vector<double> strSplit(std::string str) {
    std::vector<double> nums = {};
    char* cstr = new char [str.length()+1];
    std::strcpy (cstr, str.c_str());
    char * p = std::strtok (cstr,",");
    while (p!=0)
    {

        nums.push_back(std::stod(p));
        p = std::strtok(NULL,",");
    }
    delete[] cstr;
    return nums;
}

Eigen::MatrixXd InitData(int len) {
    Eigen::MatrixXd dataIn(3, len);
    std::string line;
    std::ifstream myfile("data.txt");
    if (myfile.is_open()) {
        int i = 0;
        while (getline(myfile, line) && i < len) {

            std::vector<double> nums = strSplit(line);

            dataIn(0, i) = nums[0];
            dataIn(1, i) = nums[1];
            dataIn(2, i) = nums[2];

            i++;
        }
        myfile.close();
    } else std::cout << "Unable to open file";
    return dataIn;
}

Eigen::VectorXd regress(Eigen::MatrixXd InputData, int len) {
    double xxsum = 0;
    double xysum = 0;
    double xzsum = 0;
    double yysum = 0;
    double yzsum = 0;
    double xsum = 0;
    double ysum = 0;
    double zsum = 0;

    for (int i = 0; i < len; i++) {
        xxsum += InputData(0, i) * InputData(0, i);
        xysum += InputData(0, i) * InputData(1, i);
        xzsum += InputData(0, i) * InputData(2, i);
        yysum += InputData(1, i) * InputData(1, i);
        yzsum += InputData(1, i) * InputData(2, i);
        xsum += InputData(0, i);
        ysum += InputData(1, i);
        zsum += InputData(2, i);
    }
    //std::cout << double(len) << ' ' << xsum << ' ' << ysum<< '\n';
    Eigen::MatrixXd computeMat {
            {xxsum, xysum, xsum},
            {xysum, yysum, ysum},
            {xsum, ysum, double(len)}
    };

    Eigen::VectorXd computeVec {{xzsum, yzsum, zsum}};

    Eigen::VectorXd x = computeMat.inverse() * computeVec;
    Eigen::VectorXd outvec {{x(0), x(1), x(2), xsum / len, ysum / len}};
    return outvec;
}

Eigen::VectorXd calcSums(Eigen::MatrixXd data) {
    Eigen::VectorXd vecOut(10);
    vecOut(0) = 0;      //x
    vecOut(1) = 0;      //y
    vecOut(2) = 0;      //xx
    vecOut(3) = 0;      //xy
    vecOut(4) = 0;      //yy
    vecOut(5) = 0;      //xxx
    vecOut(6) = 0;      //xxy
    vecOut(7) = 0;      //xyy
    vecOut(8) = 0;      //yyy
    vecOut(9) = 0;      //n
    for (int i = 0; i < data.cols(); i++) {
        vecOut(0) += data(0, i);      //x
        vecOut(1) += data(1, i);      //y
        vecOut(2) += data(0, i) * data(0, i);      //xx
        vecOut(3) += data(0, i) * data(1, i);      //xy
        vecOut(4) += data(1, i) * data(1, i);      //yy
        vecOut(5) += data(0, i) * data(0, i) * data(0, i);      //xxx
        vecOut(6) += data(0, i) * data(0, i) * data(1, i);      //xxy
        vecOut(7) += data(0, i) * data(1, i) * data(1, i);      //xyy
        vecOut(8) += data(1, i) * data(1, i) * data(1, i);      //yyy
        vecOut(9)++;        //n
    }
    return vecOut;
}

Eigen::VectorXd calcCircle(Eigen::VectorXd sums) {
    double x = sums(0);
    double y = sums(1);
    double xx = sums(2);
    double xy = sums(3);
    double yy = sums(4);
    double xxx = sums(5);
    double xxy = sums(6);
    double xyy = sums(7);
    double yyy = sums(8);
    double n = sums(9);

    Eigen::VectorXd outVec(3);

    outVec(0) = (xxx*y*y + x*yy*yy + xyy*y*y + n*xxy*xy - n*xxx*yy + n*xy*yyy - n*xyy*yy - x*xxy*y + x*xx*yy - xx*xy*y - x*y*yyy - xy*y*yy)/(2*(yy*x*x - 2*x*xy*y + n*xy*xy + xx*y*y - n*xx*yy));
    outVec(1) = (x*x*xxy + xx*xx*y + x*x*yyy - n*xx*xxy + n*xxx*xy + n*xy*xyy - n*xx*yyy - x*xx*xy - x*xxx*y - x*xyy*y - x*xy*yy + xx*y*yy)/(2*(yy*x*x - 2*x*xy*y + n*xy*xy + xx*y*y - n*xx*yy));
    outVec(2) = -pow((n*n*xx*xx*xxy*xxy + 2*n*n*xx*xx*xxy*yyy + n*n*xx*xx*yyy*yyy - 2*n*n*xx*xxx*xxy*xy - 2*n*n*xx*xxx*xy*yyy - 2*n*n*xx*xxy*xy*xyy - 2*n*n*xx*xy*xyy*yyy + n*n*xxx*xxx*xy*xy + n*n*xxx*xxx*yy*yy - 2*n*n*xxx*xxy*xy*yy + 2*n*n*xxx*xy*xy*xyy - 2*n*n*xxx*xy*yy*yyy + 2*n*n*xxx*xyy*yy*yy + n*n*xxy*xxy*xy*xy + 2*n*n*xxy*xy*xy*yyy - 2*n*n*xxy*xy*xyy*yy + n*n*xy*xy*xyy*xyy + n*n*xy*xy*yyy*yyy - 2*n*n*xy*xyy*yy*yyy + n*n*xyy*xyy*yy*yy - 2*n*x*x*xx*xxy*xxy - 4*n*x*x*xx*xxy*yyy - 2*n*x*x*xx*yyy*yyy + 2*n*x*x*xxx*xxy*xy + 2*n*x*x*xxx*xy*yyy + 2*n*x*x*xxy*xy*xyy + 2*n*x*x*xy*xyy*yyy + 2*n*x*xx*xx*xxy*xy + 2*n*x*xx*xx*xy*yyy + 2*n*x*xx*xxx*xxy*y - 2*n*x*xx*xxx*xy*xy + 2*n*x*xx*xxx*y*yyy - 6*n*x*xx*xxx*yy*yy + 8*n*x*xx*xxy*xy*yy + 2*n*x*xx*xxy*xyy*y - 2*n*x*xx*xy*xy*xyy + 8*n*x*xx*xy*yy*yyy + 2*n*x*xx*xyy*y*yyy - 6*n*x*xx*xyy*yy*yy - 2*n*x*xxx*xxx*xy*y + 2*n*x*xxx*xxy*y*yy + 2*n*x*xxx*xy*xy*yy - 4*n*x*xxx*xy*xyy*y + 2*n*x*xxx*y*yy*yyy - 2*n*x*xxx*yy*yy*yy - 2*n*x*xxy*xxy*xy*y - 4*n*x*xxy*xy*xy*xy - 4*n*x*xxy*xy*y*yyy + 2*n*x*xxy*xy*yy*yy + 2*n*x*xxy*xyy*y*yy - 4*n*x*xy*xy*xy*yyy + 2*n*x*xy*xy*xyy*yy - 2*n*x*xy*xyy*xyy*y - 2*n*x*xy*y*yyy*yyy + 2*n*x*xy*yy*yy*yyy + 2*n*x*xyy*y*yy*yyy - 2*n*x*xyy*yy*yy*yy - 2*n*xx*xx*xx*xxy*y - 2*n*xx*xx*xx*y*yyy + 4*n*xx*xx*xx*yy*yy + 2*n*xx*xx*xxx*xy*y - 6*n*xx*xx*xxy*y*yy - 8*n*xx*xx*xy*xy*yy + 2*n*xx*xx*xy*xyy*y - 6*n*xx*xx*y*yy*yyy + 4*n*xx*xx*yy*yy*yy + 8*n*xx*xxx*xy*y*yy + 2*n*xx*xxy*xy*xy*y + 4*n*xx*xy*xy*xy*xy + 2*n*xx*xy*xy*y*yyy - 8*n*xx*xy*xy*yy*yy + 8*n*xx*xy*xyy*y*yy - 2*n*xxx*xxx*y*y*yy + 2*n*xxx*xxy*xy*y*y - 4*n*xxx*xy*xy*xy*y + 2*n*xxx*xy*y*y*yyy + 2*n*xxx*xy*y*yy*yy - 4*n*xxx*xyy*y*y*yy - 2*n*xxy*xy*xy*y*yy + 2*n*xxy*xy*xyy*y*y + 4*n*xy*xy*xy*xy*yy - 4*n*xy*xy*xy*xyy*y - 2*n*xy*xy*y*yy*yyy + 2*n*xy*xyy*y*y*yyy + 2*n*xy*xyy*y*yy*yy - 2*n*xyy*xyy*y*y*yy + x*x*x*x*xxy*xxy + 2*x*x*x*x*xxy*yyy + x*x*x*x*yyy*yyy - 2*x*x*x*xx*xxy*xy - 2*x*x*x*xx*xy*yyy - 2*x*x*x*xxx*xxy*y - 2*x*x*x*xxx*y*yyy + 4*x*x*x*xxx*yy*yy - 6*x*x*x*xxy*xy*yy - 2*x*x*x*xxy*xyy*y - 6*x*x*x*xy*yy*yyy - 2*x*x*x*xyy*y*yyy + 4*x*x*x*xyy*yy*yy + 2*x*x*xx*xx*xxy*y + x*x*xx*xx*xy*xy + 2*x*x*xx*xx*y*yyy - 3*x*x*xx*xx*yy*yy + 2*x*x*xx*xxx*xy*y + 4*x*x*xx*xxy*y*yy + 6*x*x*xx*xy*xy*yy + 2*x*x*xx*xy*xyy*y + 4*x*x*xx*y*yy*yyy - 2*x*x*xx*yy*yy*yy + x*x*xxx*xxx*y*y - 10*x*x*xxx*xy*y*yy + 2*x*x*xxx*xyy*y*y + x*x*xxy*xxy*y*y + 8*x*x*xxy*xy*xy*y + 2*x*x*xxy*y*y*yyy - 2*x*x*xxy*y*yy*yy + 8*x*x*xy*xy*y*yyy + 5*x*x*xy*xy*yy*yy - 10*x*x*xy*xyy*y*yy + x*x*xyy*xyy*y*y + x*x*y*y*yyy*yyy - 2*x*x*y*yy*yy*yyy + x*x*yy*yy*yy*yy - 2*x*xx*xx*xx*xy*y - 2*x*xx*xx*xxx*y*y + 2*x*xx*xx*xy*y*yy - 2*x*xx*xx*xyy*y*y + 4*x*xx*xxx*y*y*yy - 10*x*xx*xxy*xy*y*y - 8*x*xx*xy*xy*xy*y - 10*x*xx*xy*y*y*yyy + 2*x*xx*xy*y*yy*yy + 4*x*xx*xyy*y*y*yy - 2*x*xxx*xxy*y*y*y + 8*x*xxx*xy*xy*y*y - 2*x*xxx*y*y*y*yyy + 2*x*xxx*y*y*yy*yy + 2*x*xxy*xy*y*y*yy - 2*x*xxy*xyy*y*y*y - 8*x*xy*xy*xy*y*yy + 8*x*xy*xy*xyy*y*y + 2*x*xy*y*y*yy*yyy - 2*x*xy*y*yy*yy*yy - 2*x*xyy*y*y*y*yyy + 2*x*xyy*y*y*yy*yy + xx*xx*xx*xx*y*y - 2*xx*xx*xx*y*y*yy + 4*xx*xx*xxy*y*y*y + 5*xx*xx*xy*xy*y*y + 4*xx*xx*y*y*y*yyy - 3*xx*xx*y*y*yy*yy - 6*xx*xxx*xy*y*y*y + 6*xx*xy*xy*y*y*yy - 6*xx*xy*xyy*y*y*y + xxx*xxx*y*y*y*y - 2*xxx*xy*y*y*y*yy + 2*xxx*xyy*y*y*y*y + xy*xy*y*y*yy*yy - 2*xy*xyy*y*y*y*yy + xyy*xyy*y*y*y*y), 0.5)/(2*(yy*x*x - 2*x*xy*y + n*xy*xy + xx*y*y - n*xx*yy));
    return outVec;
}



