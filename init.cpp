#include "init.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "./Eigen/Dense"


Eigen::MatrixXd InputData =  InitData();

//   ###########################
//   Initialzation and file reading, Reads to MatrixXd InputData
//   Linear Regression #1 Fitting data to bisecting Plane
//   ##########################

Eigen::VectorXd regressOut = regress(InputData);

double a = regressOut(0);
double b = regressOut(1);
double c = regressOut(2);

for (int i = 0; i < InputData.cols(); i++) {
    InputData(2, i) -= c;
}
//   ###########################
//   Squishing data onto the plane
//   ###########################

Eigen::MatrixXd planeData = PlanifyData(InputData, a, b);

//   ###########################
//   Finding Rotation of that plane compared to XY plane
//   &
//   Rotating plane back to XY plane
//   ###########################

Eigen::MatrixXd FlatData = FlattenData(planeData, a, b);
Eigen::VectorXd State {{0, 0, 0}};

for (int i = 0; i < 5; i++)
    State = updateState(State, FlatData);

//   ###########################
//   The Hair tearing out part to do in CPP, optimization
//   Finding critical points of squared error function and solving for
//   translation and radius values of the fit circle
//   ###########################

//   ###########################
//   Chose Real valued answer with r > 0
//   ###########################

std::cout << "The x location of the circle is: " << State(0) << '\n';
std::cout << "The y location of the circle is: " << State(1) << '\n';
std::cout << "The radius of the circle is: " << State(2) << '\n';



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

Eigen::MatrixXd InitData() {
    const static int datasize = 570;
    Eigen::MatrixXd dataIn(3, datasize);
    std::string line;
    std::ifstream myfile("data.txt");
    if (myfile.is_open()) {
        int i = 0;
        while (getline(myfile, line)) {

            std::vector<double> nums = strSplit(line);

            std::cout << nums[0] << ' ' << nums[1] << ' ' << nums[2] << '\n';
            std::cout << "i = " << i << '\n';

            dataIn(0, i) = nums[0];
            dataIn(1, i) = nums[1];
            dataIn(2, i) = nums[2];

            i++;
        }
        myfile.close();
    } else std::cout << "Unable to open file";
    return dataIn;
}

Eigen::VectorXd regress(Eigen::MatrixXd InputData) {
    double xxsum = 0;
    double xysum = 0;
    double xzsum = 0;
    double yysum = 0;
    double yzsum = 0;
    double xsum = 0;
    double ysum = 0;
    double zsum = 0;
    int n = 0;

    for (int i = 0; i < InputData.rows(); i++) {
        xxsum += InputData(0, i) * InputData(0, i);
        xysum += InputData(0, i) * InputData(1, i);
        xzsum += InputData(0, i) * InputData(2, i);
        yysum += InputData(1, i) * InputData(1, i);
        yzsum += InputData(1, i) * InputData(2, i);
        xsum += InputData(0, i);
        ysum += InputData(1, i);
        zsum += InputData(2, i);
        n++;
    }
    Eigen::MatrixXd computeMat {
        {xxsum, xysum, xsum},
        {xysum, yysum, ysum},
        {xsum, ysum, double(n)}
    };
    
    Eigen::VectorXd computeVec {{xzsum, yzsum, zsum}};

    Eigen::VectorXd x = computeMat.colPivHouseholderQr().solve(computeVec);
    Eigen::VectorXd outArray {{0, 0, 0}};
    return outArray;
}

Eigen::MatrixXd PlanifyData(Eigen::MatrixXd DataIn, double a, double b) {
    Eigen::MatrixXd ProjectionMatrix {
        {a, b, -1},
        {a, b, -1},
        {a, b, -1}
    };

    Eigen::MatrixXd Difmatrix = ProjectionMatrix * DataIn;
    for (int i = 0; i < Difmatrix.cols()) {
        Difmatrix(1, i) *= a ;
        Difmatrix(2, i) *= b ;
        Difmatrix(3, i) *= -1 ;
    }
    return DataIn - Difmatrix;
}

Eigen::MatrixXd FlattenData(Eigen::MatrixXd DataIn, double a, double b) {
    double norm = pow((a * a + b * b + 1), 0.5);
    double theta = acos(-1/norm);
    double phi = asin(a/(pow(1 - 1/(a * a + b * b + 1), 0.5)*norm));

    Eigen::MatrixXd Aout {
        {            cos(phi),             sin(phi),          0),                                                                                             sin(phi)/(cos(phi)^2 + sin(phi)^2),                                        0},
        {-cos(theta)*sin(phi),  cos(phi)*cos(theta), sin(theta)},
        { sin(phi)*sin(theta), -cos(phi)*sin(theta), cos(theta)}
    };
    return Aout * DataIn;
}

Eigen::MatrixXd updateState(Eigen::VectorXd State, Eigen::MatrixXd DataIn) {
    Eigen::MatrixXd Jacobian() {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
    };
    Eigen::MatrixXd Function() {
        {{0, 0, 0}};
    };
    for (int i = 0; i < DataIn.cols(); i++) {
        Jacobian(0, 0) += 12 * State(0) * State(0) - 24 * State(0) * DataIn(1, i) + 4 * State(1) * State(1) - 8 * State(1) * DataIn(2, i)  -  4 * State(2) * State(2) + 12 * DataIn(1, i) * DataIn(1, i)  +  4 * DataIn(2, i) * DataIn(2, i);
        Jacobian(0, 1) += 8 * State(0)* State(1)  -  8 * State(0)* DataIn(2, i)  -  8 * State(1) * DataIn(1, i)  +  8 * DataIn(1, i) * DataIn(2, i);
        Jacobian(0, 2) += 8 * State(2) * DataIn(1, i)  -  8 * State(0)* State(2);
        Jacobian(1, 0) += 8 * State(0)* State(1)  -  8 * State(0)* DataIn(2, i)  -  8 * State(1) * DataIn(1, i)  +  8 * DataIn(1, i) * DataIn(2, i);
        Jacobian(1, 1) += 4 * State(0)* State(0) -  8 * State(0)* DataIn(1, i) + 12 * State(1) * State(1)  -  24 * State(1) * DataIn(2, i)  -  4 * State(2) * State(1)  +  4 * DataIn(1, i) * DataIn(1, i)  +  12 * DataIn(2, i) * DataIn(2, i);
        Jacobian(1, 2) += 8 * State(2) * DataIn(2, i)  -  8 * State(1) * State(2);
        Jacobian(2, 0) += 8 * State(2) * DataIn(1, i)  -  8 * State(0)* State(2);
        Jacobian(2, 1) += 8 * State(2) * DataIn(2, i)  -  8 * State(1) * State(2);
        Jacobian(2, 2) += -4 * State(0)* State(0) +  8 * State(0)* DataIn(1, i)  -  4 * State(1) * State(1)  +  8 * State(1) * DataIn(2, i)  +  12 * State(2) * State(2)  -  4 * DataIn(1, i) * DataIn(1, i)  -  4 * DataIn(2, i) * DataIn(2, i)]

        Function(0) += 4 * State(0) * State(0) * State(0)  -  12 * State(0) * State(0) * DataIn(1, i)  +  4 * State(0) * State(1) * State(1)  -  8 * State(0) * State(1) * DataIn(2, i)  -  4 * State(0) * State(2) * State(2)  +  12 * State(0) * DataIn(1, i) * DataIn(1, i)  +  4 * State(0) * DataIn(2, i) * DataIn(2, i)  -  4 * State(1) * State(1) * DataIn(1, i)  +  8 * State(1) * DataIn(1, i) * DataIn(2, i)  +  4 * State(2) * State(2) * DataIn(1, i)  -  4 * DataIn(1, i) * DataIn(1, i) * DataIn(1, i)  -  4 * DataIn(1, i) * DataIn(2, i) * DataIn(2, i);
        Function(1) += 4 * State(0) * State(0) * State(1)  -  4 * State(0) * State(0) * DataIn(2, i)  -  8 * State(0) * State(1) * DataIn(1, i)  +  8 * State(0) * DataIn(1, i) * DataIn(2, i)  +  4 * State(1) * State(1) * State(1)  -  12 * State(1) * State(1) * DataIn(2, i)  -  4 * State(1) * State(2) * State(2)  +  4 * State(1) * DataIn(1, i) * DataIn(1, i)  +  12 * State(1) * DataIn(2, i) * DataIn(2, i)  +  4 * State(2) * State(2) * DataIn(2, i)  -  4 * DataIn(1, i) * DataIn(1, i) * DataIn(2, i)  -  4 * DataIn(2, i) * DataIn(2, i) * DataIn(2, i);
        Function(2) += -4 * State(0) * State(0) * State(2)  +  8 * State(0) * State(2) * DataIn(1, i)  -  4 * State(1) * State(1) * State(2)  +  8 * State(1) * State(2) * DataIn(2, i)  +  4 * State(2) * State(2) * State(2)  -  4 * State(2) * DataIn(1, i) * DataIn(1, i)  -  4 * State(2) * DataIn(2, i) * DataIn(2, i);
    }
    return State - Jacobian.inverse() * Function;
}




/*
 * ans =

4*a^3 - 12*a^2*x + 4*a*b^2 - 8*a*b*y - 4*a*r^2 + 12*a*x^2 + 4*a*y^2 - 4*b^2*x + 8*b*x*y + 4*r^2*x - 4*x^3 - 4*x*y^2


ans =

4*a^2*b - 4*a^2*y - 8*a*b*x + 8*a*x*y + 4*b^3 - 12*b^2*y - 4*b*r^2 + 4*b*x^2 + 12*b*y^2 + 4*r^2*y - 4*x^2*y - 4*y^3


ans =

- 4*a^2*r + 8*a*r*x - 4*b^2*r + 8*b*r*y + 4*r^3 - 4*r*x^2 - 4*r*y^2

 functionState = [
    4*a^3 - 12*a^2*x + 4*a*b^2 - 8*a*b*y - 4*a*r^2 + 12*a*x^2 + 4*a*y^2 - 4*b^2*x + 8*b*x*y + 4*r^2*x - 4*x^3 - 4*x*y^2;
    4*a^2*b - 4*a^2*y - 8*a*b*x + 8*a*x*y + 4*b^3 - 12*b^2*y - 4*b*r^2 + 4*b*x^2 + 12*b*y^2 + 4*r^2*y - 4*x^2*y - 4*y^3;
    - 4*a^2*r + 8*a*r*x - 4*b^2*r + 8*b*r*y + 4*r^3 - 4*r*x^2 - 4*r*y^2
];

 */