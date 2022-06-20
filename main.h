//
// Created by nickr on 2022-06-16.
//

#ifndef REGRESSION_MAIN_H
#define REGRESSION_MAIN_H

#include "./Eigen/Dense"

std::vector<double> strSplit(std::string str);

Eigen::MatrixXd InitData(int len);

Eigen::VectorXd regress(Eigen::MatrixXd InputData, int len);

Eigen::MatrixXd PlanifyData(Eigen::MatrixXd DataIn, double a, double b, int len);

Eigen::MatrixXd FlattenData(Eigen::MatrixXd DataIn, double a, double b);

Eigen::VectorXd calcSums(Eigen::MatrixXd data);

Eigen::VectorXd calcCircle(Eigen::VectorXd sums);

#endif //REGRESSION_MAIN_H
