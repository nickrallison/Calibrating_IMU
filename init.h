#include "./Eigen/Dense"

std::vector<double> strSplit(std::string str);

Eigen::MatrixXd InitData();

Eigen::VectorXd regress(Eigen::MatrixXd InputData);

Eigen::MatrixXd PlanifyData(Eigen::MatrixXd DataIn, double a, double b);

Eigen::MatrixXd FlattenData(Eigen::MatrixXd DataIn, double a, double b);

Eigen::MatrixXd updateState(Eigen::VectorXd State, Eigen::MatrixXd DataIn);