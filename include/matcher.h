#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <cmath>

class Matcher {
public:
//	Matcher(const Eigen::MatrixXd& pt_src, const Eigen::MatrixXd& pt_trg);
	std::tuple<int, Eigen::MatrixXd> icp(const Eigen::MatrixXd& pt_src, const Eigen::MatrixXd& pt_trg, int max_iterations, double tolerance);

private:
	std::vector<int> nearest_neighbor(const Eigen::MatrixXd& pt_src, const Eigen::MatrixXd& pt_trg);
	Eigen::MatrixXd arun(const Eigen::MatrixXd& pt_src, const Eigen::MatrixXd& pt_trg);
};
