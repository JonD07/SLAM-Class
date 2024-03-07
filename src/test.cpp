#include<iostream>
#include<time.h>
#include<Eigen/Dense>

#include "matcher.h"


int main() {
	printf("Hello\n");

	// random point generation
	srand(clock_gettime(0,NULL));

	// Generate a random matrix then randomly rotate/translate it
	int num_points = 5;
//	Eigen::Matrix3d pt1 = Eigen::MatrixXd::Random(num_points);

	// call ICP lib to match points
	Matcher matcher;
//	matcher.icp()
}
