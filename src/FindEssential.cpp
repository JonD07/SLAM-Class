#include<iostream>
#include<Eigen/Dense>

int main() {
	// Two sets of corresponding points
	double p[8][3] = {
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{3, 3, 6},
		{5, 4, 5},
		{2, 2, 3},
		{3, 6, 7},
		{1, 0, 6}};
	double pp[8][3] = {
		{1.5, 1.5, 8.5},
		{4.4, 3.4, 5.4},
		{7.7, 2.7, 3.7},
		{3.9, 4.9, 7.8},
		{5.9, 9.8, 9.9},
		{2.4, 7.4, 2.4},
		{3.1, 2.2, 5.1},
		{1.4, 6.4, 8.4}};

	// Build A matrix (Kronecker Product)
	Eigen::Matrix<double, 8, 9> A;
	for(int i = 0; i < 8; i++) {
		// First entry in pp
		A(i,0) = pp[i][0]*p[i][0];
		A(i,1) = pp[i][0]*p[i][1];
		A(i,2) = pp[i][0]*p[i][2];
		// Second entry in pp
		A(i,3) = pp[i][1]*p[i][0];
		A(i,4) = pp[i][1]*p[i][1];
		A(i,5) = pp[i][1]*p[i][2];
		// Third entry in pp
		A(i,6) = pp[i][2]*p[i][0];
		A(i,7) = pp[i][2]*p[i][1];
		A(i,8) = pp[i][2]*p[i][2];
	}
	std::cout << "Matrix A:\n" << A << std::endl;

	// Complete SVD for Ax = 0
	Eigen::BDCSVD<Eigen::Matrix<double, 8, 9>> svd;
	svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::VectorXd x_vec = svd.matrixV().transpose().col(8);

	const Eigen::Matrix3d E = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(x_vec.data());

	std::cout << "Final: " << std::endl << E << std::endl;

	// Force to rank 2
	Eigen::BDCSVD<Eigen::Matrix<double, 3, 3>> svdE;

	svdE.compute(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix<double,3,3>De;

	De << 1,0,0,
		  0,1,0,
		  0,0,0;

	Eigen::Matrix<double,3,3>E_hat = svdE.matrixU() * De * svdE.matrixV().transpose();

	std::cout << "Rank2: " << std::endl << E_hat << std::endl;
}
