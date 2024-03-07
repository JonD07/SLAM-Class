#include<iostream>
#include<Eigen/Dense>
#include<unsupported/Eigen/KroneckerProduct>

typedef Eigen::Matrix<double, 3, 1> Point3D;

Eigen::Matrix<double, 3, 8> applyTransformation(const Eigen::Matrix<double, 3, 8>& initialPoints,
                                                const Eigen::Matrix3d& rotation,
                                                const Eigen::Vector3d& translation) {
	Eigen::Matrix<double, 3, 8> transformedPoints;
	for(int i = 0; i < initialPoints.cols(); i++){
		transformedPoints.col(i) = rotation * initialPoints.col(i) + translation;
	}

	return transformedPoints;
}

int main() {
//	/// Manually build Kronecker Product
//	// Two sets of corresponding points
//	double p[8][3] = {
//		{1, 2, 3},
//		{4, 5, 6},
//		{7, 8, 9},
//		{3, 3, 6},
//		{5, 4, 5},
//		{2, 2, 3},
//		{3, 6, 7},
//		{1, 0, 6}};
//	double pp[8][3] = {
//		{1.5, 1.5, 8.5},
//		{4.4, 3.4, 5.4},
//		{7.7, 2.7, 3.7},
//		{3.9, 4.9, 7.8},
//		{5.9, 9.8, 9.9},
//		{2.4, 7.4, 2.4},
//		{3.1, 2.2, 5.1},
//		{1.4, 6.4, 8.4}};
//
//	// Build A matrix (Kronecker Product)
//	Eigen::Matrix<double, 8, 9> A;
//	for(int i = 0; i < 8; i++) {
//		// First entry in pp
//		A(i,0) = pp[i][0]*p[i][0];
//		A(i,1) = pp[i][0]*p[i][1];
//		A(i,2) = pp[i][0]*p[i][2];
//		// Second entry in pp
//		A(i,3) = pp[i][1]*p[i][0];
//		A(i,4) = pp[i][1]*p[i][1];
//		A(i,5) = pp[i][1]*p[i][2];
//		// Third entry in pp
//		A(i,6) = pp[i][2]*p[i][0];
//		A(i,7) = pp[i][2]*p[i][1];
//		A(i,8) = pp[i][2]*p[i][2];
//	}
//	std::cout << "Matrix A:\n" << A << std::endl;

	Eigen::Matrix<double, 3, 8> initialPoints;

	//Test points
	initialPoints << Point3D(1.0, 2.0, 1.0),   // Point 1
					 Point3D(3.0, 4.0, 1.0),   // Point 2
					 Point3D(5.0, 6.0, 1.0),   // Point 3
					 Point3D(7.0, 8.0, 1.0),   // Point 4
					 Point3D(9.0, 10.0, 1.0),  // Point 5
					 Point3D(11.0, 12.0, 1.0), // Point 6
					 Point3D(13.0, 14.0, 1.0), // Point 7
					 Point3D(15.0, 16.0, 1.0); // Point 8

	Eigen::Matrix3d rotation;
	rotation << 0,-1,0,
				1,0,0,
				0,0,1;

	Eigen::Vector3d translation(5.0, 5.0, 0.0);


	//Calculate E exactly to compare
	Eigen::Matrix3d t_cross;
	t_cross << 0, -translation(2), translation(1),
			   translation(2), 0, -translation(0),
			   -translation(1), translation(0), 0;
	Eigen::Matrix3d E_ex = t_cross * rotation;
	std::cout << "Exact E: " << std::endl << E_ex << std:: endl;

	Eigen::Matrix<double, 3, 8> correspondences = applyTransformation(initialPoints,
																	  rotation,
																	  translation);
	//std::cout << "initial: " << std::endl << initialPoints << std::endl << "cor: " << std::endl << correspondences << std::endl;


	//Create A from the kronecker products of point pairs
	Eigen::Matrix<double, 8, 9> A;

	for(int i = 0; i < 8; i++){
		Eigen::Matrix<double, 9, 1> kronecker = Eigen::kroneckerProduct(correspondences.col(i), initialPoints.col(i));

		A.row(i) = kronecker;
	}

	//std::cout << A << std::endl;

	Eigen::BDCSVD<Eigen::Matrix<double, 8, 9>> svd;

	//Complete SVD for Ax = 0
	svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::VectorXd x_vec = svd.matrixV().transpose().col(8);

	const Eigen::Matrix3d E = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(x_vec.data());

	std::cout << "Final: " << std::endl << E << std::endl;

	//Force to rank 2
	Eigen::BDCSVD<Eigen::Matrix<double, 3, 3>> svdE;

	svdE.compute(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix<double,3,3>De;

	De << 1,0,0,
		  0,1,0,
		  0,0,0;

	Eigen::Matrix<double,3,3>E_hat = svdE.matrixU() * De * svdE.matrixV().transpose();

	std::cout << "Rank2: " << std::endl << E_hat << std::endl;
}
