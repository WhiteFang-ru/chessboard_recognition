
#include <cstdio>
#include <utility> // For std::pair etc
#include "math_methods.cpp"
#include <cmath> // for M_PI etc
#include <algorithm>
#include <set>
#include <vector>
#include <stdexcept>
#include <numeric>
using namespace Math;


const int parallel_threshold = 64; // intersections of 2 lines occured on distance > {}*image_size is assumed as parallel

namespace Utils
{
    void cannyPF(const cv::Mat& image, const cv::Mat& image_after_canny, const double sigma = 0.25) {
        double med = Math::median_grayscale(image);
        double lower = std::max(0.0, (1.0 - sigma) * med);
        double upper = std::min(255.0, (1.0 - sigma) * med);
        cv::Canny(image, image_after_canny, lower, upper);
        return;
    }

    //  compute intersection of two lines
    std::pair<double, double> intersection(const cv::Vec2d& line1, const cv::Vec2d& line2, const cv::Size& img_shape) {  
        double rho1 = line1[0], theta1 = line1[1];
        double rho2 = line2[0], theta2 = line2[1];

        // matrix A (2x2) and the right-hand side vector b (2x1)
        cv::Mat A = (cv::Mat_<double>(2, 2) << cos(theta1), sin(theta1), cos(theta2), sin(theta2));
        cv::Mat b = (cv::Mat_<double>(2, 1) << rho1, rho2);

        // Solve the system A * [x0, y0] = b
        cv::Mat x0y0;
        try {
            // Use SVD decomposition to solve the system of linear equations A * [x0, y0] = b
            cv::solve(A, b, x0y0, cv::DECOMP_SVD);
        } catch (const cv::Exception &e) {
            // If lines are parallel), return an invalid intersection point
            std::cerr << "Error solving the system: " << e.what() << std::endl;
            return std::make_pair(-1.0, -1.0); // no valid intersection
        }

        double x0 = x0y0.at<double>(0, 0);  // x-coordinate
        double y0 = x0y0.at<double>(1, 0);  // y-coordinate

        if (std::abs(x0) > img_shape.width * parallel_threshold || std::abs(y0) > img_shape.height * parallel_threshold)
        {
            return std::make_pair(-1.0, -1.0); // Invalid intersection point
        }

        return std::make_pair(y0, x0);
    }


    
    cv::Vec2d fix_hessian_form(const cv::Vec2d& line, bool reverse = false) {
        if (!reverse && line[0] < 0) {
            double new_rho = -line[0];
            double new_alpha = -(M_PI - line[1]);
            return cv::Vec2d(new_rho, new_alpha);
        } else if (reverse && line[1] < 0) {
            double new_rho = -line[0];
            double new_alpha = M_PI + line[1];
            return cv::Vec2d(new_rho, new_alpha);
        }
        return line;
    }


    std::vector<cv::Vec2d> fix_hessian_form_vectorized (const std::vector<cv::Vec2d>& lines) 
    {
        std::vector<cv::Vec2d> resulting_lines = lines;

        std::vector<bool> neg_rho_mask(resulting_lines.size());

        for (size_t i = 0; i < resulting_lines.size(); ++i) {
            neg_rho_mask[i] = resulting_lines[i][0] < 0; //rho value (index 0)
        }
        for (size_t i = 0; i < resulting_lines.size(); ++i) {
            if (neg_rho_mask[i]) {
                resulting_lines[i][0] *= -1;      // Negate rho
                resulting_lines[i][1] -= M_PI;    // Subtract pi from theta
            }
        }

        return resulting_lines;
}



    double angle_diff(const cv::Vec2d& line1, const cv::Vec2d& line2) { 
        double diff = std::numeric_limits<double>::infinity();

        if ((line1[0] < 0) ^ (line2[0] < 0)) {
            if (line1[0] < 0) {
                diff = std::abs(Utils::fix_hessian_form(line1)[1] - line2[1]);
            } else {
                diff = std::abs(line1[1] - Utils::fix_hessian_form(line2)[1]);
            }
            diff = std::fmod(diff, M_PI);  // Equivalent to  % M_PI
        }

        diff = std::min(diff, std::fmod(std::abs(line1[1] - line2[1]), M_PI));

        return diff;
    }

    std::vector<double> angle_diff_vectorized(const std::vector<cv::Vec2d>& lines, const cv::Vec2d& line_to_calculate_diff) {
        std::vector<cv::Vec2d> hess_fixed_lines = Utils::fix_hessian_form_vectorized (lines);
        cv::Vec2d hess_fixed_calculate_line = Utils::fix_hessian_form(line_to_calculate_diff);
        std::vector<double> diff_fixed(lines.size(), std::numeric_limits<double>::infinity());

        std::vector<bool> hess_test_mask(lines.size());
        for (size_t i = 0; i < lines.size(); ++i) {
            hess_test_mask[i] = lines[i][0] < 0;
        }

        if (line_to_calculate_diff[0] >= 0) {
            for (size_t i = 0; i < lines.size(); ++i) {
                if (hess_test_mask[i]) {
                    diff_fixed[i] = std::fmod(std::abs(hess_fixed_lines[i][1] - line_to_calculate_diff[1]), M_PI);
                }
            }
        } else {
            for (size_t i = 0; i < lines.size(); ++i) {
                if (hess_test_mask[i] == 0) {
                    diff_fixed[i] = std::fmod(lines[i][1] - hess_fixed_calculate_line[1], M_PI);
                }
            }
        }
        std::vector<double> diff_normal(lines.size());

        for (size_t i = 0; i < lines.size(); ++i) {
            diff_normal[i] = std::fmod(std::abs(lines[i][1] - line_to_calculate_diff[1]), M_PI);
        }

        std::vector<double> minimum(lines.size());

        for (size_t i = 0; i < lines.size(); ++i) {
            minimum[i] = std::min(diff_normal[i], diff_fixed[i]);
        }

        return minimum;
    }


    cv::Vec2d cluster_mean_hessfixed(const std::vector<cv::Vec2d>& lines_best, const std::set<size_t>& cluster) {
        std::vector<cv::Vec2d> cluster_lines;

        for (size_t index : cluster) {
            if (index < lines_best.size()) {
                cluster_lines.push_back(lines_best[index]);
            }
        }
        cv::Vec2d normal_mean = Math::calculate_mean(cluster_lines); 

        std::vector<cv::Vec2d> hess_fixed_cluster_lines = Utils::fix_hessian_form_vectorized(cluster_lines);

        cv::Vec2d hess_fixed_mean = Math::calculate_mean(hess_fixed_cluster_lines);

        std::vector<double> angle_dv_normal = Utils::angle_diff_vectorized(cluster_lines, normal_mean);
        double normal_mean_diff = std::accumulate(angle_dv_normal.begin(), angle_dv_normal.end(), 0.0) / angle_dv_normal.size();

        std::vector<double> angle_dv_hess = Utils::angle_diff_vectorized(cluster_lines, hess_fixed_mean);
        double hess_fixed_mean_diff = std::accumulate(angle_dv_hess.begin(), angle_dv_hess.end(), 0.0) / angle_dv_hess.size();

        if(normal_mean_diff < hess_fixed_mean_diff) {
            return normal_mean;
        } else {
            return Utils::fix_hessian_form(hess_fixed_mean, true);
        }
    }



    template<typename T>
    std::vector<std::vector<T>> generate_combinations(const std::vector<T>& vec, size_t combo_size) {
        std::vector<std::vector<T>> combinations;
        size_t n = vec.size();
    
        if (combo_size > n) return combinations;

        std::vector<size_t> indices(combo_size);
        std::iota(indices.begin(), indices.end(), 0);
    
        while (true) {
            std::vector<T> combination;
            for (size_t i : indices) {
                combination.push_back(vec[i]);
            }
            combinations.push_back(combination);
        
            size_t i = combo_size - 1;
            while (i >= 0 && indices[i] == i + n - combo_size) {
                --i;
            }
            if (i < 0) break;
        
            ++indices[i];
            for (size_t j = i + 1; j < combo_size; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
        }
        return combinations;
    }



    std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int polynomial_degree = 3) {
        assert(x.size() == y.size());
        int n = x.size();
        assert(n > polynomial_degree);

        std::vector<std::vector<double>> X(polynomial_degree + 1, std::vector<double>(polynomial_degree + 1, 0));
        std::vector<double> b(polynomial_degree + 1, 0);

        // Build the matrix X and vector b
        for (int i = 0; i <= polynomial_degree; ++i) {
            for (int j = 0; j <= polynomial_degree; ++j) {
                for (int k = 0; k < n; ++k) {
                    X[i][j] += std::pow(x[k], i + j);
                }
            }
            for (int k = 0; k < n; ++k) {
                b[i] += std::pow(x[k], i) * y[k];
            }
        }


        std::vector<double> coeffs(polynomial_degree + 1, 0.0);

        for (int j = 0; j <= polynomial_degree; ++j) {
            for (int i = j + 1; i < n; ++i) {
                double factor = X[i][j] / X[j][j];
                for(int k = j; k <= polynomial_degree; ++k) {
                    X[i][k] -= factor * X[j][k];
                }
                b[i] -= factor * b[j];
            }
        }

        for (int i = polynomial_degree; i >= 0; --i) {
            coeffs[i] = b[i] / X[i][i];
            for(int j = i - 1; j >= 0; --j) {
                b[j] -= X[j][i] * coeffs[i];
            }
        }

        return coeffs;
    }

    double poly_eval(const std::vector<double>& coeffs, int x) {
        return coeffs[0] * std::pow(x, 3) + coeffs[1] * std::pow(x, 2) + coeffs[2] * x + coeffs[3];
    }

};
