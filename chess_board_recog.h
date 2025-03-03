#include <iostream>
#include <string>
#include <algorithm>
#include "utils.cpp"
#include <cmath>  // For std::abs, std::pow, std::sqrt
#include <tuple>
#include <iterator>
#include <functional>
#include <cassert>
#include <limits>
#include <map>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <ensmallen.hpp>
#include <armadillo>  // For Armadillo library
using namespace mlpack;
using namespace ens;
using namespace arma;
using namespace Utils;
using namespace cv;

const double parallel_angle_threshold = 0.04; // merge two parallel line clusters if angle difference is < {} (in radians)
const double two_line_cluster_threshold = 1.0; // angle difference between two line clusters of chess table should be < {} (in radians)
const double dbscan_eps_intersection_clustering = 10.0;
const double dbscan_eps_duplicate_elimination = 3.0;
const int min_points_intersecting = 8;
const int min_points_eliminate = 1;
const int polynomial_degree = 3; //  used for fitting polynomial to intersection data, last step of the pipeline

class ChessBoardRecognition {

    cv::Mat img;
    ChessBoardRecognition();
    ~ChessBoardRecognition();

public:
    static std::vector<std::pair<size_t, size_t>> intersections_info; // coordinates of intersections
    static std::vector<std::set<size_t>> parallel_sets_list; // sets of parallel lines
    static std::vector<std::pair<double, double>> intersections;

    static void get_chess_board_intersections(const cv::Mat& image);
    
    static void get_all_line_intersections(const std::vector<cv::Vec2d>& lines, const cv::Size& size);

    static std::vector<std::set<size_t>> get_intersecting_line_clusters();

    static std::vector<std::set<size_t>> get_parallel_line_clusters(const std::vector<cv::Vec2d>& lines_best);

    static std::pair<std::set<size_t>, std::set<size_t>> select_best_performing_cluster_pair(
                                                        const std::vector<cv::Vec2d>& lines_best, 
                                                        const std::vector<std::set<size_t>>& intersecting_clusters, 
                                                        const std::vector<std::set<size_t>>& parallel_clusters);

    static std::vector<cv::Vec2d> cluster_eliminate_duplicate(const std::vector<cv::Vec2d>& lines_best, 
                                                        const std::set<size_t>& cluster, 
                                                        const cv::Vec2d& intersect_line, 
                                                        const cv::Size& img_shape);

    static std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> cluster_eliminate_non_chessboard(
                                        const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> &merged_clusters_pair, 
                                        const std::pair<cv::Vec2d, cv::Vec2d> &cluster_means, 
                                        const cv::Size &img_shape);

    static std::vector<size_t> select_nine_fittable_intersections(
                                        const std::vector<std::pair<double, double>> &intersections);               

    static std::vector<std::vector<std::pair<double, double>>> get_intersections_between_clusters(
                                        const std::vector<cv::Vec2d>& merged_cluster1,
                                        const std::vector<cv::Vec2d>&  merged_cluster2,
                                        const cv::Size &img_shape);
}
