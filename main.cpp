#include "chess_board_recog.h"
#include <stdexcept>


const int bilat_filter_size = 7; 
const int lines_amount = 200; // amount of best lines selected from hough lines 
const int dbscan_eps_intersection_clustering = 10;


int main() {
    std:: cout << "main() works\n";
    cv::String image_path = "0046.png";
    cv::Mat image = cv::imread(image_path);
    if(!image.data) {
        throw std::logic_error("image is not read correctly");
        return -1;
    }
    cv::Mat image_gray, gray_filtered, canny_edges;
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    bilateralFilter(image_gray, gray_filtered, bilat_filter_size, 75, 75);
    Utils::cannyPF(gray_filtered, canny_edges, 0.25);
    std::vector<cv::Vec2f> hough_lines;
    cv::HoughLines(canny_edges, hough_lines, 1, CV_PI / 720.0, 50, 0, 0);
    int num_lines = std::min(static_cast<int>(hough_lines.size()), lines_amount);
    std::vector<cv::Vec2d> lines_best(hough_lines.begin(), hough_lines.begin() + num_lines);
    ChessBoardRecognition::get_all_line_intersections(lines_best, image.size);
    std::vector<std::set<size_t>> intersecting_clusters = ChessBoardRecognition::get_intersecting_line_clusters();

    std::vector<std::set<size_t>> parallel_clusters = ChessBoardRecognition::get_parallel_line_clusters(lines_best);
    
    std::pair<std::set<size_t>, std::set<size_t>> best_cluster_pair = ChessBoardRecognition::select_best_performing_cluster_pair(
                                                            lines_best, intersecting_clusters, parallel_clusters);
    std::pair<cv::Vec2d, cv::Vec2d> cluster_means = std::make_pair(Utils::cluster_mean_hessfixed(
                                                                        lines_best, best_cluster_pair.first), 
                                                                    Utils::cluster_mean_hessfixed(
                                                                        lines_best, best_cluster_pair.second));

    std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> best_cluster_pair_duplicate_eliminated = std::make_pair(
                            ChessBoardRecognition::cluster_eliminate_duplicate(
                                                lines_best, best_cluster_pair.first, cluster_means.second, image.size),
                            ChessBoardRecognition::cluster_eliminate_duplicate(
                                                lines_best, best_cluster_pair.second, cluster_means.first, image.size)
    );

    std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> best_cluster_pair_chessboard = 
                        ChessBoardRecognition::cluster_eliminate_non_chessboard(best_cluster_pair_duplicate_eliminated,
                                                                                cluster_means,
                                                                                image.size);

    std::vector<std::vector<std::pair<double, double>>> all_corners_in_chessboard = 
                                        ChessBoardRecognition::get_intersections_between_clusters(
                                                        std::get<0>(best_cluster_pair_chessboard), 
                                                        std::get<1>(best_cluster_pair_chessboard), image.size);
    std::cout << "All corners in chessboard:\n";
    int count = 1;
    int vec_num = 1;
    for(auto& vec_inner: all_corners_in_chessboard) {
        std::cout << "vec # " << vec_num << '\n';
        for(auto& pair_d: vec_inner) {
            std::cout << '\t' << count << ": " << pair_d.first << '\t' << pair_d.second << '\n';
            count++;
        }
        count = 1;
        vec_num++;
    }

    return 0;

}
