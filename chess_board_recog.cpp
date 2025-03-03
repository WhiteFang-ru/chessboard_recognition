#include "chess_board_recog.h"


void ChessBoardRecognition::get_all_line_intersections(const std::vector<cv::Vec2d>& lines, const cv::Size& size)
{
    bool set_exists = true;
    for (size_t i = 0; i < lines.size(); ++i) {
        for (size_t j = i + 1; j < lines.size(); ++j) {
            std::pair<double, double> single_intersection = Utils::intersection(lines[i], lines[j], size);
            if((std::get<0>(single_intersection) == -1) && (std::get<1>(single_intersection) == -1)) {
                bool set_exists = false;
                for (auto& next_set : parallel_sets_list) {
                    if (next_set.find(i) != next_set.end() || next_set.find(j) != next_set.end()) {  
                        set_exists = true;
                        next_set.insert(i);
                        next_set.insert(j);
                        break;
                    }
                }
                if (!set_exists) { 
                    parallel_sets_list.push_back(std::set{i, j});
                }
            } else {
                if (single_intersection.first > 0 && single_intersection.second < size.height && 
                    single_intersection.first > 0 && single_intersection.second < size.width) {
                    intersections.push_back(single_intersection);
                    intersections_info.push_back(std::make_pair(i, j));
                }
            }
        }
    }

    std::sort(parallel_sets_list.begin(), parallel_sets_list.end(),
              [](const std::set<size_t>& a, const std::set<size_t>& b) {
                  return a.size() > b.size();
              });

}



std::vector<std::set<size_t>> ChessBoardRecognition::get_intersecting_line_clusters() {

    arma::mat data(2, intersections.size());
    for (size_t i = 0; i < intersections.size(); ++i) {
        data(0, i) = intersections[i].first;  // x-coordinate
        data(1, i) = intersections[i].second; // y-coordinate
    }
    
    DBSCAN<> dbscan(dbscan_eps_intersection_clustering, min_points_intersecting); // or type mlpack::dbscan::DBSCAN<>
    arma::Row<size_t> labels_intersections;
    dbscan.Cluster(data, labels_intersections);

    std::vector<std::vector<std::pair<size_t, size_t>>> intersection_clusters;

    size_t max_cluster_id = arma::max(labels_intersections);
    for (size_t cluster_id = 0; cluster_id <= max_cluster_id; ++cluster_id) {
        std::vector<std::pair<size_t, size_t>> cluster;

        for (size_t i = 0; i < labels_intersections.n_elem; ++i) {
            if (labels_intersections(i) == cluster_id) {
                cluster.push_back(intersections_info[i]);
            }
        }
        intersection_clusters.push_back(cluster);
    }

    std::vector<std::set<size_t>> unique_lines_each_cluster;

    for (const auto& cluster : intersection_clusters) {
        std::set<size_t> unique_lines;
        for (const auto& line : cluster) {
            unique_lines.insert(line.first);
            unique_lines.insert(line.second);
        }
        unique_lines_each_cluster.push_back(unique_lines);
    }

    std::sort(unique_lines_each_cluster.begin(), unique_lines_each_cluster.end(),
              [](const std::set<size_t>& a, const std::set<size_t>& b) {
                  return a.size() > b.size();
              });

    return unique_lines_each_cluster;
}



std::vector<std::set<size_t>> ChessBoardRecognition::get_parallel_line_clusters(const std::vector<cv::Vec2d>& lines_best) {
    std::vector<std::set<size_t>> cur_sets = parallel_sets_list;
    std::vector<double> cur_means;

    for (const auto& next_set : cur_sets) {
        double mean = 0.0;
        int count = 0;
        for (const auto& index : next_set) {
            mean += lines_best[index][1];
            count++;
        }
        mean /= count;
        cur_means.push_back(mean);
    }

    int i = 0;
    while (i < (cur_sets.size() - 1)) {
        for (int j = i + 1; j < cur_sets.size(); ++j) {
            if (std::abs(cur_means[i] - cur_means[j]) < parallel_angle_threshold) {
                cur_sets[i].insert(cur_sets[j].begin(), cur_sets[j].end());
                double new_mean = 0.0;
                int count = 0;
                for (const auto& index : cur_sets[i]) {
                    new_mean += lines_best[index][1];
                    count++;
                }
                new_mean /= count;
                cur_means[i] = new_mean;

                cur_sets.erase(cur_sets.begin() + j);
                cur_means.erase(cur_means.begin() + j);

                i = 0;
                break;
            }
        }
        i++;
    }

    std::sort(cur_sets.begin(), cur_sets.end(), [](const std::set<int>& a, const std::set<int>& b) {
        return a.size() > b.size();
    });

   return cur_sets;
}




std::pair<std::set<size_t>, std::set<size_t>> ChessBoardRecognition::select_best_performing_cluster_pair(
                                                const std::vector<cv::Vec2d>& lines_best,
                                                const std::vector<std::set<size_t>>& intersecting_clusters, 
                                                const std::vector<std::set<size_t>>& parallel_clusters)
{
    std::vector<std::set<size_t>> merged_clusters = intersecting_clusters;

    for (const auto& cluster : parallel_clusters) {
        merged_clusters.push_back(cluster);
    }

    std::vector<size_t> merged_sizes;
    for (const auto& cluster : merged_clusters) {
        merged_sizes.push_back(cluster.size());
    }

    std::vector<std::pair<size_t, size_t>> pass_list;
    for (size_t i = 0; i < merged_clusters.size(); ++i) {
        for (size_t j = i + 1; j < merged_clusters.size(); ++j) {
            double angle_d = Utils::angle_diff(Utils::cluster_mean_hessfixed(lines_best, merged_clusters[i]), 
                                                Utils::cluster_mean_hessfixed(lines_best, merged_clusters[j]));
            if(angle_d > two_line_cluster_threshold) {
                pass_list.push_back(std::make_pair(i, j));
            }
        }
    }

    std::sort(pass_list.begin(), pass_list.end(), [&merged_sizes](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
        return merged_sizes[a.first] * merged_sizes[a.second] > merged_sizes[b.first] * merged_sizes[b.second];
    });

    std::pair<size_t, size_t> winner_pair = pass_list.front();

    return std::make_pair(merged_clusters[winner_pair.first], merged_clusters[winner_pair.second]);
}




std::vector<cv::Vec2d> ChessBoardRecognition::cluster_eliminate_duplicate(
                                        const std::vector<cv::Vec2d> &lines_best, 
                                        const std::set<size_t> &cluster, 
                                        const cv::Vec2d &intersect_line, 
                                        const cv::Size &img_shape)
{
    std::vector<cv::Vec2d> cluster_lines;
    for (const auto& index : cluster) {
        cluster_lines.push_back(lines_best[index]);
    }

    std::vector<std::pair<double, double>> intersection_points;
    intersection_points.reserve(cluster_lines.size());

    auto intersection_fn = [&intersect_line, &img_shape](const cv::Vec2d& x) {
        return Utils::intersection(x, intersect_line, img_shape);
    };

    std::transform(cluster_lines.begin(), cluster_lines.end(), std::back_inserter(intersection_points), intersection_fn);

    arma::mat data(intersection_points.size(), 2);
    for (size_t i = 0; i < intersection_points.size(); ++i) {
        data(i, 0) = intersection_points[i].first;
        data(i, 1) = intersection_points[i].second;
    }

    DBSCAN<> dbscan(dbscan_eps_duplicate_elimination, min_points_eliminate); // or type mlpack::methods::dbscan::DBSCAN<>
    arma::Row<size_t> labels;
    dbscan.Cluster(data, labels);

    std::vector<cv::Vec2d> merged_cluster;
    size_t max_label = *std::max_element(labels.begin(), labels.end());
    
    for (size_t i = 0; i <= max_label; ++i) {
        std::set<size_t> cluster_indices;
        for (size_t j = 0; j < labels.n_elem; ++j){
            if (labels[j] == i)
                cluster_indices.insert(j);
        }

        cv::Vec2d cluster_mean = Utils::cluster_mean_hessfixed(cluster_lines, cluster_indices);
        merged_cluster.push_back(cluster_mean);
    }
}




std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> ChessBoardRecognition::cluster_eliminate_non_chessboard(
                                        const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec2d>> &merged_clusters_pair, 
                                        const std::pair<cv::Vec2d, cv::Vec2d> &cluster_means, 
                                        const cv::Size &img_shape) 
{    std::vector<cv::Vec2d> first_cluster = merged_clusters_pair.first;
    std::vector<cv::Vec2d> second_cluster = merged_clusters_pair.second;

    cv::Vec2d mean_first_cluster = cluster_means.first;
    cv::Vec2d mean_second_cluster = cluster_means.second;

    std::vector<std::pair<double, double>> intersections_first_cluster, intersection_second_cluster;

    for (const auto& x : first_cluster) {
        std::pair<double, double> intersection_result = Utils::intersection(x, mean_second_cluster, img_shape);
        intersections_first_cluster.push_back(intersection_result);
    }
    for(const auto& x: second_cluster) {
        std::pair<double, double> intersection_result = Utils::intersection(x, mean_first_cluster, img_shape);
        intersection_second_cluster.push_back(intersection_result);
    }


    std::vector<size_t> best_intersections_first_cluster = 
                        ChessBoardRecognition::select_nine_fittable_intersections(intersections_first_cluster);
    std::vector<size_t> best_intersections_second_cluster = 
                        ChessBoardRecognition::select_nine_fittable_intersections(intersection_second_cluster);

    std::vector<cv::Vec2d> selected_first_cluster, selected_second_cluster;

    for (size_t idx : best_intersections_first_cluster) {
        selected_first_cluster.push_back(first_cluster[idx]);
    }

    for (size_t idx : best_intersections_second_cluster) {
        selected_second_cluster.push_back(second_cluster[idx]);
    }

    return std::make_pair(selected_first_cluster, selected_second_cluster); 
}




std::vector<size_t> select_nine_fittable_intersections(const std::vector<std::pair<double, double>> &intersections) {
        double var0 = 0.0, var1 = 0.0;
        size_t n = intersections.size();

        double mean0 = 0.0, mean1 = 0.0;
        for (const auto &pair : intersections)
        {
            mean0 += pair.first;
            mean1 += pair.second;
        }
        mean0 /= n;
        mean1 /= n;

        for (const auto &pair : intersections)
        {
            var0 += std::pow(pair.first - mean0, 2);
            var1 += std::pow(pair.second - mean1, 2);
        }
        var0 /= n;
        var1 /= n;

        int metric_col = (var0 > var1) ? 0 : 1;

        std::vector<double> metric_values;
        for (const auto &pair : intersections) {
            metric_values.push_back((metric_col == 0) ? pair.first : pair.second);
        }

        std::vector<std::tuple<int, double>> indexed_metric_values;
        for (int i = 0; i < metric_values.size(); ++i) {
            indexed_metric_values.push_back({i, metric_values[i]});
        }

        std::sort(indexed_metric_values.begin(), indexed_metric_values.end(),
                  [](const std::tuple<int, double>& a, const std::tuple<int, double>& b) {
                      return std::get<1>(a) < std::get<1>(b);
                });

        for (int i = 0; i < metric_values.size(); ++i) {
            metric_values[i] = std::get<1>(indexed_metric_values[i]);
        }


        auto all_combinations_iter = Utils::generate_combinations<std::tuple<int, double>>(metric_values, 9); // vector<vector<tuple<int, double>>>

        std::vector<std::vector<std::tuple<int, double>>> all_combinations;
        for (const auto& combination : all_combinations_iter) {
            std::vector<std::tuple<int, double>> combo;
            for (size_t i = 0; i < combination.size(); ++i) {
                combo.push_back(std::make_tuple(i, combination[i]));
            }
            all_combinations.push_back(combo);
        }

        std::vector<double> x_vals = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; 

        std::vector<std::vector<double>> all_combinations_fitted_calculated;
        std::vector<double> all_combinations_mse;

        for (const auto& combination : all_combinations) {
            std::vector<double> y_vals;
            for (const auto& pair : combination) {
                y_vals.push_back(std::get<1>(pair));
            }

            std::vector<double> coeffs = Utils::polyfit(x_vals, y_vals);

            std::vector<double> fitted_y_vals;
            for (int x : x_vals) {
                fitted_y_vals.push_back(Utils::poly_eval(coeffs, x));
            }

            all_combinations_fitted_calculated.push_back(fitted_y_vals);

            double mse = 0;
            for (size_t i = 0; i < y_vals.size(); ++i) {
                mse += std::pow(y_vals[i] - fitted_y_vals[i], 2);
            }
            mse /= y_vals.size();  // Mean squared error
            all_combinations_mse.push_back(mse);
        }

        auto min_mse_iter = std::min_element(all_combinations_mse.begin(), all_combinations_mse.end());
        size_t best_combination_idx = std::distance(all_combinations_mse.begin(), min_mse_iter);
        const std::vector<std::tuple<int, double>>& best_combination = all_combinations[best_combination_idx];
    
        std::vector<int> best_combination_indexes;
        for (const auto& elem : best_combination) {
            best_combination_indexes.push_back(std::get<0>(elem));
        }

        std::vector<size_t> sorted_idx(all_combinations_mse.size());
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(), 
                [&all_combinations_mse](size_t i1, size_t i2) {
                        return all_combinations_mse[i1] < all_combinations_mse[i2];
                });

        std::map<size_t, size_t> sorted_idx_reverse_dict;
        for (size_t i = 0; i < sorted_idx.size(); ++i) {
            sorted_idx_reverse_dict[sorted_idx[i]] = i;
        }

        std::vector<size_t> best_combination_indexes_reversed;
        for (int idx : best_combination_indexes) {
            best_combination_indexes_reversed.push_back(sorted_idx_reverse_dict[idx]);
        }
    }



std::vector<std::vector<std::pair<double, double>>> ChessBoardRecognition::get_intersections_between_clusters(
                                            const std::vector<cv::Vec2d>& cluster1, 
                                            const std::vector<cv::Vec2d>& cluster2, 
                                            const cv::Size& img_shape){
        std::vector<std::vector<std::pair<double, double>>> intersections(cluster1.size(), 
                                                                      std::vector<std::pair<double, double>>(cluster2.size()));
        for (size_t i = 0; i < cluster1.size(); ++i) {
            for (size_t j = 0; j < cluster2.size(); ++j) {
                intersections[i][j] = Utils::intersection(cluster1[i], cluster2[j], img_shape);
            }
        }
        return intersections;
}
