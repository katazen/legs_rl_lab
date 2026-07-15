#include <iostream>
#include <deque>
#include <algorithm>
#include <numeric>

// 滑动窗口类
class SlidingWindow {
public:
    SlidingWindow(size_t window_size) : window_size_(window_size) {}

    // 添加新数据点
    void addData(double value) {
        // 如果窗口已满，移除最旧的数据
        if (data_.size() >= window_size_) {
            data_.pop_front();
        }
        // 添加新数据点
        data_.push_back(value);
    }

    // 获取当前窗口数据
    std::deque<double> getWindow() const {
        return data_;
    }

private:
    size_t window_size_;          // 滑动窗口大小
    std::deque<double> data_;     // 存储窗口数据
};

// 中值滤波器类
class MedianFilter {
public:
    MedianFilter(size_t window_size) : window_(window_size) {}

    double applyFilter(double value) {
        // 更新滑动窗口
        window_.addData(value);

        // 获取窗口数据
        std::deque<double> data = window_.getWindow();

        // 计算中值
        std::vector<double> sorted_data(data.begin(), data.end());
        std::sort(sorted_data.begin(), sorted_data.end());
        return sorted_data[sorted_data.size() / 2];
    }

private:
    SlidingWindow window_;  // 滑动窗口
};

// 均值滤波器类
class MeanFilter {
public:
    MeanFilter(size_t window_size) : window_(window_size) {}

    double applyFilter(double value) {
        // 更新滑动窗口
        window_.addData(value);

        // 获取窗口数据
        std::deque<double> data = window_.getWindow();

        // 计算均值
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / data.size();
    }

private:
    SlidingWindow window_;  // 滑动窗口
};

// 组合滤波器类
class CombinedFilter {
public:
    CombinedFilter(int median_window_size, int mean_window_size)
        : median_filter_(median_window_size), mean_filter_(mean_window_size) {}

    double applyFilter(double value) {
        // 首先应用中值滤波
        double median_result = median_filter_.applyFilter(value);

        // 然后应用均值滤波
        return mean_filter_.applyFilter(median_result);
    }

private:
    MedianFilter median_filter_;  // 中值滤波器
    MeanFilter mean_filter_;      // 均值滤波器
};

// int main() {
//     // 滤波器窗口大小
//     size_t median_window_size = 5;
//     size_t mean_window_size = 3;

//     // 创建组合滤波器
//     CombinedFilter filter(median_window_size, mean_window_size);

//     // 模拟实时输入数据
//     std::vector<double> input_data = {1, 2, 3, 100, 3, 2, 1, 4, 5, 6, 7, 8, 9};
//     std::cout << "原始数据: ";
//     for (double d : input_data) {
//         std::cout << d << " ";
//     }
//     std::cout << "\n";

//     std::cout << "滤波后数据: ";
//     for (double d : input_data) {
//         double filtered_value = filter.applyFilter(d);
//         std::cout << filtered_value << " ";
//     }
//     std::cout << "\n";

//     return 0;
// }