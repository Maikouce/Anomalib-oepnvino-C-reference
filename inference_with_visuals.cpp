#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <iomanip>

namespace fs = std::filesystem;

int main() {
    // 模型加载和编译
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("C:\\Users\\Administrator\\Desktop\\OPENVINO_MODEL\\model.xml");
    const int INPUT_WIDTH = 256;
    const int INPUT_HEIGHT = 256;
    ov::Shape input_shape = { 1, 3, INPUT_HEIGHT, INPUT_WIDTH };
    model->reshape({ {model->input().get_any_name(), input_shape} });
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");//GPU Intel核显更快
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 图片路径
    std::string image_folder = "C:\\Users\\Administrator\\Desktop\\OPENVINO_MODEL\\TEST_IMAGE";
    std::string output_folder = "C:\\Users\\Administrator\\Desktop\\OPENVINO_MODEL\\OUTPUT"; // 输出文件夹

    // 创建输出文件夹（如果不存在）
    fs::create_directories(output_folder);


    std::vector<std::string> image_paths;

    // 读取文件夹中的所有图片
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    if (image_paths.empty()) {
        std::cerr << "Error: No images found in the specified folder." << std::endl;
        return -1;
    }

    // 归一化参数
    const float image_threshold = 58.635990142822266;
    const float pixel_threshold = 58.635990142822266;
    const float min_val = 1.3460818529129028;
    const float max_val = 186.42552185058594;

    // 循环处理图片
    std::vector<double> inference_times;
    std::vector<double> total_times;
    std::vector<float> pixel_scores;

    for (const auto& image_path : image_paths) {
        auto start_total = std::chrono::high_resolution_clock::now();

        // 加载图像
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read the image file: " << image_path << std::endl;
            continue;
        }

        // 预处理
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        cv::Mat blob;
        cv::dnn::blobFromImage(resized_image, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
        ov::Tensor input_tensor(ov::element::f32, input_shape, blob.data);
        infer_request.set_input_tensor(input_tensor);

        // 推理
        auto start_infer = std::chrono::high_resolution_clock::now();
        infer_request.infer();
        auto end_infer = std::chrono::high_resolution_clock::now();
        auto duration_infer = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer);

        // 获取输出
        auto output_tensor = infer_request.get_output_tensor(0);


        const ov::Shape& output_shape = output_tensor.get_shape();
        const int out_h = output_shape[2];
        const int out_w = output_shape[3];

        // 异常图处理
        cv::Mat anomaly_map(out_h, out_w, CV_32FC1, output_tensor.data<float>());
        cv::Mat normalized_map;
        anomaly_map.convertTo(normalized_map, CV_32F);
        normalized_map = (normalized_map - min_val) / (max_val - min_val) * 255;
        normalized_map.convertTo(normalized_map, CV_8UC1);
        cv::Mat heatmap;
        cv::applyColorMap(normalized_map, heatmap, cv::COLORMAP_JET);
        cv::resize(heatmap, heatmap, image.size());
        cv::Mat mask;
        cv::threshold(anomaly_map, mask, pixel_threshold, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8UC1);
        cv::resize(mask, mask, image.size());

        // 叠加和可视化
        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);
        cv::Mat overlay;
        cv::addWeighted(rgb_image, 0.6, heatmap, 0.4, 0, overlay);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 2);

        double minVal, maxVal;
        cv::minMaxLoc(anomaly_map, &minVal, &maxVal);
        float pixel_score = static_cast<float>(maxVal);
        pixel_scores.push_back(pixel_score);
        std::string status = (pixel_score > pixel_threshold) ? "Abnomaly" : "Normal";
        cv::putText(overlay, "Status: " + status + ", Score: " + std::to_string(pixel_score), cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        // --- 保存结果 ---
        std::string filename = fs::path(image_path).filename().string();
        size_t last_dot = filename.find_last_of(".");
        if (last_dot != std::string::npos) {
            filename = filename.substr(0, last_dot);
        }

        // 构建输出文件路径
        fs::path output_path = fs::path(output_folder);
        output_path /= (filename + "_result.png");
        std::string output_path_str = output_path.string(); // Convert to std::string
        cv::imwrite(output_path_str, overlay);


        double inference_time = duration_infer.count() / 1000.0;
        inference_times.push_back(inference_time);

        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
        double total_time = duration_total.count() / 1000.0;
        total_times.push_back(total_time);

        std::cout << "Processed: " << filename << ", Status: " << status
            << ", Score: " << std::fixed << std::setprecision(2) << pixel_score
            << ", Time: " << total_time << " ms" << std::endl;
    }


    // 计算平均时间
    double total_inference_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
    double average_inference_time = total_inference_time / inference_times.size();

    double total_processing_time = std::accumulate(total_times.begin(), total_times.end(), 0.0);
    double average_processing_time = total_processing_time / total_times.size();

    std::cout << "\nAverage Inference Time: " << average_inference_time << " ms" << std::endl;
    std::cout << "Average Total Processing Time: " << average_processing_time << " ms" << std::endl;
}
