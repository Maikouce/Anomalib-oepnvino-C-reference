#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <numeric> // for std::accumulate

namespace fs = std::filesystem;

int main() {
    // 模型加载和编译 (在循环外进行)
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("C:\\Users\\Y\\Desktop\\OPENVINO_MODEL\\model.xml");

    // 固定输入形状 (例如，224x224)
    const int INPUT_WIDTH = 256;
    const int INPUT_HEIGHT = 256;
    ov::Shape input_shape = { 1, 3, INPUT_HEIGHT, INPUT_WIDTH };
    model->reshape({ {model->input().get_any_name(), input_shape} });

    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 图片路径
    std::string image_folder = "C:\\Users\\Y\\Desktop\\OPENVINO_MODEL\\TEST_IMAGE";
    std::vector<std::string> image_paths;

    // 读取文件夹中的所有图片
    auto start_find = std::chrono::high_resolution_clock::now();
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower); // 转换为小写

            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                image_paths.push_back(entry.path().string());
            }
        }
    }
    auto end_find = std::chrono::high_resolution_clock::now();
    auto duration_find = std::chrono::duration_cast<std::chrono::microseconds>(end_find - start_find);
    double find_time = duration_find.count() / 1000.0;
    std::cout << "图片读取时间: " << find_time << " ms" << std::endl;


    if (image_paths.empty()) {
        std::cerr << "Error: No images found in the specified folder." << std::endl;
        return -1;
    }

    // 循环处理图片
    std::vector<double> inference_times;
    std::vector<double> total_times; // 添加total_times
    std::vector<float> pixel_scores;

    for (const auto& image_path : image_paths) {
        // 时间戳：开始处理单张图片
        auto start_total = std::chrono::high_resolution_clock::now();

        // 加载图像
        auto start_load = std::chrono::high_resolution_clock::now();
        cv::Mat image = cv::imread(image_path);
        auto end_load = std::chrono::high_resolution_clock::now();
        auto duration_load = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load);
        double load_time = duration_load.count() / 1000.0;


        if (image.empty()) {
            std::cerr << "Error: Could not read the image file: " << image_path << std::endl;
            continue; // Skip to the next image
        }

        // 预处理
        auto start_preprocess = std::chrono::high_resolution_clock::now();
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        // 调整图像大小
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

        // 转换为CHW格式并归一化

        cv::Mat blob;
        cv::dnn::blobFromImage(resized_image, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

        // 创建输入张量
        ov::Tensor input_tensor(ov::element::f32, input_shape, blob.data); // 直接使用 blob.data
        infer_request.set_input_tensor(input_tensor);
        auto end_preprocess = std::chrono::high_resolution_clock::now();
        auto duration_preprocess = std::chrono::duration_cast<std::chrono::microseconds>(end_preprocess - start_preprocess);
        double preprocess_time = duration_preprocess.count() / 1000.0;


        auto start_infer = std::chrono::high_resolution_clock::now();

        // 执行推理
        infer_request.infer();

        auto end_infer = std::chrono::high_resolution_clock::now();
        // 计算时间差
        auto duration_infer = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer);

        // 获取输出
        auto output_tensor = infer_request.get_output_tensor(0);
        const ov::Shape& output_shape = output_tensor.get_shape();
        const int out_h = output_shape[2];
        const int out_w = output_shape[3];

        auto start_postprocess = std::chrono::high_resolution_clock::now();
        // 处理异常图
        cv::Mat anomaly_map(out_h, out_w, CV_32FC1, output_tensor.data<float>());

        // 计算最大值，获取pixel_score
        double minVal, maxVal;
        cv::minMaxLoc(anomaly_map, &minVal, &maxVal);
        float pixel_score = static_cast<float>(maxVal); // 转换为float
        pixel_scores.push_back(pixel_score);
        auto end_postprocess = std::chrono::high_resolution_clock::now();
        auto duration_postprocess = std::chrono::duration_cast<std::chrono::microseconds>(end_postprocess - start_postprocess);
        double postprocess_time = duration_postprocess.count() / 1000.0;



        // 输出运行时间
        double inference_time = duration_infer.count() / 1000.0; // 毫秒
        std::cout << "Inference Time: " << inference_time << " ms, Pixel Score: " << pixel_score << std::endl;
        inference_times.push_back(inference_time);

        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
        double total_time = duration_total.count() / 1000.0;
        total_times.push_back(total_time); // 保存total_time

        std::cout << "Image Path: " << image_path << std::endl;
        std::cout << "  Load Time: " << load_time << " ms" << std::endl;
        std::cout << "  Preprocess Time: " << preprocess_time << " ms" << std::endl;
        std::cout << "  Inference Time: " << inference_time << " ms" << std::endl;
        std::cout << "  Postprocess Time: " << postprocess_time << " ms" << std::endl;
        std::cout << "  Total Time: " << total_time << " ms" << std::endl;
        std::cout << "------------------------------------" << std::endl;
    }

    // 计算平均时间
    double total_inference_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
    double average_inference_time = total_inference_time / inference_times.size();

    double total_processing_time = std::accumulate(total_times.begin(), total_times.end(), 0.0);
    double average_processing_time = total_processing_time / total_times.size();

    std::cout << "Average Inference Time: " << average_inference_time << " ms" << std::endl;
    std::cout << "Average Total Processing Time: " << average_processing_time << " ms" << std::endl; // 输出平均总时间

    return 0;
}
