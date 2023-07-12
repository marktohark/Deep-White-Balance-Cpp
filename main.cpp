#include "Awb.h"
#include <filesystem>
#include <fstream>

int main() {
    auto projectPath = std::filesystem::current_path().parent_path();
    auto modelPath = projectPath / "models" / "awb.onnx";
    auto exPath = projectPath / "example_images";
    auto resPath = projectPath / "results";

    Awb m(modelPath.string());

    for(const auto& entry : std::filesystem::directory_iterator(exPath)) {
        auto imgPath = entry.path().string();
        auto fileName = entry.path().filename();
        std::cout << "process " << fileName << std::endl;
        auto img = cv::imread(imgPath);
        auto balancedImg = m.predict(img).clone();
        cv::imwrite((resPath / fileName).string(), balancedImg, {cv::IMWRITE_JPEG_QUALITY, 85});
        std::cout << fileName << " done" << std::endl;
    }
    return 0;
}
