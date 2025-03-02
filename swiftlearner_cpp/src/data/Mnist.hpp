#pragma once
#include <vector>
#include <utility>
#include <optional>
#include <string>
#include <zlib.h>

using vec = std::vector<float>;

namespace Mnist {

constexpr int ImageWidth = 28;
constexpr int ImageHeight = 28;
constexpr int ImageSize = ImageWidth * ImageHeight;
constexpr int TrainSetSize = 60000;
constexpr int TestSetSize = 10000;

std::pair<std::vector<std::pair<int, vec>>, std::vector<std::pair<int, vec>>>
loadData(std::optional<long> randomSeed,
         int nTrainPoints = TrainSetSize,
         const std::string& trainImagesFile = "resources/mnist/train-images-idx3-ubyte.gz",
         const std::string& trainLabelsFile = "resources/mnist/train-labels-idx1-ubyte.gz",
         const std::string& testImagesFile = "resources/mnist/t10k-images-idx3-ubyte.gz",
         const std::string& testLabelsFile = "resources/mnist/t10k-labels-idx1-ubyte.gz");

std::vector<int> readLabels(gzFile file, int setSize);
std::vector<vec> readImages(gzFile file, int setSize);

} // namespace Mnist