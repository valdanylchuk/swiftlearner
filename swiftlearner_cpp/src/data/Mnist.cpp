#include "data/Mnist.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <zlib.h>
#include <sstream>
#include <cstdint>

namespace Mnist {

std::vector<int> readLabels(gzFile file, int setSize) {
    std::vector<int> labels(setSize);

    // Skip the magic number and number of items (4 bytes each)
    gzseek(file, 8, SEEK_SET);

    for (int i = 0; i < setSize; ++i) {
        unsigned char label;
        if (gzread(file, &label, 1) != 1) {
            throw std::runtime_error("Error reading label data or EOF reached prematurely.");
        }
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

std::vector<vec> readImages(gzFile file, int setSize) {
    std::vector<vec> images(setSize, vec(ImageSize));
    
    // Skip the magic number, number of images, rows, and columns (4 bytes each)
    gzseek(file, 16, SEEK_SET);
    
    for (int i = 0; i < setSize; ++i) {
        for (int j = 0; j < ImageSize; ++j) {
            unsigned char pixel;
            if (gzread(file, &pixel, 1) != 1) {
                throw std::runtime_error("Error reading image data or EOF reached prematurely.");
            }
            images[i][j] = static_cast<float>(pixel);
        }
    }
    
    return images;
}

std::pair<std::vector<std::pair<int, vec>>, std::vector<std::pair<int, vec>>>
loadData(std::optional<long> randomSeed, int nTrainPoints,
         const std::string& trainImagesFile, const std::string& trainLabelsFile,
         const std::string& testImagesFile, const std::string& testLabelsFile) {

    gzFile trainImagesGz = gzopen(trainImagesFile.c_str(), "rb");
    if (!trainImagesGz) {
        throw std::runtime_error("Could not open file: " + trainImagesFile);
    }
    gzFile trainLabelsGz = gzopen(trainLabelsFile.c_str(), "rb");
    if (!trainLabelsGz) {
        gzclose(trainImagesGz);
        throw std::runtime_error("Could not open file: " + trainLabelsFile);
    }
    gzFile testImagesGz = gzopen(testImagesFile.c_str(), "rb");
    if (!testImagesGz) {
        gzclose(trainImagesGz);
        gzclose(trainLabelsGz);
        throw std::runtime_error("Could not open file: " + testImagesFile);
    }
    gzFile testLabelsGz = gzopen(testLabelsFile.c_str(), "rb");
    if (!testLabelsGz) {
        gzclose(trainImagesGz);
        gzclose(trainLabelsGz);
        gzclose(testImagesGz);
        throw std::runtime_error("Could not open file: " + testLabelsFile);
    }

    std::vector<std::pair<int, vec>> trainingSet;
    std::vector<std::pair<int, vec>> testSet;

    std::vector<int> trainLabels = readLabels(trainLabelsGz, nTrainPoints);
    std::vector<int> testLabels = readLabels(testLabelsGz, TestSetSize);
    std::vector<vec> trainImages = readImages(trainImagesGz, nTrainPoints);
    std::vector<vec> testImages = readImages(testImagesGz, TestSetSize);

    // Create train/test sets
    for (size_t i = 0; i < nTrainPoints; ++i) {
        trainingSet.push_back({trainLabels[i], trainImages[i]});
    }
    for (size_t i = 0; i < TestSetSize; ++i) {
        testSet.push_back({testLabels[i], testImages[i]});
    }

    // Shuffle the training data
    std::mt19937 generator(randomSeed.value_or(std::random_device{}()));
    std::shuffle(trainingSet.begin(), trainingSet.end(), generator);

    gzclose(trainImagesGz);
    gzclose(trainLabelsGz);
    gzclose(testImagesGz);
    gzclose(testLabelsGz);

    return {trainingSet, testSet};
}

} // namespace Mnist