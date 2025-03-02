#include "nn/BackpropClassifier.hpp"
#include "data/FisherIris.hpp"
#include "data/Mnist.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <chrono>

TEST(BackpropClassifierTest, ClassifyFisherIris) {
    // Load the Fisher Iris data
    auto [trainingSet, testSet] = FisherIris::loadData();

    // Normalize the input values to speed up learning
    auto normalize = [](float x) { return (x - 25.0f) / 25.0f; };

    auto start = std::chrono::high_resolution_clock::now();

    // Create the BackpropClassifier
    BackpropClassifier classifier(trainingSet, 3, 5000, 1.0f, normalize);

    // Calculate the accuracy
    double correctPredictions = 0;
    for (const auto& [species, params] : testSet) {
        if (classifier.predict(params) == species) {
            correctPredictions++;
        }
    }
    double accuracy = static_cast<double>(correctPredictions) / testSet.size();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    std::cout << "Fisher Iris backprop time: " << duration << "s" << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;

    // Assert that the accuracy is above the expected threshold
    ASSERT_GT(accuracy, 0.8); // 0.96 is typical
}

TEST(BackpropClassifierTest, ClassifyMnist) {
    int nHidden = 70;
    int nRepeat = 1;  // Increase for better results
    float learnRate = 1.0f;
    double expectedAccuracy = 0.87;  // 0.95 with nRepeat=20
    std::optional<long> seed = 0L;

    // Load the MNIST data
    auto [trainingSet, testSet] = Mnist::loadData(seed);

    // Careful normalization is essential with this simple network.
    auto scale = [](float x) { return x / 255.0f; }; // source byte range to (0; 1)

    //Calculate mean
    double mean = 0.0;
    int numSamples = 0;
    for (const auto& [digit, image] : trainingSet) {
      if (numSamples >= 10000) break; // Limit to 10000 *pixels*
        for (float pixelValue : image) {
            mean += scale(pixelValue);
            numSamples++;
            if (numSamples >= 10000) break;
        }
    }
    mean /= 10000.0;
    std::cout << "mean: " << mean << std::endl;

    auto normalize = [scale, mean](float x) { return scale(x) - mean; }; // balance around 0 for stability

    std::cout << "creating the classifier" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    BackpropClassifier classifier(trainingSet, nHidden, nRepeat, learnRate, normalize, seed);

    std::cout << "checking the accuracy" << std::endl;
    double correctPredictions = 0;
    for (const auto& [digit, params] : testSet) {
        if (classifier.predict(params) == digit) {
            correctPredictions++;
        }
    }
    double accuracy = static_cast<double>(correctPredictions) / testSet.size();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    std::cout << "MNIST backprop time: " << duration << "s" << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;

    ASSERT_GT(accuracy, expectedAccuracy);
}