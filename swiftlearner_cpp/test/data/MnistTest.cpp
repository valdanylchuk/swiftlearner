#include "data/Mnist.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(MnistTest, DatasetSize) {
    auto [trainingSet, testSet] = Mnist::loadData(0L, 1000);
    ASSERT_EQ(trainingSet.size(), 1000); // Limited to 1000
    ASSERT_EQ(testSet.size(), Mnist::TestSetSize); // Should be TestSetSize (10000)
}

TEST(MnistTest, ImageSize) {
    auto [trainingSet, testSet] = Mnist::loadData(0L, 1);
    const auto& image = trainingSet[0].second;
    ASSERT_EQ(image.size(), Mnist::ImageSize);
}