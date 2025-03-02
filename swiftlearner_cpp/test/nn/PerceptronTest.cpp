#include "gtest/gtest.h"
#include "nn/Perceptron.hpp"

#include <vector>

TEST(PerceptronTest, ActivationFunction) {
    Perceptron p({ 2.0, 3.0 }, -25.0);
    ASSERT_FALSE(p.activate({ 4.0, 5.0 }));
    ASSERT_TRUE(p.activate({ 4.0, 6.0 }));
}

TEST(PerceptronTest, LearnAndFunction) {
    std::vector<std::pair<vec, bool>> and_examples = {
        {{0, 0}, false},
        {{0, 1}, false},
        {{1, 0}, false},
        {{1, 1}, true}
    };

    Perceptron p(2);
    auto trained = p.train(and_examples, 10);
    for (const auto& [input, target] : and_examples)
        EXPECT_EQ(trained.activate(input), target);
}

TEST(PerceptronTest, LearnOrFunction) {
    std::vector<std::pair<vec, bool>> or_examples = {
        {{0, 0}, false},
        {{0, 1}, true},
        {{1, 0}, true},
        {{1, 1}, true}
    };

    Perceptron p(2);
    auto trained = p.train(or_examples, 10);
    for (const auto& [input, target] : or_examples)
        EXPECT_EQ(trained.activate(input), target);
}