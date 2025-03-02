#include "nn/BackpropNet.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(BackpropNetTest, ReproduceKnownExample) {
    Example ex = {{0.35f, 0.9f}, {0.5f}};

    std::vector<Node> hiddenLayer = {
        Node({0.1f, 0.8f}),
        Node({0.4f, 0.6f})
    };
    std::vector<Node> outputLayer = {
        Node({0.3f, 0.9f})
    };

    BackpropNet nn(std::move(hiddenLayer), std::move(outputLayer));
    BackpropNet& learned = nn.learn(ex, 1.0f);

    ASSERT_NEAR(learned.hiddenLayer[0].output, 0.6803f, 0.0001f);
    ASSERT_NEAR(learned.hiddenLayer[1].output, 0.6637f, 0.0001f);
    ASSERT_NEAR(learned.outputLayer[0].output, 0.6903f, 0.0001f);

    ASSERT_NEAR(learned.outputLayer[0].error, -0.0406f, 0.0001f);
    ASSERT_NEAR(learned.outputLayer[0].weights[0], 0.2723f, 0.0001f);
    ASSERT_NEAR(learned.outputLayer[0].weights[1], 0.8730f, 0.0001f);

    ASSERT_NEAR(learned.hiddenLayer[0].error, -0.0025f, 0.0002f);
    ASSERT_NEAR(learned.hiddenLayer[0].weights[0], 0.0991f, 0.0002f);
    ASSERT_NEAR(learned.hiddenLayer[0].weights[1], 0.7977f, 0.0002f);

    ASSERT_NEAR(learned.hiddenLayer[1].error, -0.008f, 0.0002f);
    ASSERT_NEAR(learned.hiddenLayer[1].weights[0], 0.3972f, 0.0002f);
    ASSERT_NEAR(learned.hiddenLayer[1].weights[1], 0.5927f, 0.0002f);

    float out = learned.testOut(ex.input);
    ASSERT_NEAR(out, 0.682f, 0.0001f);
}

TEST(BackpropNetTest, LearnOrFunction) {
    std::vector<Example> orExamples = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {1}}
    };

    BackpropNet nn = BackpropNet::randomNet(2, 2, 1, 10L);
    nn.train(orExamples, 1000);

    for (const auto& ex : orExamples) {
        float out = nn.testOut(ex.input);
        ASSERT_NEAR(out, ex.target[0], 0.1f);
    }
}

TEST(BackpropNetTest, LearnAndFunction) {
    std::vector<Example> andExamples = {
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}}
    };

    BackpropNet nn = BackpropNet::randomNet(2, 3, 1, 10L);
    nn.train(andExamples, 1000);

    for (const auto& ex : andExamples) {
        float out = nn.testOut(ex.input);
        ASSERT_NEAR(out, ex.target[0], 0.1f);
    }
}

TEST(BackpropNetTest, LearnXorFunction) {
    std::vector<Example> xorExamples = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    BackpropNet nn = BackpropNet::randomNet(2, 4, 1, 10L);
    nn.train(xorExamples, 6000);

    for (const auto& ex : xorExamples) {
        float out = nn.testOut(ex.input);
        ASSERT_NEAR(out, ex.target[0], 0.1f);
    }
}
