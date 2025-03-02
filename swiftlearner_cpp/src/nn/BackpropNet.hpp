#pragma once
#include <vector>
#include <optional>

using vec = std::vector<float>;
struct Example { vec input, target; };

class Node {
public:
    vec weights;
    float output, error;

    Node(const vec& w) : weights(std::move(w)) {}

    float calculateOutputFor(const vec& input);
    void update(const vec& oldInputs, float partialError, float rate);

private:
    // Activation function (using the logistic function)
    static float f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    // Activation function derivative, used for learning
    static float ff(float oldOutput) { return oldOutput * (1.0f - oldOutput); }
};

class BackpropNet {
public:
    std::vector<Node> hiddenLayer, outputLayer;
    vec hiddenLayerOutput;

    BackpropNet(std::vector<Node> hiddenLayer, std::vector<Node> outputLayer);

    int predict(const vec& input);
    float testOut(const vec& input);
    BackpropNet& learn(const Example& example, float rate);
    BackpropNet& train(const std::vector<Example>& examples, int epochs = 1, float rate = 1.0f);

    static BackpropNet randomNet(size_t nInput, size_t nHidden, size_t nOutput, std::optional<long> seed = {});
};
