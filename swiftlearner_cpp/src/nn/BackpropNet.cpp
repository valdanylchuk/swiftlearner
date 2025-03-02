#include "nn/BackpropNet.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits> // numeric_limits

float Node::calculateOutputFor(const vec& input) {
    return f(std::inner_product(input.begin(), input.end(), weights.begin(), 0.0f));
}

void Node::update(const vec& oldInputs, float partialError, float rate) {
    error = ff(output) * partialError;
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] += rate * error * oldInputs[i];
}

BackpropNet::BackpropNet(std::vector<Node> hiddenLayer, std::vector<Node> outputLayer)
    : hiddenLayer(std::move(hiddenLayer)),
      outputLayer(std::move(outputLayer)),
      hiddenLayerOutput(this->hiddenLayer.size()) // Initialize directly
{}

/** @return Index of the highest output */
int BackpropNet::predict(const vec& input) {
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayerOutput[i] = hiddenLayer[i].calculateOutputFor(input);
    }

    float maxOutput = -std::numeric_limits<float>::infinity();
    int maxOutputIndex = 0;

    for (int i = 0; i < outputLayer.size(); ++i) {
        float nodeOutput = outputLayer[i].calculateOutputFor(hiddenLayerOutput);
        if (nodeOutput > maxOutput) {
            maxOutput = nodeOutput;
            maxOutputIndex = i;
        }
    }
    return maxOutputIndex;
}

/** @return Calculate output[0], for testing only */
float BackpropNet::testOut(const vec& input) {
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayerOutput[i] = hiddenLayer[i].calculateOutputFor(input);
    }
    return outputLayer[0].calculateOutputFor(hiddenLayerOutput);
}

/**
  * Learn an example using backpropagation.
  *
  * @param rate Learning rate. Must be between 0 and 1. Use 1 if in doubt.
  *             Smaller values might help converge complex cases sometimes.
  **/
 BackpropNet& BackpropNet::learn(const Example& ex, float rate) {
    if (rate > 1.0f || rate <= 0.0f) {
        throw std::invalid_argument("learning rate must be between 0 and 1");
    }

    // Forward pass: Calculate and *store* outputs in Node::output
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayer[i].output = hiddenLayer[i].calculateOutputFor(ex.input);
        hiddenLayerOutput[i] = hiddenLayer[i].output;
    }

    // Forward pass for output layer
    for (size_t j = 0; j < outputLayer.size(); ++j) {
        outputLayer[j].output = outputLayer[j].calculateOutputFor(hiddenLayerOutput);
    }

    // Backward pass: Update output layer
    for (size_t j = 0; j < outputLayer.size(); ++j) {
        float partialError = ex.target[j] - outputLayer[j].output;
        outputLayer[j].update(hiddenLayerOutput, partialError, rate);
    }

    // Backward pass: Update hidden layer
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        float partialError = 0.0f;
        for (size_t j = 0; j < outputLayer.size(); ++j) {
            partialError += outputLayer[j].weights[i] * outputLayer[j].error;
        }
        hiddenLayer[i].update(ex.input, partialError, rate);
    }

    return *this;
}

BackpropNet& BackpropNet::train(const std::vector<Example>& examples, int epochs, float rate) {
    for (int i = 0; i < epochs; ++i)
        for (const auto& ex : examples)
            learn(ex, rate);
    return *this;
}

BackpropNet BackpropNet::randomNet(size_t nInput, size_t nHidden, size_t nOutput, std::optional<long> seed) {
    std::mt19937 generator(seed.value_or(std::random_device{}()));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto randWeights = [&](size_t n) {
        vec w(n);
        std::generate(w.begin(), w.end(), [&]() { return dist(generator); });
        return w;
    };
    std::vector<Node> hiddenLayer(nHidden, Node(randWeights(nInput)));
    std::vector<Node> outputLayer(nOutput, Node(randWeights(nHidden)));
    return BackpropNet(std::move(hiddenLayer), std::move(outputLayer));
}