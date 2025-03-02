#include "nn/Perceptron.hpp"
#include <numeric> // std::inner_product

/**
  * A very simple single-layer perceptron model.
  *
  * Ref.: https://en.wikipedia.org/wiki/Perceptron
  */
Perceptron::Perceptron(const vec& weights, double bias) : weights_(weights), bias_(bias) {}

bool Perceptron::activate(const vec& input) const {
  if (input.size() != weights_.size())
    throw std::invalid_argument("input size mismatch");

  return std::inner_product(weights_.begin(), weights_.end(), input.begin(), bias_) > 0;
}

Perceptron Perceptron::learn(const vec& input, bool target) {
  if (input.size() != weights_.size())
    throw std::invalid_argument("input size mismatch");

  double output = activate(input) ? 1.0 : 0.0;
  double error = (target ? 1.0 : 0.0) - output;

  bias_ += error * 1.0; // Input is implicitly 1 for bias update

  for (size_t i = 0; i < weights_.size(); ++i)
    weights_[i] += error * input[i];

  return *this;
}

/** Convenience shortcut for feeding several examples */
Perceptron Perceptron::train(const std::vector<std::pair<vec, bool>>& examples, int epochs) {
  for (int i = 0; i < epochs; ++i)
    for (const auto& [input, target] : examples)
      *this = learn(input, target);
  return *this;
}