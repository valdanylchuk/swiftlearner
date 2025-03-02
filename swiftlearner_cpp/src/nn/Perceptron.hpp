#pragma once
#include <vector>

using vec = std::vector<double>;

class Perceptron {
public:
  Perceptron(const vec& weights, double bias = 0.0);

  Perceptron(size_t input_size, double initial_value = 0.0)
    : weights_(input_size, initial_value), bias_(initial_value) {}

  bool activate(const vec& input) const;

  Perceptron learn(const vec& input, bool target);

  Perceptron train(const std::vector<std::pair<vec, bool>>& examples, int epochs = 1);

private:
  vec weights_;
  double bias_;
};
