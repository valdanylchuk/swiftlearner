#pragma once

#include <vector>
#include <functional>
#include <optional>
#include "nn/BackpropNet.hpp"

class BackpropClassifier {
public:
    BackpropClassifier(
        const std::vector<std::pair<int, vec>>& trainingSet,
        size_t nHidden,
        int nTimes,
        float learnRate = 1.0f,
        std::function<float(float)> normalize = [](float x) { return x; },
        std::optional<long> seed = std::nullopt
    );

    int predict(const vec& parameters);

private:
    size_t nClasses;
    size_t nParams;
    vec normalizedInput;
    std::vector<Example> examples;
    BackpropNet learned;
    std::function<float(float)> normalizeFunc;

    std::vector<Example> createExamples(
        const std::vector<std::pair<int, vec>>& trainingSet,
        const std::function<float(float)>& normalize
    );
};