#include "nn/BackpropClassifier.hpp"
#include <iostream> // For error messages
#include <algorithm>
#include <numeric>

BackpropClassifier::BackpropClassifier(
    const std::vector<std::pair<int, vec>>& trainingSet,
    size_t nHidden,
    int nTimes,
    float learnRate,
    std::function<float(float)> normalize,
    std::optional<long> seed
)   : nClasses([&](){
        int maxClass = 0;
        for (const auto& entry : trainingSet) {
            maxClass = std::max(maxClass, entry.first);
        }
        return static_cast<size_t>(maxClass + 1);
    }()),
    nParams(trainingSet.empty() ? 0 : trainingSet[0].second.size()),
    normalizedInput(nParams),
    examples(createExamples(trainingSet, normalize)),
    learned([&](){
        BackpropNet trainedNet = BackpropNet::randomNet(nParams, nHidden, nClasses, seed);
        trainedNet.train(examples, nTimes, learnRate);
        return trainedNet;
    }()),
    normalizeFunc(normalize)
{
    if (trainingSet.empty()) {
        std::cerr << "Warning: BackpropClassifier initialized with an empty training set." << std::endl;
    }
}

int BackpropClassifier::predict(const vec& parameters) {
    if (parameters.size() != nParams)
        throw std::invalid_argument("wrong number of parameters");

    for (size_t i = 0; i < nParams; ++i)
        normalizedInput[i] = normalizeFunc(parameters[i]);

    return learned.predict(normalizedInput);
}

std::vector<Example> BackpropClassifier::createExamples(
    const std::vector<std::pair<int, vec>>& trainingSet,
    const std::function<float(float)>& normalize
) {
    std::vector<Example> examples;
    examples.reserve(trainingSet.size());

    for (const auto& entry : trainingSet) {
        int classIdx = entry.first;
        const vec& params = entry.second;

        vec targetOneHot(nClasses, 0.0f);
        targetOneHot[classIdx] = 1.0f;

        vec normalizedParams(params.size());
        std::transform(params.begin(), params.end(), normalizedParams.begin(), normalize);

        examples.emplace_back(Example(normalizedParams, targetOneHot));
    }

    return examples;
}