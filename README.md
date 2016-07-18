# SAIML: Scala AI and ML library

[![Build Status](https://travis-ci.org/valdanylchuk/saiml.svg?branch=master)](https://travis-ci.org/valdanylchuk/saiml) [![Join the chat at https://gitter.im/valdanylchuk/saiml](https://badges.gitter.im/valdanylchuk/saiml.svg)](https://gitter.im/valdanylchuk/saiml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

These simple implementations might be useful if you are learning the AI algorithms,
or want to write your own version.

You can use this project as a prototyping library, a cookbook or a cheat sheet.
For high performance and rich features, there are better options.

If you want to read the library name aloud, try "say-mel" :)

## Contents

* [Perceptron](src/main/scala/saiml/nn/perceptron)
([tests](src/test/scala/saiml/nn/perceptron))
A single layer, single node granddaddy of neural networks.
* [Backprop](src/main/scala/saiml/nn/backprop)
([tests](src/test/scala/saiml/nn/backprop))
A neural network with one hidden layer, using backpropagation.
* [Genetic Algorithm](src/main/scala/saiml/ga)
([tests](src/test/scala/saiml/ga))
Genetic Algorithm with elitist tournament selection.
* [Gaussian Naive Bayes](src/main/scala/saiml/bayes/GaussianNaiveBayes.scala)
([tests](src/test/scala/saiml/bayes/GaussianNaiveBayesTest.scala))
Gaussian naive Bayes classifier for continuous parameters.

## Examples

The examples I wrote so far are small enough to fit in the tests, so take a look there.

The best example is classifying the classic
[Fisher Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
with three different algorithms:
* [BackpropClassifier](src/test/scala/saiml/nn/backprop/BackpropClassifierTest.scala): 96% accuracy
* [GeneticIris](src/test/scala/saiml/ga/GeneticTest.scala): 94% accuracy
* [GaussianNaiveBayes](src/test/scala/saiml/bayes/GaussianNaiveBayesTest.scala): 94% accuracy

The accuracy for backprop and the genetic algorithm go even higher with longer training;
these figures are for the quick settings in the automated tests.

## License

This is free software under BSD-style license.
Copyright (c) 2016 Valentyn Danylchuk. See [LICENSE](LICENSE) for details.