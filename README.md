# SAIML: Scala AI and ML library

[![Build Status](https://travis-ci.org/valdanylchuk/saiml.svg?branch=master)](https://travis-ci.org/valdanylchuk/saiml) [![Join the chat at https://gitter.im/valdanylchuk/saiml](https://badges.gitter.im/valdanylchuk/saiml.svg)](https://gitter.im/valdanylchuk/saiml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

The simple implementations here may help you learn the AI algorithms
or write your own versions. Some of them fit within 10 lines of code.
It shows the simplicity of the classic machine learning techniques,
and the expressive power of Scala.

Use this project as a prototyping library, a cookbook, or a cheat sheet.
For high performance and rich features, there are better options.
Still, these methods are fully functional and work well for small problem sizes.

To read the library name aloud, try "say-mel" :)

To make one ML enthusiast happy, please star or fork this project,
or leave a comment in the chat. ;)

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
* [k-Nearest Neighbors](src/main/scala/saiml/knn)
([tests](src/test/scala/saiml/knn))
k-Nearest-Neighbors classifier.

## Examples

The examples I wrote so far are small enough to fit in the tests, so take a look there.

<img align="right" src="img/iris-virginica.jpg" alt="Iris Virginica flower; credit: Wikimedia Commons"/>

One example is classifying the classic
[Fisher Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
with different algorithms:
* [BackpropClassifier](src/test/scala/saiml/nn/backprop/BackpropClassifierTest.scala): 96% accuracy
* [GeneticIris](src/test/scala/saiml/ga/GeneticTest.scala): 94% accuracy
* [GaussianNaiveBayes](src/test/scala/saiml/bayes/GaussianNaiveBayesTest.scala): 94% accuracy
* [KNearestNeighbors](src/test/scala/saiml/knn/KNearestNeighborsTest.scala): 94% accuracy

The accuracy for backprop and the genetic algorithm goes higher with longer training;
these figures are for the quick settings in the automated tests.

## License

This is free software under a BSD-style license.
Copyright (c) 2016 Valentyn Danylchuk. See [LICENSE](LICENSE) for details.