package com.danylchuk.swiftlearner.bayes

import com.danylchuk.swiftlearner.data.Mnist
import org.specs2.mutable.Specification

class BernoulliNaiveBayesTest extends Specification {
  "BernoulliNaiveBayes" should {
    "sort the digits from the MNIST dataset" >> {
      val (trainingSet, testSet) =
        Mnist.shuffledTrainingAndTestDataBinary(nTrainPoints = 6000, randomSeed = Some(0L))

      val classifier = BernoulliNaiveBayes.fromTrainingSet(trainingSet)

      val accuracy = (for ((digit, params) <- testSet) yield {
        classifier.predict(params) == digit
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.84 with full training set
    }
  }
}
