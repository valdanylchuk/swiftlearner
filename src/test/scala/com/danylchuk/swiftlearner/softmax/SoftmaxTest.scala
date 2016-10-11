package com.danylchuk.swiftlearner.softmax

import com.danylchuk.swiftlearner.data.{FisherIris, Mnist}
import com.typesafe.scalalogging.LazyLogging
import org.specs2.matcher.DataTables
import org.specs2.mutable.Specification


class SoftmaxTest extends Specification with LazyLogging with DataTables {
  val randomSeed = Some(0L)

  "Softmax" should {
    "operate on primitives" >> {
      val examples = Vector((Array(1.0f, 1.0f), Array(1.0f)),
                            (Array(1.0f, 0.0f), Array(0.0f)),
                            (Array(0.0f, 1.0f), Array(0.0f)),
                            (Array(0.0f, 0.0f), Array(0.0f)))
      val softmax = Softmax.withRandomWeights(2, 1, 0.1f, 10)
      val trained = softmax.learnSeq(examples)

      "elementClass" |
        trained.weights(0).getClass |
        trained.target(0)(0).getClass |
        trained.input(0)(0).getClass |
        trained.lineOut(0)(0).getClass |
        trained.predicted(0)(0).getClass |> { elementClass =>
        elementClass.isPrimitive must beTrue
      }
    }

    "classify the flowers from the Fisher Iris dataset" >> {
      val (trainingSet, testSet) = FisherIris.trainingAndTestDataFloat(randomSeed)

      // Normalize the input values to speed up learning
      def normalize(x: Float): Float = (x - 25.0f) / 25.0f

      val classifier = new SoftmaxClassifier(trainingSet, 1, 0.5f,
        normalize = normalize, randomSeed = randomSeed)

      val accuracy = (for ((species, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $species")
        predicted == species
      }).count { x: Boolean => x } / testSet.size.toFloat

      accuracy must be_>(0.9f)  // 0.94 with the current settings and seed
    }

    "classify the handwritten digits from the MNIST dataset" >> {
      val nRepeat = 50  // usually reaches a minimum and stops much sooner
      // val learnSpeed = 0.01  // for not normalized stable: 0.01 => 0.855
      val learnSpeed = 0.1f  // for normalized naive
      val batchSize = 1  // 1 works best most of the time
      val useStable = false  // "false" achieves better accuracy for normalized inputs
      val stuckIterationLimit = 300  // Increase to 100000 for better results
      val numberOfExamplesToLoad = 30000  // Increase to 60000 for the full training set
      val expectedAccuracy = 0.7f  // 0.92 with stuckIterationLimit = 100000 and full training set

      val (trainingSet, testSet) = Mnist.shuffledTrainingAndTestDataFloat(numberOfExamplesToLoad, randomSeed = randomSeed)

      val start = System.currentTimeMillis

      def normalize(x: Float): Float = x / 256.0f  // source byte range to (0.0; 1.0)

      logger.info("creating the classifier")
      val classifier = new SoftmaxClassifier(trainingSet, nRepeat,
        learnSpeed, stuckIterationLimit, batchSize, normalize, randomSeed, useStable)

      logger.info("checking the accuracy")
      val accuracyData = for ((digit, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $digit")
        (predicted == digit, predicted, digit)
      }

      val overallAccuracy = accuracyData.count { x: (Boolean, Int, Int) => x._1 } / testSet.size.toFloat

      val time = (System.currentTimeMillis - start) / 1000.0
      logger.debug(s"MNIST softmax time: ${time}s")

      overallAccuracy must be_>(expectedAccuracy)
    }
  }
}