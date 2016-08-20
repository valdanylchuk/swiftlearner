package saiml.softmax

import com.typesafe.scalalogging.LazyLogging
import org.specs2.matcher.DataTables
import org.specs2.mutable.Specification
import saiml.data.{FisherIris, Mnist}


class SoftmaxTest extends Specification with LazyLogging with DataTables {
  val randomSeed = Some(0L)

  "Softmax" should {
    "operate on primitives" >> {
      val examples = Vector((Array(1.0, 1.0), Array(1.0)),
                            (Array(1.0, 0.0), Array(0.0)),
                            (Array(0.0, 1.0), Array(0.0)),
                            (Array(0.0, 0.0), Array(0.0)))
      val softmax = Softmax.withRandomWeights(2, 1, 0.1, 10)
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
      val (trainingSet, testSet) = FisherIris.trainingAndTestDataDouble(randomSeed)

      // Normalize the input values to speed up learning
      def normalize(x: Double): Double = (x - 25.0) / 25.0

      val classifier = new SoftmaxClassifier(trainingSet, 1, 0.5,
        normalize = normalize, randomSeed = randomSeed)

      val accuracy = (for ((species, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $species")
        predicted == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.9)  // 0.94 with the current settings and seed
    }

    "classify the handwritten digits from the MNIST dataset" >> {
      val nRepeat = 50  // usually reaches a minimum and stops much sooner
      // val learnSpeed = 0.01  // for not normalized stable: 0.01 => 0.855
      val learnSpeed = 0.1  // for normalized naive
      val batchSize = 1  // 1 works best most of the time
      val useStable = false  // "false" achieves better accuracy for normalized inputs
      val stuckIterationLimit = 200  // Increase to 100000 for better results
      val numberOfExamplesToLoad = 60000  // Increase to 60000 for the full training set
      val expectedAccuracy = 0.7  // 0.92 with stuckIterationLimit = 100000 and full training set

      val (trainingSet, testSet) = Mnist.shuffledTrainingAndTestData(numberOfExamplesToLoad, randomSeed = randomSeed)

      val start = System.currentTimeMillis

      def normalize(x: Double): Double = x / 256.0  // source byte range to (0.0; 1.0)

      logger.info("creating the classifier")
      val classifier = new SoftmaxClassifier(trainingSet, nRepeat,
        learnSpeed, stuckIterationLimit, batchSize, normalize, randomSeed, useStable)

      logger.info("checking the accuracy")
      val accuracyData = for ((digit, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $digit")
        (predicted == digit, predicted, digit)
      }

      val overallAccuracy = accuracyData.count { x: (Boolean, Int, Int) => x._1 } / testSet.size.toDouble

      val time = (System.currentTimeMillis - start) / 1000.0
      logger.debug(s"MNIST softmax time: ${time}s")

      overallAccuracy must be_>(expectedAccuracy)
    }
  }
}