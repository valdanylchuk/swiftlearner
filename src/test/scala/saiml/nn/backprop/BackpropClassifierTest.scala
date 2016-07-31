package saiml.nn.backprop

import com.typesafe.scalalogging.LazyLogging
import org.specs2.mutable.Specification
import saiml.data.{FisherIris, Mnist}


class BackpropClassifierTest extends Specification with LazyLogging {
  "BackpropClassifier" should {
    "classify the flowers from the Fisher Iris dataset" >> {
      val (trainingSet, testSet) = FisherIris.trainingAndTestDataFloat

      // Normalize the input values to speed up learning
      def normalize(x: Float): Float = (x - 25) / 25

      val classifier = new BackpropClassifier(trainingSet, 3, 5000, normalize)

      val accuracy = (for ((species, params) <- testSet) yield {
        classifier.predict(params) == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.96 is typical
    }

    "classify the handwritten digits from the MNIST dataset" >> {
      val (trainingSet, testSet) = Mnist.trainingAndTestData(10000)

      val start = System.currentTimeMillis

      val classifier = new BackpropClassifier(trainingSet, 200, 1)

      val accuracy = (for ((digit, params) <- testSet) yield {
        classifier.predict(params) == digit
      }).count { x: Boolean => x } / testSet.size.toDouble

      val time = (System.currentTimeMillis - start) / 1000.0
      logger.debug(s"MNIST time: ${time}s")

      accuracy must be_>(0.0)
    }
  }
}