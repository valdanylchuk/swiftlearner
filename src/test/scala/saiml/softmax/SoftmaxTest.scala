package saiml.softmax

import com.typesafe.scalalogging.LazyLogging
import org.specs2.mutable.Specification
import saiml.data.FisherIris


class SoftmaxTest extends Specification with LazyLogging {
  "Softmax" should {
    "classify the flowers from the Fisher Iris dataset" >> {
      val randomSeed = Some(0L)
      val (trainingSet, testSet) = FisherIris.trainingAndTestDataFloat(randomSeed)

      // Normalize the input values to speed up learning
      def normalize(x: Float): Float = (x - 25) / 25

      val classifier = new SoftmaxClassifier(trainingSet, 10, 0.01f, 2, normalize, randomSeed)

      val accuracy = (for ((species, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $species")
        predicted == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.9 is common
    }
  }
}