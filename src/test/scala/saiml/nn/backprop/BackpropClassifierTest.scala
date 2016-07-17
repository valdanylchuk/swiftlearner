package saiml.nn.backprop

import com.typesafe.scalalogging.LazyLogging
import org.specs2.mutable.Specification
import saiml.data.FisherIris


class BackpropClassifierTest extends Specification with LazyLogging {
  "BackpropClassifier" should {
    "sort the flowers from the Fisher Iris dataset" >> {
      val (trainingSet, testSet) = FisherIris.trainingAndTestData

      // Normalize the input values to speed up learning
      def normalize(x: Double): Double = (x - 25) / 25

      val classifier = new BackpropClassifier(trainingSet, 3, 5000, normalize)

      val accuracy = (for ((species, params) <- testSet) yield {
        classifier.predict(params) == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.96 is typical
    }
  }
}