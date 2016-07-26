package saiml.knn

import org.specs2.mutable.Specification
import saiml.data.FisherIris


class KNearestNeighborsTest extends Specification {
  "KNearestNeighbors" should {
    "sort the flowers from the Fisher Iris dataset" >> {
      val (trainingSet, testSet) = FisherIris.trainingAndTestData

      val classifier = new KNearestNeighbors(trainingSet)

      val accuracy = (for ((species, params) <- testSet) yield {
        classifier.predict(params, 1) == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.94 is typical
    }
  }
}