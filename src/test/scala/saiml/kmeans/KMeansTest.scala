package saiml.kmeans

import org.specs2.mutable.Specification
import saiml.data.FisherIris


class KMeansTest extends Specification{
  "KMeans" should {
    "create clusters similar to the known Iris dataset labels" >> {
      val (knownLabels, data) = FisherIris.irisData.unzip
      val k = knownLabels.distinct.size
      val kMeans = new KMeans(data, k)

      // Now, we have a small problem because the label values are different,
      // although the sets they define should be similar.

      // Pair the labels. Then the most common k pairs will be the correct labels,
      // and we can count the rest to find the number of errors and the accuracy.
      val labelPairs = knownLabels zip kMeans.labels
      val groups = labelPairs.groupBy(identity).values.toVector
      val errorGroups = groups.sortWith((a, b) => a.size > b.size).drop(k)
      val nErrors = errorGroups.map(_.size).sum
      val accuracy = 1.0 - nErrors.toDouble / data.size

      accuracy must be_>(0.8)  // typical is 0.87
    }
  }
}