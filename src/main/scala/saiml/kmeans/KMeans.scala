package saiml.kmeans

import saiml.coll.SeqOp._
import saiml.math.VectorOp
import saiml.math.VectorOp._


/**
  * k-Means Clustering
  *
  * @param data Data points (observations)
  * @param k Target number of clusters
  *
  * Ref: https://en.wikipedia.org/wiki/K-means_clustering
  */
class KMeans(data: Seq[Vector[Double]], k: Int, maxIterations: Int = 10000) {
  require(k <= data.size, "k must be no smaller than the dataset size")

  /** @return Labels assigned by the algorithm, corresponding to the input data sequence. */
  lazy val labels: Seq[Int] = {
    var i = 0
    var oldLabels: Seq[Int] = Nil
    var labels: Seq[Int] = Nil
    var means = firstMeans
    do {
      oldLabels = labels
      labels = assignLabels(means)
      means = updateMeans(labels)
      i += 1
    } while (labels != oldLabels && i < maxIterations)
    labels
  }

  /** @return Input data grouped by cluster / label */
  lazy val clusters: Seq[Seq[Vector[Double]]] = clustersFromLabels(labels)

  private lazy val firstMeans = data.randomSample(k)

  /** @return Sequence of labels corresponding to our data sequence */
  private def assignLabels(means: Seq[Vector[Double]]): Seq[Int] = data.map { x =>
    val labelsWithDistance =
      for ((mean, label) <- means.zipWithIndex)
      yield (label, VectorOp.squaredDistance(x, mean) )

    labelsWithDistance.minBy(_._2)._1
  }

  /** @return New means. Labels are discarded, their numeric values are random anyway. */
  private def updateMeans(labels: Seq[Int]): Seq[Vector[Double]] = {
    clustersFromLabels(labels).map { cluster =>
      cluster.reduce(_ + _) / cluster.size.toDouble  // new mean vector in each cluster
    }.toSeq
  }

  private def clustersFromLabels(labels: Seq[Int]): Seq[Seq[Vector[Double]]] =
    data.zip(labels).groupBy(_._2)
      .mapValues(_.map(_._1))  // remove the remaining zipped labels
      .values.toSeq
}