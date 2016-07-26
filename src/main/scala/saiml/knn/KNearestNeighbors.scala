package saiml.knn

import saiml.math.VectorOp

/**
  * k-nearest neighbors classifier
  *
  * @param trainingData: Seq[(classId, params)]
  *
  * Ref.: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
  * http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
  */
class KNearestNeighbors(trainingData: Seq[(Int, Vector[Double])]) {

  /** Predicts the class for testParams based on k nearest neighbors. */
  def predict(testParams: Vector[Double], k: Int): Int =
    getClassFromNeighbors(getNeighbors(testParams, k))

  /**
    * This is a slow, naive implementation.
    * We use squared distances to make it a bit faster while keeping it simple.
    * For a more efficient search, organize your points into a specialized
    * data structure, e.g. a kd-tree.
    */
  private def getNeighbors(x: Vector[Double], k: Int): Seq[(Int, Vector[Double])] =
    trainingData.map(y => (y, VectorOp.squaredDistance(x, y._2)))
      .sortBy(_._2).take(k).map(_._1)

  /** Gets class by majority vote from the neighbors. */
  private def getClassFromNeighbors(neighbors: Seq[(Int, Vector[Double])]) =
    neighbors.map(_._1).groupBy(identity).maxBy(_._2.size)._1
}