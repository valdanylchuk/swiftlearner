package com.danylchuk.swiftlearner.softmax

import com.danylchuk.swiftlearner.coll.TraversableOp._
import com.danylchuk.swiftlearner.math.ArrayOp


/**
  * A convenience shortcut for applying Softmax regression to classification problems.
  *
  * Assumes consecutive class indices 0..N
  *
  * @param trainingSet A sequence of entries: (class: Int, parameters: Seq[Double])
  * @param nTimes Number of times to repeat the training sequence
  * @param normalize Optional input normalization function to speed up learning
  */
class SoftmaxClassifier(
  trainingSet: Seq[(Int, Seq[Double])],
  nTimes: Int,
  learnRate: Double = 0.001,
  stuckIterationLimit: Int = 10000,
  batchSize: Int = 1,
  normalize: (Double) => Double = identity,
  randomSeed: Option[Long] = None,
  useStable: Boolean = true
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Double]): Int = {
    val predicted = learned.predict(normalizeSeq(parameters))
    ArrayOp.indexOfMax(predicted)
  }

  private def normalizeSeq(params: Seq[Double]): Array[Double] = {
    val n = params.length
    val normalized = new Array[Double](n)
    var i = 0
    while (i < n) {
      normalized(i) = normalize(params(i))
      i += 1
    }
    normalized
  }

  val nClasses = trainingSet.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    val targetOneHot =  Array.fill[Double](nClasses)(0.0)
    targetOneHot(classIdx) = 1.0
    (normalizeSeq(params), targetOneHot)
  }

  val softmax = Softmax.withRandomWeights(nParams, nClasses, learnRate,
    stuckIterationLimit, randomSeed, batchSize, useStable)
  lazy val learned = {
    for (_ <- Iterator.range(0, nTimes))
      softmax.learnSeq(examples.repeat(nTimes))
    softmax
  }
}