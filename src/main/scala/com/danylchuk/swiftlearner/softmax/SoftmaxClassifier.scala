package com.danylchuk.swiftlearner.softmax

import com.danylchuk.swiftlearner.coll.TraversableOp._
import com.danylchuk.swiftlearner.math.ArrayOp


/**
  * A convenience shortcut for applying Softmax regression to classification problems.
  *
  * Assumes consecutive class indices 0..N
  *
  * @param trainingSet A sequence of entries: (class: Int, parameters: Seq[Float])
  * @param nTimes Number of times to repeat the training sequence
  * @param normalize Optional input normalization function to speed up learning
  */
class SoftmaxClassifier(
  trainingSet: Seq[(Int, Seq[Float])],
  nTimes: Int,
  learnRate: Float = 0.001f,
  stuckIterationLimit: Int = 10000,
  batchSize: Int = 1,
  normalize: (Float) => Float = identity,
  randomSeed: Option[Long] = None,
  useStable: Boolean = true
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Float]): Int = {
    val predicted = learned.predict(normalizeSeq(parameters, normalizedInput))
    ArrayOp.indexOfMax(predicted)
  }

  val nClasses = trainingSet.iterator.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  private val normalizedInput = new Array[Float](nParams)

  private def normalizeSeq(params: Seq[Float],
                           outArray: Array[Float] = new Array[Float](nParams))
  : Array[Float] = {
    var i = 0
    while (i < nParams) {
      outArray(i) = normalize(params(i))
      i += 1
    }
    outArray
  }

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    val targetOneHot =  Array.fill[Float](nClasses)(0.0f)
    targetOneHot(classIdx) = 1.0f
    (normalizeSeq(params), targetOneHot)
  }

  val softmax = Softmax.withRandomWeights(nParams, nClasses, learnRate,
    stuckIterationLimit, randomSeed, batchSize, useStable)
  lazy val learned = {
    for (_ <- Iterator.range(0, nTimes))
      softmax.learnSeq(examples)
    softmax
  }
}