package saiml.softmax

import saiml.coll.TraversableOp._
import saiml.math.ArrayOp


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
  batchSize: Int = 1,
  normalize: (Float) => Float = identity,
  randomSeed: Option[Long] = None
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Float]): Int =
    ArrayOp.indexOfMax(learned.predict(Array(parameters.map(normalize): _*)))

  val nClasses = trainingSet.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    val targetOneHot =  Array.fill[Float](nClasses)(0.0f)
    targetOneHot(classIdx) = 1.0f
    (Array(params.map(normalize): _*), targetOneHot)
  }

  val softmax = Softmax.withRandomWeights(nParams, nClasses, learnRate, randomSeed, batchSize)
  val learned = softmax.learnSeq(examples.repeat(nTimes).toIterable)
}