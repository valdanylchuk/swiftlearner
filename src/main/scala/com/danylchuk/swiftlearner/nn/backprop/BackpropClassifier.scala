package com.danylchuk.swiftlearner.nn.backprop

import com.danylchuk.swiftlearner.coll.TraversableOp._


/**
  * A convenience shortcut for applying BackpropNet to classification problems.
  *
  * Assumes consecutive class indices 0..N
  *
  * @param trainingSet A sequence of entries: (class: Int, parameters: Seq[Float])
  * @param nHidden Number of hidden nodes
  * @param nTimes Number of times to repeat the training sequence
  * @param normalize Optional normalization function to speed up learning
  */
class BackpropClassifier(
  trainingSet: Seq[(Int, Seq[Float])],
  nHidden: Int,
  nTimes: Int,
  learnRate: Float = 1.0f,
  normalize: (Float) => Float = identity,
  seed: Option[Long] = None
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Float]): Int = {
    val predicted = learned.calculateOutput(Array(parameters.map(normalize): _*))
    predicted.indexOf(predicted.max)
    // ArrayOp.indexOfMax(learned.calculateOutput(Array(parameters.map(normalize): _*)))
  }

  val nClasses = trainingSet.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    val targetOneHot =  Array.fill[Float](nClasses)(0.0f)
    targetOneHot(classIdx) = 1.0f
    (Array(params.map(normalize): _*), targetOneHot)
  }

  val nn = BackpropNet.randomNet(nParams, nHidden, nClasses, seed)
  lazy val learned = {
    for (_ <- Iterator.range(0, nTimes))
      nn.learnSeq(examples, learnRate)
    nn
  }
}