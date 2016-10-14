package com.danylchuk.swiftlearner.nn.backprop


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
  val nClasses = trainingSet.iterator.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  private val normalizedInput = new Array[Float](nParams)

  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Float]): Int = {
    require(parameters.size == nParams, "wrong number of parameters")
    val iter = parameters.iterator
    var i = 0
    while (i < nParams) {
      normalizedInput(i) = normalize(iter.next)
      i += 1
    }
    learned.predict(normalizedInput)
  }

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