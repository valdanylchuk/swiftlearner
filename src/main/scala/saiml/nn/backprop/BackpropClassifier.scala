package saiml.nn.backprop

import saiml.coll.TraversableOp._


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
  normalize: (Float) => Float = identity
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Float]): Int =
    indexOfMax(learned.calculateOutput(Vector(parameters.map(normalize): _*)))

  val nClasses = trainingSet.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    (Vector(params.map(normalize): _*), Vector.fill[Float](nClasses)(0.0f).updated(classIdx, 1.0f))
  }

  val nn = BackpropNet.randomNet(nParams, nHidden, nClasses)
  val learned = nn.learnSeq(examples.repeat(nTimes))

  private def indexOfMax(xs: Vector[Float]): Int = xs.zipWithIndex.maxBy(_._1)._2
}