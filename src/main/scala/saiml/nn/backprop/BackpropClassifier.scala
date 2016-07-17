package saiml.nn.backprop

import saiml.coll.TraversableOp._


/**
  * A convenience shortcut for applying BackpropNet to classification problems.
  *
  * Assumes consecutive class indices 0..N
  *
  * @param trainingSet A sequence of entries: (class: Int, parameters: Seq[Double])
  * @param nHidden Number of hidden nodes
  * @param nTimes Number of times to repeat the training sequence
  * @param normalize Optional normalization function to speed up learning
  */
class BackpropClassifier(
  trainingSet: Seq[(Int, Seq[Double])],
  nHidden: Int,
  nTimes: Int,
  normalize: (Double) => Double = identity
) {
  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Double]): Int =
    indexOfMax(learned.calculateOutput(Vector(parameters.map(normalize): _*)))

  val nClasses = trainingSet.map(_._1).max + 1
  val nParams = trainingSet.head._2.size

  val examples = trainingSet map Function.tupled { (classIdx, params) =>
    (Vector(params.map(normalize): _*), Vector.fill[Double](nClasses)(0.0).updated(classIdx, 1.0))
  }

  val nn = BackpropNet.randomNet(nParams, nHidden, nClasses)
  val learned = nn.learnSeq(examples.repeat(5000))

  private def indexOfMax(xs: Vector[Double]): Int = xs.zipWithIndex.maxBy(_._1)._2
}