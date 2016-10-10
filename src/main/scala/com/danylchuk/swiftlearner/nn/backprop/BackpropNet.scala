package com.danylchuk.swiftlearner.nn.backprop

import com.danylchuk.swiftlearner.math.{ArrayOp, VectorOp}
import com.danylchuk.swiftlearner.math.VectorOp._
import com.typesafe.scalalogging.LazyLogging


/**
  * A neural network with one hidden layer using backpropagation.
  *
  * Use BackpropNet.randomNet() to create a new network with random weights.
  *
  * Ref.: Robert Gordon University, "The Back Propagation Algorithm"
  * https://web.archive.org/web/20150317210621/https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
  * https://en.wikipedia.org/wiki/Backpropagation
  */
class BackpropNet (
  val hiddenLayer: Array[Node],
  val outputLayer: Array[Node]
) extends LazyLogging {
  val nInput = hiddenLayer(0).weights.length
  val nHidden = hiddenLayer.length
  val nOutput = outputLayer.length

  private lazy val hiddenLayerOutput = new Array[Float](nHidden)

  /** Calculate the result without updating the network */
  def calculateOutput(input: Array[Float]): Array[Float] = {
    val l2Output = hiddenLayer.map(_.calculateOutputFor(input))
    outputLayer.map(_.calculateOutputFor(l2Output))
  }

  /**
    * Learn an example using backpropagation.
    *
    * @param rate Learning rate. Must be between 0 and 1. Use 1 if in doubt.
    *             Smaller values might help converge complex cases sometimes.
    **/
  def learn(example: Array[Float], target: Array[Float], rate: Float = 1): BackpropNet = {
    require(rate <= 1 && rate > 0, "learning rate must be between 0 and 1")

    // There is time to think in monads, and there is time to update a few billion weights.

    var i = 0
    while (i < nHidden) {
      hiddenLayer(i).output = hiddenLayer(i).calculateOutputFor(example)
      hiddenLayerOutput(i) = hiddenLayer(i).output
      i += 1
    }

    var j = 0
    while (j < nOutput) {
      outputLayer(j).output = outputLayer(j).calculateOutputFor(hiddenLayerOutput)

      val partialError = target(j) - outputLayer(j).output

      outputLayer(j) = outputLayer(j).updated(hiddenLayerOutput, partialError, rate)
      j += 1
    }

    i = 0
    while (i < nHidden) {
      j = 0
      var partialError = 0.0f
      while (j < nOutput) {
        partialError += outputLayer(j).weights(i) * outputLayer(j).error
        j += 1
      }

      hiddenLayer(i) = hiddenLayer(i).updated(example, partialError, rate)
      i += 1
    }

    this
  }

  /** Convenience shortcut for feeding several examples */
  def learnSeq(examples: Traversable[(Array[Float], Array[Float])],
               rate: Float = 1): BackpropNet = {
    logger.info(s"Learning the training set: ${examples.size} entries")
    examples.foreach { ex =>
      learn(ex._1, ex._2, rate)
    }
    this
  }
}
object BackpropNet {
  /**
    * Creates a new network with given layer sizes and random weights.
    *
    * @param seed Random seed. Use it to keep your tests stable.
    **/
  def randomNet(nInput: Int, nHidden: Int, nOutput: Int, seed: Option[Long] = None) = {
    val r = new scala.util.Random()
    seed.foreach(r.setSeed)
    val hiddenLayer = Array.fill(nHidden)(Node(Array.fill(nInput)(r.nextFloat())))
    val outputLayer = Array.fill(nOutput)(Node(Array.fill(nHidden)(r.nextFloat())))
    new BackpropNet(hiddenLayer, outputLayer)
  }
}

case class Node(
  weights: Array[Float],
  var output: Float = 0,
  var error: Float = 0
) {
  /** Activation function (using the logistic function) */
  private def f(x: Float): Float = (1.0 / (1.0 + java.lang.Math.exp(-x))).toFloat
//  private def f(x: Float): Float = (1.0 / (1.0 + Math.exp(-x))).toFloat

  /** Activation function derivative, used for learning */
  private def ff(oldOutput: Float): Float = oldOutput * (1.0f - oldOutput)

  def calculateOutputFor(input: Array[Float]): Float = f(ArrayOp.dot(input, weights))

  def updated(oldInputs: Array[Float], partialError: Float, rate: Float) = {
    error = ff(output) * partialError

    val n = weights.length
    var i = 0
    while (i < n) {
      weights(i) = weights(i) + rate * error * oldInputs(i)
      i += 1
    }

    this
  }
}