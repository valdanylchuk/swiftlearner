package saiml.nn.backprop

import saiml.math.VectorOp._


/**
  * A neural network with one hidden layer using backpropagation.
  *
  * Use BackpropNet.randomNet() to create a new network with random weights.
  *
  * Ref.: Robert Gordon University, "The Back Propagation Algorithm"
  * https://web.archive.org/web/20150317210621/https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
  */
class BackpropNet(
  val hiddenLayer: Vector[Node],
  val outputLayer: Vector[Node]
) {
  /** Calculate the result without updating the network */
  def calculateOutput(input: Vector[Double]): Vector[Double] = {
    val l2Output = hiddenLayer.map(_.calculateOutputFor(input))
    outputLayer.map(_.calculateOutputFor(l2Output))
  }

  /**
    * Learn an example using backpropagation.
    *
    * @param rate Learning rate. Must be between 0 and 1. Use 1 if in doubt.
    *             Smaller values might help converge complex cases sometimes.
    **/
  def learn(example: Vector[Double], target: Vector[Double], rate: Double = 1): BackpropNet = {
    require(rate <= 1 && rate > 0, "learning rate must be between 0 and 1")

    val calcHiddenLayer = hiddenLayer.map(_.withCalculatedOutputFor(example))
    val calcHiddenLayerOutput = calcHiddenLayer.map(_.output)
    val calcOutputLayer = outputLayer.map(_.withCalculatedOutputFor(calcHiddenLayerOutput))

    val updatedOutputLayer = for ((outputNode, t) <- calcOutputLayer zip target) yield {

      val partialError = t - outputNode.output

      outputNode.updated(calcHiddenLayerOutput, partialError, rate)
    }
    val updatedOutputLayerErrors = updatedOutputLayer.map(_.error)

    val updatedHiddenLayer = for ((hiddenNode, outputLayerWeightsForThisHiddenNode) <-
                             calcHiddenLayer zip outputLayer.map(_.weights).transpose) yield {

      val partialError = outputLayerWeightsForThisHiddenNode * updatedOutputLayerErrors

      hiddenNode.updated(example, partialError, rate)
    }

    new BackpropNet(updatedHiddenLayer, updatedOutputLayer)
  }

  /** Convenience shortcut for feeding several examples */
  def learnSeq(examples: Traversable[(Vector[Double], Vector[Double])],
               rate: Double = 1): BackpropNet = {
    examples.foldLeft(this) { (nn, ex) =>
      nn.learn(ex._1, ex._2, rate)
    }
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
    val hiddenLayer = Vector.fill(nHidden)(new Node(Vector.fill(nInput)(r.nextDouble())))
    val outputLayer = Vector.fill(nOutput)(new Node(Vector.fill(nHidden)(r.nextDouble())))
    new BackpropNet(hiddenLayer, outputLayer)
  }
}

case class Node(
  weights: Vector[Double],
  output: Double = 0,
  error: Double = 0
) {
  /** Activation function (using the logistic function) */
  private def f(x: Double) = 1.0 / (1.0 + Math.exp(-x))

  /** Activation function derivative, used for learning */
  private def ff(oldOutput: Double) = oldOutput * (1.0 - oldOutput)

  def calculateOutputFor(input: Vector[Double]): Double = f(input * weights)

  def withCalculatedOutputFor(input: Vector[Double]) =
    this.copy(output = calculateOutputFor(input))

  def updated(oldInputs: Vector[Double], partialError: Double, rate: Double) = {
    val error = ff(output) * partialError
    val newWeights = for ((w, x) <- weights zip oldInputs)
                     yield w + rate * error * x
    new Node(newWeights, output, error)
  }
}