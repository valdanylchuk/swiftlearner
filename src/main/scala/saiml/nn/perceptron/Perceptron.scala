package saiml.nn.perceptron

import saiml.math.VectorOp._

/**
  * A very simple single-layer perceptron model.
  *
  * Ref.: https://en.wikipedia.org/wiki/Perceptron
  */
class Perceptron(val weights: Vector[Double], val bias: Double = 0.0) {

  /** Activation function */
  def f(input: Vector[Double]): Boolean = weights * input + bias > 0

  def learn(input: Vector[Double], target: Boolean): Perceptron = {
    require(input.size == weights.size, "input size mismatch")

    // Internally uses vector size one larger, always sets input(0) to 1.
    // Then, weights(0) is effectively the learned bias.
    val p = new Perceptron(bias +: weights)
    val learnInput = 1.0 +: input

    val outNum: Double = if (p.f(learnInput)) 1 else 0
    val targetNum: Double = if (target) 1 else 0

    val newWeights = p.weights zip learnInput map Function.tupled { (w, x) =>
      w + (targetNum - outNum) * x
    }

    new Perceptron(weights = newWeights.tail, bias = newWeights.head)
  }

  /** Convenience shortcut for feeding several examples */
  def learn(examples: Traversable[(Vector[Double], Boolean)]): Perceptron = {
    examples.foldLeft(this) { (acc, x) =>
      acc.learn(x._1, x._2)
    }
  }
}