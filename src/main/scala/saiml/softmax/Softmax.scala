package saiml.softmax

import com.typesafe.scalalogging.LazyLogging
import saiml.math.{ArrayOp, MatrixOp}


/**
  * Softmax (multinomial logistic) regression
  *
  * Ref: https://en.wikipedia.org/wiki/Multinomial_logistic_regression
  * http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
  * http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
  * http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
  * https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
  * https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad
  *
  * @param weights Initial weights (pre-trained or random)
  *   weights(i)(j) is the weight for evidence i based on input parameter j
  *     i = 0..nClasses
  *     j = 0..nInputs (where nInputs = inputLength + 1 for pseudo-input used for bias handling)
  *   Modeled as an array of concatenated rows.
  * @param learnRate Between 0 and 1. As we use AdaGrad, the effective rate will gradually decrease.
  *                  You can start relatively high: 0.01 - 0.1
  * @param stuckIterationLimit How many more samples to try if there is no improvement.
  *                            Set based on input size, your patience and target accuaracy.
  * @param batchSize Mini-batch size for "mini-batch SGD". Most of the time, 1 is the best size.
  * @param useStable Use a numerically stable softmax version, more tolerant to broad input ranges.
  *                  Recommended most of the time.
  */
class Softmax(weights: Array[Double],
              nClasses: Int,
              inputDataLength: Int,
              learnRate: Double,
              stuckIterationLimit: Int = 10000,
              batchSize: Int = 1,
              useStable: Boolean = true)
  extends LazyLogging {

  val nInputs = inputDataLength + 1  // with an additional control pseudo-input fixed at 1

  /** Current batch size */
  private var m = 0

  /**
    * Target "true" distribution of nClasses output values for supervised learning.
    * Use one-hot encoding: all zeroes except 1.0 for the target class.
    */
  private val target: Array[Array[Double]] = Array.fill(batchSize)(new Array[Double](nClasses))

  /** current examples batch input[m][nInputs] */
  private val input: Array[Array[Double]] = Array.fill(batchSize)(new Array[Double](nInputs))

  /** lineOut[m][nClasses]: output of the first level linear predictor before the softmax layer */
  private val lineOut: Array[Array[Double]] = Array.fill(batchSize)(new Array[Double](nClasses))

  /** predicted[m][nClasses] = predict(input) */
  private val predicted: Array[Array[Double]] = Array.fill(batchSize)(new Array[Double](nClasses))

  /** gradients of the loss function */
  private val gradients: Array[Array[Double]] = Array.fill(nClasses)(new Array[Double](nInputs))

  /**
    * y = softmax(x) = normalize(exp(x)) = exp(x(i)) / sum (exp(x))
    *
    * Naive "by the book" version; only works with normalized, stable input.
    */
  def softmax(x: Array[Double]): Array[Double] = {
    val n = x.length
    var i = 0
    val y = new Array[Double](n)
    var sum: Double = 0.0
    while (i < n) {
      y(i) = math.exp(x(i)).toFloat
      sum += y(i)
      i += 1
    }
    i = 0
    while (i < n) {
      y(i) /= sum
      i += 1
    }
    y
  }

  /**
    * This version works around some numeric issues (overflow/underflow).
    *
    * Original form: y(i) = exp(x(i)) / sum (exp(x))
    *
    * Stable form: y(i) = exp( x(i) - logSumExp(x) )
    *   where logSumExp(x) = max(x) + log(sum(x-max(x)))
    */
  def softmaxStable(x: Array[Double]): Array[Double] = {
    val xMax = x.max  // for logSumExp
    val n = x.length
    var i = 0
    val y = new Array[Double](n)
    var sum: Double = 0.0
    while (i < n) {
      sum += x(i) - xMax  // for logSumExp
      i += 1
    }
    val logSum1 = math.log(sum)
    // Underflow is okay here; we can treat a very small number as zero
    val logSum2 = if (logSum1.isNaN) 0.0 else logSum1
    val logSumExp = xMax + logSum2
    i = 0
    while (i < n) {
      val yi = math.exp(x(i) - logSumExp)
      // Underflow is okay here; with our formula we get at least one y(i) = 1;
      // other small values can be a "very small probability".
      y(i) = if (yi < VerySmallProbability || yi.isNaN) VerySmallProbability else yi
      i += 1
    }
    y
  }
  private val VerySmallProbability = 0.0001

  /** Predict the likelihoods of each class given the inputs.  */
  def predict(x: Array[Double]): Array[Double] = {
    predict(x, 0)
  }

  /**
    * Predict the likelihoods of each class given the inputs.
    *
    * @param idx Index within learning batch for storing the intermediate values.
    */
  private def predict(x: Array[Double], idx: Int): Array[Double] = {
    require(x.length == inputDataLength, s"Input length ${x.length} does not match the configured one: $inputDataLength")
    var i = 0
    while (i < inputDataLength) {
      input(idx)(i) = x(i)
      i += 1
    }
    predictWithBias(idx)
  }

  /**
    * Predict a batch of likelihoods of each class given the batch of inputs.
    *
    * TODO: Optimize the collection handling.
    **/
  private def predictBatch(xs: Traversable[(Array[Double], Array[Double])]): Unit = {
    var i = 0
    while (i < m) {
      xs.map(x => predict(x._1, i))
      i += 1
    }
  }

  /**
    * Softmax logistic regression uses a linear predictor function (a*x+b) as the first layer.
    * We use m+1 inputs with an additional control pseudo-input fixed at 1 to simplify the model for biases.
    * (see https://en.wikipedia.org/wiki/Logistic_regression#Setup)
    *
    * @param idx Index within learning batch for storing the intermediate values.
    */
  private def predictWithBias(idx: Int): Array[Double] = {
    input(idx)(inputDataLength) = 1.0  // constant pseudo-input for simplified handling of bias
    lineOut(idx) = MatrixOp.mulMatrixByColumnDouble(weights, input(idx), nClasses, nInputs)
    predicted(idx) = if (useStable) softmaxStable(lineOut(idx)) else softmax(lineOut(idx))
    predicted(idx)
  }

  /**
    * Gradients of the loss function used for backprop weight updates. See ref.
    *
    * We have nClasses gradient vectors of the form:
    *
    * grad(w(j)) = - (1/m) * sum(x * (target(j) - predicted(j))) + lambda * w(j)
    *   where j = 0...nClasses;
    *   sum is over a batch of m examples;
    *   w(j) = vector of weights for input parameter j;
    *   x = input(i) = input vector (iterates over a batch);
    *   target(j) = known value (0 or 1) indicating if input x belongs to class j (iterates over a batch);
    *   predicted(j) = predicted likelihood of "input x belongs to class j" (iterates over a batch);
    *   lambda > 0 is the weight decay parameter necessary for convergence.
    */
  def gradientsOfLoss(): Unit = {

    val sum = new Array[Double](nInputs)
    var j = 0
    while (j < nClasses) {
      val rowOffset = j * nInputs
      gradients(j) = weights.slice(rowOffset, rowOffset + nInputs)  // 1 * w(j) so far

      var k = 0
      while (k < nInputs) {
        gradients(j)(k) = Softmax.Lambda * gradients(j)(k)  // completing the lambda * w(j) part
        sum(k) = 0.0
        k += 1
      }

      var i = 0
      while (i < m) {
        val diff = target(i)(j) - predicted(i)(j)
        k = 0
        while (k < nInputs) {
          sum(k) += input(i)(k) * diff
          k += 1
        }
        i += 1
      }

      k = 0
      while (k < nInputs) {
        gradients(j)(k) -= sum(k) / m
        k += 1
      }

      j += 1
    }
  }

  /** Loss (cost) function. Only used for debugging, so no priority to optimize. See ref. */
  private def lossFunction = {
    val weightDecay = weights.map(x => x * x).sum
    val logLoss = - (for {
      i <- 0 until m
      j <- 0 until nClasses
    } yield {
      val logPredicted1 = math.log(predicted(i)(j))
      val logPredicted2 = if (logPredicted1.isNegInfinity) Double.MinValue else logPredicted1
      target(i)(j) * logPredicted2
    })
      .sum / m
    logLoss + weightDecay
  }

  /**
    * Update weights using backprop.
    */
  private def updateWeights() = {
    var i = 0
    var j = 0
    while (i < nClasses) {
      j = 0
      while (j < nInputs) {
        val rowOffset = i * nInputs
        val idx = rowOffset + j
        val gradElem = gradients(i)(j)
        historicalGrad(idx) += gradElem * gradElem  // AdaGrad
        val adaGradLearnRate = learnRate / (Softmax.AdaGradStabilityFactor + math.sqrt(historicalGrad(idx)))
        weights(idx) -= (adaGradLearnRate * gradElem)
        j += 1
      }
      i += 1
    }
  }
  private val historicalGrad = new Array[Double](weights.length)


  def learn(examples: Traversable[(Array[Double], Array[Double])]): (Softmax, Boolean) = {
    val (example, inTarget) = examples.head
    require(example.length == inputDataLength,
      s"Input length ${example.length} does not match the configured one: $inputDataLength")
    require(inTarget.length == nClasses,
      s"Target length ${inTarget.length} does not match the configured number of classes: $nClasses")

    m = examples.size  // it is not just for the loop, we will need it often
    var i = 0
    val ex = examples.toIterator
    while (i < m) {
      target(i) = ex.next._2
      i += 1
    }

    predictBatch(examples)

    gradientsOfLoss()
    logger.trace("Gradients: \n" + gradients.map(_.mkString(", ")).mkString("\n"))

    val currentLoss = lossFunction

    if (currentLoss < minLoss) {
      minLoss = currentLoss
      logger.trace(s"Loss function: $currentLoss")
      iterationsSinceLastImprovement = 0
    } else {
      iterationsSinceLastImprovement += 1
    }
    updateWeights()
    logger.trace("Updated weights: \n" + weights.grouped(nInputs).map(_.mkString(", ")).mkString("\n"))

    val shouldStop = iterationsSinceLastImprovement > stuckIterationLimit
    if (shouldStop) {
      logger.debug(s"No improvement in the last ${stuckIterationLimit} iterations," +
                   s" terminating early. Loss function: $currentLoss")
    }

    (this, shouldStop)
  }

  private var minLoss = Double.PositiveInfinity
  private var iterationsSinceLastImprovement = 0

  /**
    * Convenience shortcut for feeding a sequence of examples,
    * splitting it into suitable batches.
    */
  def learnSeq(examples: Iterable[(Array[Double], Array[Double])]): Softmax = {
    minLoss = Double.PositiveInfinity

    var i = 0
    val n = historicalGrad.length
    while (i < n) {
      historicalGrad(i) = 0.0
      i += 1
    }

    examples.grouped(batchSize).foldLeft(this) { (acc, ex) =>
      val (learned, shouldStop) = acc.learn(ex)
      if (shouldStop) return this
      else learned
    }
  }
}
object Softmax {
  /**
    * Creates a new Softmax object with given input/output sizes and random weights.
    *
    * @param seed Random seed. Use it to keep your tests stable.
    */
  def withRandomWeights(inputDataLength: Int,
                        nClasses: Int,
                        learnRate: Double,
                        stuckIterationLimit: Int,
                        seed: Option[Long] = None,
                        batchSize: Int = 1,
                        useStable: Boolean = true) = {
    val r = new scala.util.Random()
    seed.foreach(r.setSeed)
    // nInput + 1 for handling of bias (intercept)
    val weights = Array.fill((inputDataLength + 1) * nClasses)(r.nextDouble())
    new Softmax(weights, nClasses, inputDataLength, learnRate,
      stuckIterationLimit, batchSize, useStable)
  }

  val Lambda = 0.0001  // for convergence
  val AdaGradStabilityFactor = 0.000001  // for numerical stability
}