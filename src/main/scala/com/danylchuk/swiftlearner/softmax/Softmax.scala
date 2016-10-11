package com.danylchuk.swiftlearner.softmax

import com.danylchuk.swiftlearner.math.{ArrayOp, MatrixOp}
import com.danylchuk.swiftlearner.util._
import com.typesafe.scalalogging.LazyLogging


/**
  * Softmax (multinomial logistic) regression with SGD and AdaGrad
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
class Softmax(val weights: Array[Float],
              nClasses: Int,
              inputDataLength: Int,
              learnRate: Float,
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
  private[softmax] val target: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** current examples batch input[m][nInputs] */
  private[softmax] val input: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nInputs))

  /** lineOut[m][nClasses]: output of the first level linear predictor before the softmax layer */
  private[softmax] val lineOut: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** predicted[m][nClasses] = predict(input) */
  private[softmax] val predicted: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** gradients of the loss function */
  private[softmax] val gradients: Array[Array[Float]] = Array.fill(nClasses)(new Array[Float](nInputs))

  /**
    * y = softmax(x) = normalize(exp(x)) = exp(x(i)) / sum (exp(x))
    *
    * Naive "by the book" version; only works with normalized, stable input.
    *
    * @param idx The index of the currently processed example from the mini-batch.
    *            Used to save memory by writing the result directly to predicted(idx).
    */
  def softmax(x: Array[Float], idx: Int): Array[Float] = {
    val n = x.length
    var i = 0
    val y = predicted(idx)
    var sum: Float = 0.0f
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
    *
    * @param idx The index of the currently processed example from the mini-batch.
    *            Used to save memory by writing the result directly to predicted(idx).
    */
  def softmaxStable(x: Array[Float], idx: Int): Array[Float] = {
    val xMax = ArrayOp.max(x)  // for logSumExp
    val n = x.length
    var i = 0
    val y = predicted(idx)
    var sum: Float = 0.0f
    while (i < n) {
      sum += x(i) - xMax  // for logSumExp
      i += 1
    }
    val logSum1 = math.log(sum).toFloat
    // Underflow is okay here; we can treat a very small number as zero
    val logSum2 = if (java.lang.Float.isNaN(logSum1)) 0.0 else logSum1
    val logSumExp = xMax + logSum2
    i = 0
    while (i < n) {
      val yi = math.exp(x(i) - logSumExp).toFloat
      // Underflow is okay here; with our formula we get at least one y(i) = 1;
      // other small values can be a "very small probability".
      y(i) = if (yi < VerySmallProbability || java.lang.Float.isNaN(yi)) VerySmallProbability
             else yi
      i += 1
    }
    y
  }
  private val VerySmallProbability = 0.0001f

  /** Predict the likelihoods of each class given the inputs.  */
  def predict(x: Array[Float]): Array[Float] = {
    predict(x, 0)
  }

  /**
    * Predict the likelihoods of each class given the inputs.
    *
    * @param idx Index within learning batch for storing the intermediate values.
    */
  private def predict(x: Array[Float], idx: Int): Array[Float] = {
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
  private def predictBatch(xs: Traversable[(Array[Float], Array[Float])]): Unit = {
    var i = 0
    while (i < m) {
      xs.foreach(x => predict(x._1, i))
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
  private def predictWithBias(idx: Int): Array[Float] = {
    input(idx)(inputDataLength) = 1.0f  // constant pseudo-input for simplified handling of bias
    lineOut(idx) = MatrixOp.mulMatrixByColumnFloat(weights, input(idx), nClasses, nInputs)
    predicted(idx) = if (useStable) softmaxStable(lineOut(idx), idx) else softmax(lineOut(idx), idx)
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
    var j = 0
    while (j < nClasses) {
      val rowOffset = j * nInputs

      var k = 0
      val gradJ = gradients(j)
      while (k < nInputs) {
        gradJ(k) = Softmax.Lambda * weights(rowOffset + k)  // gradients(j) = lambda * w(j) part
        partialGradSum(k) = 0.0f
        k += 1
      }

      var i = 0
      while (i < m) {
        val diff = target(i)(j) - predicted(i)(j)
        k = 0
        while (k < nInputs) {
          partialGradSum(k) += input(i)(k) * diff
          k += 1
        }
        i += 1
      }

      k = 0
      while (k < nInputs) {
        gradJ(k) -= partialGradSum(k) / m
        k += 1
      }

      j += 1
    }
  }
  private val partialGradSum = new Array[Float](nInputs)

  /** Loss (cost) function. Used for tracking progress and stopping at a minimum. See ref. */
  private def lossFunction = {
    var weightDecay = 0.0f
    var i = 0
    val n = weights.length
    while (i < n) {
      val x = weights(i)
      weightDecay += x * x
      i += 1
    }

    var logLoss = 0.0f
    i = 0
    var j = 0
    while (i < m) {
      val predI = predicted(i)
      val targetI = target(i)
      while (j < nClasses) {
        val logPredicted1 = math.log(predI(j)).toFloat
        val logPredicted2 = if (Float.NegativeInfinity == logPredicted1) Float.MinValue
                            else logPredicted1
        logLoss -= targetI(j) * logPredicted2

        j += 1
      }
      i += 1
    }
    logLoss /= m

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
      val gradI = gradients(i)
      while (j < nInputs) {
        val rowOffset = i * nInputs
        val idx = rowOffset + j
        val gradElem = gradI(j)
        historicalGrad(idx) += gradElem * gradElem  // AdaGrad
        val adaGradLearnRate = learnRate / (Softmax.AdaGradStabilityFactor + math.sqrt(historicalGrad(idx)))
        weights(idx) -= (adaGradLearnRate * gradElem).toFloat
        j += 1
      }
      i += 1
    }
  }
  private val historicalGrad = new Array[Float](weights.length)

  def learn(examples: Traversable[(Array[Float], Array[Float])]): Softmax = {
    learnImpl(examples)
    this
  }

  private def learnImpl(examples: Traversable[(Array[Float], Array[Float])]): Boolean = {
    val (example, inTarget) = examples.head
    require(example.length == inputDataLength,
      s"Input length ${example.length} does not match the configured one: $inputDataLength")
    require(inTarget.length == nClasses,
      s"Target length ${inTarget.length} does not match the configured number of classes: $nClasses")

    this.m = examples.size

    var i = 0
    examples.foreach { ex =>
      target(i) = ex._2
      i += 1
    }

    predictBatch(examples)

    gradientsOfLoss()
    // logger.trace("Gradients: \n" + gradients.map(_.mkString(", ")).mkString("\n"))

    val currentLoss = lossFunction

    if (currentLoss < minLoss) {
      minLoss = currentLoss
      logger.trace(s"Loss function: $currentLoss")
      iterationsSinceLastImprovement = 0
    } else {
      iterationsSinceLastImprovement += 1
    }
    updateWeights()
    // logger.trace("Updated weights: \n" + weights.grouped(nInputs).map(_.mkString(", ")).mkString("\n"))

    val shouldStop = iterationsSinceLastImprovement > stuckIterationLimit
    if (shouldStop) {
      logger.debug(s"No improvement in the last $stuckIterationLimit iterations," +
                   s" terminating early. Loss function: $currentLoss")
    }
    shouldStop
  }

  private var minLoss = Float.PositiveInfinity
  private var iterationsSinceLastImprovement = 0

  /**
    * Convenience shortcut for feeding a sequence of examples,
    * splitting it into suitable batches.
    */
  def learnSeq(examples: Traversable[(Array[Float], Array[Float])]): Softmax = {
    minLoss = Float.PositiveInfinity

    var i = 0
    val n = historicalGrad.length
    while (i < n) {
      historicalGrad(i) = 0.0f
      i += 1
    }

    logger.info(s"Learning the training set: ${examples.size} entries")
    val nTotal = examples.size
    i = 0
    while (i < nTotal) {
      val until = math.min(nTotal, i + batchSize)
      val shouldStop = learnImpl(examples.view(i, until))
      if (shouldStop) return this
      i += batchSize
    }

    this
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
                        learnRate: Float,
                        stuckIterationLimit: Int,
                        seed: Option[Long] = None,
                        batchSize: Int = 1,
                        useStable: Boolean = true) = {
    val r = new scala.util.Random()
    seed.foreach(r.setSeed)
    // nInput + 1 for handling of bias (intercept)
    val weights = Array.fill((inputDataLength + 1) * nClasses)(r.nextFloat())
    new Softmax(weights, nClasses, inputDataLength, learnRate,
      stuckIterationLimit, batchSize, useStable)
  }

  val Lambda = 0.0001f  // for convergence
  val AdaGradStabilityFactor = 0.000001  // for numerical stability
}