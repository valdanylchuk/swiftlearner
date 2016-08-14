package saiml.softmax

import com.typesafe.scalalogging.LazyLogging
import saiml.math.{ArrayOp, MatrixOp}


/**
  * Softmax (multinomial logistic) regression
  *
  * Ref: https://en.wikipedia.org/wiki/Multinomial_logistic_regression
  * http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
  * http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
  *
  * @param weights initial weights (pre-trained or random)
  *   weights(i)(j) is the weight for evidence i based on input parameter j
  *     i = 0..nClasses
  *     j = 0..nInputs (where nInputs = inputLength + 1 for pseudo-input used for bias handling)
  *   Modeled as an array of concatenated rows.
  */
class Softmax(weights: Array[Float], nClasses: Int, inputDataLength: Int, learnRate: Float, batchSize: Int)
  extends LazyLogging {

  val nInputs = inputDataLength + 1  // with an additional control pseudo-input fixed at 1

  /** Current batch size */
  private var m = 0

  /**
    * Target "true" distribution of nClasses output values for supervised learning.
    * Use one-hot encoding: all zeroes except 1.0 for the target class.
    */
  private val target: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** current examples batch input[m][nInputs] */
  private val input: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nInputs))

  /** lineOut[m][nClasses]: output of the first level linear predictor before the softmax layer */
  private val lineOut: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** predicted[m][nClasses] = predict(input) */
  private val predicted: Array[Array[Float]] = Array.fill(batchSize)(new Array[Float](nClasses))

  /** gradients of the loss function */
  private val gradients: Array[Array[Float]] = Array.fill(nClasses)(new Array[Float](nInputs))

  /** y = softmax(x) = normalize(exp(x)) */
  def softmax(x: Array[Float]): Array[Float] = {
    val n = x.length
    var i = 0
    val y = new Array[Float](n)
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
  private def predictWithBias(idx: Int): Array[Float] = {
    input(idx)(inputDataLength) = 1.0f  // constant pseudo-input for simplified handling of bias
    lineOut(idx) = MatrixOp.mulMatrixByColumnFloat(weights, input(idx), nClasses, nInputs)
    predicted(idx) = softmax(lineOut(idx))
    predicted(idx)
  }

  /**
    * Gradients of the loss function used for backprop weight updates. See ref.
    *
    * We have nClasses gradient vectors of the form:
    *
    * grad(w(j)) = - (1/m) * sum(x * (target(j) - predicted(j))) + lam * w(j)
    *   where j = 0...nClasses;
    *   sum is over a batch of m examples;
    *   w(j) = vector of weights for input parameter j;
    *   x = input(i) = input vector (iterates over a batch);
    *   target(j) = known value (0 or 1) indicating if input x belongs to class j (iterates over a batch);
    *   predicted(j) = predicted likelihood of "input x belongs to class j" (iterates over a batch);
    *   lam > 0 is the weight decay parameter necessary for convergence (let lam = 1).
    */
  def gradientsOfLoss(): Unit = {

    val sum = new Array[Float](nInputs)
    var j = 0
    while (j < nClasses) {
      val rowOffset = j * nInputs
      gradients(j) = weights.slice(rowOffset, rowOffset + nInputs)  // lam * w(j)

      var k = 0
      while (k < nInputs) {
        sum(k) = 0.0f
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
    } yield target(i)(j) * math.log(predicted(i)(j)))
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
        weights(rowOffset + j) -= learnRate * gradients(i)(j)
        j += 1
      }
      i += 1
    }
  }

  def learn(examples: Traversable[(Array[Float], Array[Float])]): Softmax = {
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

    updateWeights()
    logger.trace("Updated weights: \n" + weights.grouped(nInputs).map(_.mkString(", ")).mkString("\n"))

    logger.trace(s"Loss function: $lossFunction")

    this
  }

  /**
    * Convenience shortcut for feeding a sequence of examples,
    * splitting it into suitable batches.
    */
  def learnSeq(examples: Iterable[(Array[Float], Array[Float])]): Softmax = {
    examples.grouped(batchSize).foldLeft(this) { (acc, ex) =>
      acc.learn(ex)
    }
  }
}
object Softmax {
  /**
    * Creates a new Softmax object with given input/output sizes and random weights.
    *
    * @param seed Random seed. Use it to keep your tests stable.
    **/
  def withRandomWeights(inputDataLength: Int,
                        nClasses: Int,
                        learnRate: Float,
                        seed: Option[Long] = None,
                        batchSize: Int) = {
    val r = new scala.util.Random()
    seed.foreach(r.setSeed)
    // nInput + 1 for simplified handling of bias
    val weights = Array.fill((inputDataLength + 1) * nClasses)(r.nextFloat())
    new Softmax(weights, nClasses, inputDataLength, learnRate, batchSize)
  }
}