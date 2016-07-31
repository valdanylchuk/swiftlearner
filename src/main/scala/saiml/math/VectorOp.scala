package saiml.math

import scala.language.implicitConversions


// It is more efficient to write Double and Float versions than a generic one here.
class VectorOpDouble(v: Vector[Double]) {
  /** dot product */
  def *(that: IndexedSeq[Double]): Double = {
    require(v.size == that.size, "vector size mismatch")
    val n = v.size
    var sum: Double = 0f
    var i = 0
    while (i < n) {
      sum += v(i) * that(i)
      i += 1
    }
    sum
  }

  /** subtract a vector */
  def -(that: Vector[Double]): Vector[Double] = {
    require(v.size == that.size, "vector size mismatch")
    for ((a, b) <- v zip that) yield a - b
  }

  /** add vectors */
  def +(that: Vector[Double]): Vector[Double] = {
    require(v.size == that.size, "vector size mismatch")
    for ((a, b) <- v zip that) yield a + b
  }
}

class VectorOpFloat(v: Vector[Float]) {
  /** dot product */
  def *(that: IndexedSeq[Float]): Float = {
    require(v.size == that.size, "vector size mismatch")
    val n = v.size
    var sum: Float = 0f
    var i = 0
    while (i < n) {
      sum += v(i) * that(i)
      i += 1
    }
    sum
  }

  /** subtract a vector */
  def -(that: Vector[Float]): Vector[Float] = {
    require(v.size == that.size, "vector size mismatch")
    for ((a, b) <- v zip that) yield a - b
  }

  /** add vectors */
  def +(that: Vector[Float]): Vector[Float] = {
    require(v.size == that.size, "vector size mismatch")
    for ((a, b) <- v zip that) yield a + b
  }
}

object VectorOp {
  implicit def vectorToVectorOpDouble(v: Vector[Double])
  : VectorOpDouble = new VectorOpDouble(v)

  implicit def vectorToVectorOpFloat(v: Vector[Float])
  : VectorOpFloat = new VectorOpFloat(v)

  def squaredDistance(x: Vector[Double], y: Vector[Double]): Double =
    (for ((xi, yi) <- x zip y) yield { val d = xi - yi; d * d }).sum

  def squaredDistanceFloat(x: Vector[Float], y: Vector[Float]): Float =
    (for ((xi, yi) <- x zip y) yield { val d = xi - yi; d * d }).sum

  def distance(x: Vector[Double], y: Vector[Double]): Double =
    math.sqrt(squaredDistance(x, y))

  def dot(a: Vector[Float], b: Array[Float]): Float = {
    require(a.length == b.length, "vector size mismatch")
    val n = a.length
    var sum: Float = 0f
    var i = 0
    while (i < n) {
      sum += a(i) * b(i)
      i += 1
    }
    sum
  }
}