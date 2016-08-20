package com.danylchuk.swiftlearner.math

import scala.language.implicitConversions


class VectorOp(v: Vector[Double]) {
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

object VectorOp {
  implicit def vectorToVectorOp(v: Vector[Double]): VectorOp = new VectorOp(v)

  def squaredDistance(x: Vector[Double], y: Vector[Double]): Double =
    (for ((xi, yi) <- x zip y) yield { val d = xi - yi; d * d }).sum

  def squaredDistanceFloat(x: Vector[Float], y: Vector[Float]): Float =
    (for ((xi, yi) <- x zip y) yield { val d = xi - yi; d * d }).sum

  def distance(x: Vector[Double], y: Vector[Double]): Double =
    math.sqrt(squaredDistance(x, y))
}