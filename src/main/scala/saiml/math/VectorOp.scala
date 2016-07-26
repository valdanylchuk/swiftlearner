package saiml.math

import scala.language.implicitConversions


class VectorOp(v: Vector[Double]) {
  /** dot product */
  def *(that: Vector[Double]): Double = {
    require(v.size == that.size, "vector size mismatch")
    (for ((a, b) <- v zip that) yield a * b).sum
  }

  def -(that: Vector[Double]): Vector[Double] = {
    require(v.size == that.size, "vector size mismatch")
    for ((a, b) <- v zip that) yield a - b
  }
}
object VectorOp {
  implicit def vectorToVectorMul(v: Vector[Double]): VectorOp = new VectorOp(v)

  def squaredDistance(x: Vector[Double], y: Vector[Double]) =
    (for ((xi, yi) <- x zip y) yield { val d = xi - yi; d * d }).sum

  def distance(x: Vector[Double], y: Vector[Double]) =
    math.sqrt(squaredDistance(x, y))
}