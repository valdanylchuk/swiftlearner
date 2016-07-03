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
}