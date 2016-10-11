package com.danylchuk.swiftlearner.math

object ArrayOp {
  def dot(a: Array[Float], b: Array[Float]): Float = {
    require(a.length == b.length, "array size mismatch")
    val n = a.length
    var sum: Float = 0f
    var i = 0
    while (i < n) {
      sum += a(i) * b(i)
      i += 1
    }
    sum
  }

  def indexOfMax(xs: Array[Float]): Int = {
    var max = xs(0)
    var idxMax = 0
    var i = 1
    while (i < xs.length) {
      if (xs(i) > max) {
        max = xs(i); idxMax = i
      }
      i += 1
    }
    idxMax
  }

  def indexOfMax(xs: Array[Double]): Int = {
    var max = xs(0)
    var idxMax = 0
    var i = 1
    while (i < xs.length) {
      if (xs(i) > max) {
        max = xs(i); idxMax = i
      }
      i += 1
    }
    idxMax
  }

  def max(xs: Array[Double]): Double = {
    var max = xs(0)
    var i = 1
    while (i < xs.length) {
      if (xs(i) > max) {
        max = xs(i)
      }
      i += 1
    }
    max
  }

  def max(xs: Array[Float]): Float = {
    var max = xs(0)
    var i = 1
    while (i < xs.length) {
      if (xs(i) > max) {
        max = xs(i)
      }
      i += 1
    }
    max
  }
}