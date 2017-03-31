package com.danylchuk.swiftlearner.math

import scala.annotation.tailrec

object ArrayOp {
  def dot(a: Array[Float], b: Array[Float]): Float = {
    require(a.length == b.length, "array size mismatch")

    @tailrec def sumProducts(idx: Int, sum: Float): Float = {
      if (idx < 0) sum
      else sumProducts(idx - 1, sum + a(idx) * b(idx))
    }

    sumProducts(a.length - 1, 0)
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