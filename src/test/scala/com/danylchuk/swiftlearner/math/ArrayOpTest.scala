package com.danylchuk.swiftlearner.math

import com.danylchuk.swiftlearner.MemoryTesting
import org.specs2.mutable.Specification


class ArrayOpTest extends Specification with MemoryTesting {

  def dot(a: Array[Float], b: Array[Float]): Float = {
    //require(a.length == b.length, "array size mismatch")
    val n = a.length
    var sum: Float = 0f
    var i = 0
    while (i < n) {
      sum += a(i) * b(i)
      i += 1
    }
    sum
  }

  "ArrayOp" should {
    "calculate dot product of two vectors" >> {
      ArrayOp.dot(Array(2.0f, 3.0f), Array(4.0f, 5.0f)) must_== 2 * 4 + 3 * 5
    }

    "find indexOfMax" >> {
      val v1 = Array(0.3f, 0.9f, 0.4f)
      val v2 = Array(0.3f, 0.9f, 1.4f)
      val v3 = Array(2.3f, 0.9f, 1.4f)
      ArrayOp.indexOfMax(v1) must_== 1
      ArrayOp.indexOfMax(v2) must_== 2
      ArrayOp.indexOfMax(v3) must_== 0
    }

    "use memory sparingly in dot" >> {
      val vector = Array.tabulate(100)(_.toFloat)
      countAllocatedRepeat(1000) {
        dot(vector, vector)
      } must_== 0L
    }
  }
}