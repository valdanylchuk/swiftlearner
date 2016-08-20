package com.danylchuk.swiftlearner.math

import org.specs2.mutable.Specification


class ArrayOpTest extends Specification {
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
  }
}