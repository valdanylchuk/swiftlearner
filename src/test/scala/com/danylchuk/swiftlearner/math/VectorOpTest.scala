package com.danylchuk.swiftlearner.math

import com.danylchuk.swiftlearner.math.VectorOp._
import org.specs2.mutable.Specification


class VectorOpTest extends Specification {
  "VectorOp dot product" should {
    "calculate dot product of two vectors" >> {
      Vector(2.0, 3.0) * Vector(4.0, 5.0) must_== 2 * 4 + 3 * 5
    }
    "throw IllegalArgumentException on non-matching size" >> {
      {
        Vector(2.0, 3.0) * Vector(4.0)
      } must throwA(
        new IllegalArgumentException("requirement failed: vector size mismatch"))
    }
  }

  "VectorOp.distance" should {
    "calculate the distance correctly" >> {
      distance(Vector(0, 0), Vector(10, 0)) must_== 10
      distance(Vector(0, 0), Vector(0, 10)) must_== 10
      distance(Vector(0, 0), Vector(3, 4)) must_== 5
      distance(Vector(0, 0), Vector(4, 3)) must_== 5
      distance(Vector(3, 0), Vector(0, 4)) must_== 5
    }
  }
}