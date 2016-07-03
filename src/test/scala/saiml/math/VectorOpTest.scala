package saiml.math

import org.specs2.mutable.Specification
import saiml.math.VectorOp._


class VectorOpTest extends Specification {
  "VectorOp" should {
    "calculate dot product of two vectors" >> {
      Vector(2.0, 3.0) * Vector(4.0, 5.0) must_== 2 * 4 + 3 * 5
    }

    "throw IllegalArgumentException on non-matching size" >> {
      { Vector(2.0, 3.0) * Vector(4.0) } must throwA(
        new IllegalArgumentException("requirement failed: vector size mismatch"))
    }
  }
}