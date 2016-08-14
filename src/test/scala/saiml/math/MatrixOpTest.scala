package saiml.math

import org.specs2.mutable.Specification


class MatrixOpTest extends Specification {
  "MatrixOp" should {
    "mulMatrixByColumnFloat" >> {
      val a = Array(1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f, 99.99f, 99.99f, 99.99f)
      val x = Array(11.0f, 13.0f, 17.0f)
      val result = MatrixOp.mulMatrixByColumnFloat(a, x, 2, 3)
      val expected = Array(11.0f * 1.0f + 13.0f * 2.0f + 17.0f * 3.0f,
                           11.0f * 5.0f + 13.0f * 7.0f + 17.0f * 9.0f)
      result must_== expected
    }

    "mulMatrixByColumnDouble" >> {
      val a = Array(1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 99.99, 99.99, 99.99)
      val x = Array(11.0, 13.0, 17.0)
      val result = MatrixOp.mulMatrixByColumnDouble(a, x, 2, 3)
      val expected = Array(11.0 * 1.0 + 13.0 * 2.0 + 17.0 * 3.0,
                           11.0 * 5.0 + 13.0 * 7.0 + 17.0 * 9.0)
      result must_== expected
    }
  }
}