package com.danylchuk.swiftlearner.math

object MatrixOp {
  /**
    * Multiply a matrix by a column vector: y = a*x
    *
    * @param a An Array which is logically a matrix, modeled as concatenated rows
    * @param x An input vector of size nColumns, which is logically a column
    * @param nRows Number of logical rows in matrix a
    * @param nColumns Number of logical columns in matrix a
    * @return The result y = a*x; logically a column vector of size nRows.
    */
  def mulMatrixByColumnFloat(a: Array[Float], x: Array[Float], nRows: Int, nColumns: Int): Array[Float] = {
    var i = 0
    var j = 0
    val result = new Array[Float](nRows)
    while (i < nRows) {
      result(i) = 0
      j = 0
      while (j < nColumns) {
        val rowOffset = i * nColumns
        result(i) += a(rowOffset + j) * x(j)
        j += 1
      }
      i += 1
    }
    result
  }

  /**
    * Multiply a matrix by a column vector: y = a*x
    *
    * @param a An Array which is logically a matrix, modeled as concatenated rows
    * @param x An input vector of size nColumns, which is logically a column
    * @param nRows Number of logical rows in matrix a
    * @param nColumns Number of logical columns in matrix a
    * @return The result y = a*x; logically a column vector of size nRows.
    */
  def mulMatrixByColumnDouble(a: Array[Double], x: Array[Double], nRows: Int, nColumns: Int): Array[Double] = {
    var i = 0
    var j = 0
    val result = new Array[Double](nRows)
    while (i < nRows) {
      result(i) = 0
      j = 0
      while (j < nColumns) {
        val rowOffset = i * nColumns
        result(i) += a(rowOffset + j) * x(j)
        j += 1
      }
      i += 1
    }
    result
  }
}