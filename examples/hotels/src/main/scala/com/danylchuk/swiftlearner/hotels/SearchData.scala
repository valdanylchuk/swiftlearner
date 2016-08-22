package com.danylchuk.swiftlearner.hotels

import java.io.{BufferedInputStream, InputStream}
import java.util.zip.GZIPInputStream

import scala.collection.mutable.{Map => MutableMap}
import scala.io.Source


/**
  * Reads and pre-processes a subset of the original competition data.
  * See scripts/filter.pl for the selection details.
  */
object SearchData {
  val TrainSetSize = 20000
  val TestSetSize = 4000

  /**
    * Training data as flat vectors with one-hot encoding for categorical inputs.
    *
    * Using Iterator[Float] as a temporary sparse vector.
    */
  def trainDataEncoded: Iterator[Iterator[Float]] = {
    encode(trainDataIdMapped.iterator)
  }

  /**
    * Test data as flat vectors with one-hot encoding for categorical inputs.
    *
    * Using Iterator[Float] as a temporary sparse vector.
    */
  def testDataEncoded: Iterator[Iterator[Float]] = {
    encode(testDataIdMapped.iterator)
  }

  /** Labels corresponding to the training data */
  lazy val trainLabels: Stream[Int] = readLabels(trainLabelsFile).toStream

  /** Labels corresponding to the test data */
  lazy val testLabels: Stream[Int] = readLabels(testLabelsFile).toStream

  def labeledTrainIterator = trainLabels.iterator zip trainDataEncoded
  def labeledTestIterator = testLabels.iterator zip testDataEncoded

  /** Convenience shortcut */
  def trainingAndTestData(nTrainPoints: Int = TrainSetSize)
  : (Seq[(Int, Iterator[Float])], Seq[(Int, Iterator[Float])]) = {
    val labeledTrainSet = labeledTrainIterator take nTrainPoints
    val labeledTestSet = labeledTestIterator take nTrainPoints
    (labeledTrainSet.toSeq, labeledTestSet.toSeq)
  }

  private def encode(dataIdMapped: Iterator[SearchRecord]): Iterator[Iterator[Float]] = {
    dataIdMapped.map { record =>
      val distance = Iterator((record.distance / maxDistance - meanNormDistance).toFloat)
      val userCity = oneHotEncode(record.userCity, cityIds.size)
      val dest = oneHotEncode(record.dest, destIds.size)
      distance ++ userCity ++ dest
    }
  }

  /**
    * @param x value to encode
    * @param n total number of combinations / highest possible x
    */
  private def oneHotEncode(x: Int, n: Int): Iterator[Float] = {
    Iterator.tabulate(n) { i =>
      if (i == x) 1.0f else 0.0f
    }
  }

  /**
    * Map big IDs to smaller values before one-hot encoding.
    * Let's keep 0 for special use later on.
    * Making it a Vector because this will be our long-term storage,
    * and because we need fully populated ID maps for the next step.
    */
  private lazy val trainDataIdMapped: Vector[SearchRecord] = {
    var cityId = 0
    var destId = 0
    trainDataTyped.map { record =>
      // help normalize the distance while we are iterating here
      distanceSum += record.distance
      if (record.distance > maxDistance)
        maxDistance = record.distance

      val userCity = cityIds.getOrElseUpdate(record.userCity, {cityId += 1; cityId})
      val dest = destIds.getOrElseUpdate(record.dest, {destId += 1; destId})
      SearchRecord(userCity, record.distance, dest)
    }.toVector
  }
  private val cityIds = MutableMap.empty[Int, Int]
  private val destIds = MutableMap.empty[Int, Int]
  private var distanceSum = 0.0
  private var maxDistance = 0.0
  private lazy val meanNormDistance = distanceSum / TrainSetSize / maxDistance

  /**
    * Map big IDs to smaller values before one-hot encoding.
    * Use 0 for values not encountered in training data.
    */
  private lazy val testDataIdMapped: Vector[SearchRecord] = {
    testDataTyped.map { record =>
      val userCity = cityIds.getOrElse(record.userCity, 0)
      val dest = destIds.getOrElse(record.dest, 0)
      SearchRecord(userCity, record.distance, dest)
    }.toVector
  }

  private lazy val trainDataTyped: Iterator[SearchRecord] = readData(trainDataFile)

  private lazy val testDataTyped: Iterator[SearchRecord] = readData(testDataFile)

  private lazy val trainDataFile = getFileStream("train-data.csv.gz")
  private lazy val trainLabelsFile = getFileStream("train-labels.csv.gz")
  private lazy val testDataFile = getFileStream("test-data.csv.gz")
  private lazy val testLabelsFile = getFileStream("test-labels.csv.gz")

  private def getFileStream(name: String): InputStream = {
    new BufferedInputStream(new GZIPInputStream(
      this.getClass.getClassLoader.getResourceAsStream(name)))
  }

  private def readLabels(stream: InputStream): Iterator[Int] =
    Source.fromInputStream(stream, "UTF8").getLines.map(_.toInt)

  private def readData(stream: InputStream): Iterator[SearchRecord] = {
    Source.fromInputStream(stream, "UTF8").getLines.map(SearchRecord.fromString)
  }
}

case class SearchRecord(userCity: Int, distance: Double, dest: Int)
object SearchRecord {
  def fromString(s: String) = {
    val fields = s.split(',')
    val userCity = fields(0).toInt
    val distance = fields(1).toDouble
    val dest = fields(2).toInt
    SearchRecord(userCity, distance, dest)
  }
}