package saiml.data

import java.util.zip.GZIPInputStream


/**
  * The classic MNIST handwritten digits dataset.
  *
  * Ref.: http://yann.lecun.com/exdb/mnist/
  *       https://en.wikipedia.org/wiki/MNIST_database
  */
object Mnist {
  /** Training images as flattened vectors (of concatenated lines) */
  def trainImages: Iterator[Vector[Float]] =
    trainImagesByteArray.iterator.map(_.toVector.map(_.toFloat))

  lazy val trainImagesByteArray: Array[Array[Byte]] = readImages(trainImagesFile)

  /** Labels corresponding to the training images */
  lazy val trainLabels: Stream[Int] = readLabels(trainLabelsFile)

  /** Test images as flattened vectors (of concatenated lines) */
  def testImages: Iterator[Vector[Float]] =
    testImagesByteArray.iterator.map(_.toVector).map(_.map(_.toFloat))

  lazy val testImagesByteArray: Array[Array[Byte]] = readImages(testImagesFile)

  /** Labels corresponding to the test images */
  lazy val testLabels: Stream[Int] = readLabels(testLabelsFile)

  /** Convenience shortcut */
  def trainingAndTestData(nTrainPoints: Int = TrainSetSize)
  : (Seq[(Int, Vector[Float])], Seq[(Int, Vector[Float])]) = {
    val labeledTrainSet = trainLabels.iterator zip trainImages take nTrainPoints
    val labeledTestSet = testLabels.iterator zip testImages take nTrainPoints
    (labeledTrainSet.toSeq, labeledTestSet.toSeq)
  }

  val ImageWidth = 28
  val ImageHeight = 28
  val ImageSize = ImageWidth * ImageHeight
  val TrainSetSize = 60000
  val TestSetSize = 10000

  private val ImagesOffset = 16 // magic number + set size + image dimensions
  private val LabelsOffset = 8  // magic number + set size

  private lazy val trainImagesFile = getFileStream("train-images-idx3-ubyte")
  private lazy val trainLabelsFile = getFileStream("train-labels-idx1-ubyte")
  private lazy val testImagesFile = getFileStream("t10k-images-idx3-ubyte")
  private lazy val testLabelsFile = getFileStream("t10k-labels-idx1-ubyte")

  private def getFileStream(name: String): Iterator[Int] = {
    val inStream = new GZIPInputStream(this.getClass.getClassLoader
      .getResourceAsStream(s"mnist/$name.gz"))

    Iterator.continually(inStream.read()).takeWhile(-1 != _)
  }

  private def readLabels(bytes: Iterator[Int]): Stream[Int] =
    bytes.drop(LabelsOffset).toStream

  /** @return Images as flattened vectors (of concatenated lines) */
  private def readImages(bytes: Iterator[Int]): Array[Array[Byte]] = {
    bytes.drop(ImagesOffset)  // magic number + set size
      .map(_.toByte).grouped(ImageSize).map(_.toArray).toArray
  }
}