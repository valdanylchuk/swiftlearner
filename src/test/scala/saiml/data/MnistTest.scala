package saiml.data

import org.specs2.mutable.Specification


class MnistTest extends Specification {
  "MNIST dataset" should {
    "have the correct number of elements" >> {
//      Mnist.trainImagesByteArray.size must_== Mnist.TrainSetSize  // takes 18s
      Mnist.trainLabels.size must_== Mnist.TrainSetSize
      Mnist.testImages.size must_== Mnist.TestSetSize
      Mnist.testLabels.size must_== Mnist.TestSetSize
    }
  }
}