package saiml.data

import org.specs2.matcher.DataTables
import org.specs2.mutable.Specification


class MnistTest extends Specification with DataTables {
  "MNIST dataset" should {
    "have the correct number of elements" >> {
      Mnist.trainImagesByteArray.size must_== Mnist.TrainSetSize
      Mnist.trainLabels.size must_== Mnist.TrainSetSize
      Mnist.testImages.size must_== Mnist.TestSetSize
      Mnist.testLabels.size must_== Mnist.TestSetSize
    }

    "have primitives in trainImages" >> {
      "elementClass" |
      Mnist.trainImagesByteArray(0)(0).getClass |
      Mnist.trainImages.next()(0).getClass |
      Mnist.trainImagesDouble.next()(0).getClass |
      Mnist.trainingAndTestDataDouble()._1.head._2.head.getClass |> { elementClass =>
        elementClass.isPrimitive must beTrue
      }
    }
  }
}