package com.danylchuk.swiftlearner.hotels

import org.specs2.matcher.DataTables
import org.specs2.mutable.Specification


class SearchDataTest extends Specification with DataTables {
  "Hotels dataset" should {
    "have the correct number of elements" >> {
      SearchData.trainDataEncoded.size must_== SearchData.TrainSetSize
      SearchData.trainLabels.size must_== SearchData.TrainSetSize
      SearchData.testDataEncoded.size must_== SearchData.TestSetSize
      SearchData.testLabels.size must_== SearchData.TestSetSize
    }

    "have primitives in train data" >> {
      "elementClass" |
      SearchData.trainDataEncoded.next().next().getClass |
      SearchData.trainingAndTestData()._1.head._2.next().getClass |> { elementClass =>
        elementClass.isPrimitive must beTrue
      }
    }
  }
}