package saiml.ga

import org.specs2.mutable.Specification


class GeneticTest extends Specification {
  "Genetic algorithm" should {
    "solve the Hello World example" >> {
      val result = new Genetic[HelloGenetic, String](50, 10).optimize(100, 30000L)
      result.genomeVal must_== HelloGenetic.Hello
      result.fitness must_== 0
    }

    "sort the flowers from the Fisher Iris dataset" >> {
      val testSet = GeneticIris.testSet

      val classifier = new Genetic[GeneticIris, Seq[Double]](100, 10)
        .optimize(200, 60000L).genomeVal

      val accuracy = (for ((species, params) <- testSet) yield {
        GeneticIris.predict(classifier, params) == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.7)  // 0.94 is typical
    }
  }
}