package saiml.ga

import org.specs2.mutable.Specification


class GeneticTest extends Specification {
  "Genetic algorithm" should {
    "solve the Hello World example" >> {
      val result = new Genetic[HelloGenetic](50, 10).optimize(100, 30000L)
      result.genome must_== HelloGenetic.Hello
      result.fitness must_== 0
    }
  }
}