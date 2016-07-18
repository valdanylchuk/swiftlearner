package saiml.ga

import scala.reflect.ClassTag


/**
  * Genetic Algorithm with elitist tournament selection.
  *
  * Ref.: https://en.wikipedia.org/wiki/Genetic_algorithm
  * http://www.theprojectspot.com/tutorial-post/creating-a-genetic-algorithm-for-beginners/3
  */
class Genetic[A <: Individual[A, B] :ClassTag, B](populationSize: Int, tournamentSize: Int) {
  def optimize(maxGen: Int, maxMillis: Long): A = {
    val first = new Population[A, B](populationSize, tournamentSize)
    val start = System.currentTimeMillis
    val optPop = (0 until maxGen).foldLeft(first) { (pop, _) =>
      if (pop.getFittest.fitness == 0) pop  // Found a perfect match
      else if (System.currentTimeMillis - start > maxMillis) pop  // Timed out
      else pop.evolve  // Otherwise, continue until max generations reached
    }
    optPop.getFittest
  }
}