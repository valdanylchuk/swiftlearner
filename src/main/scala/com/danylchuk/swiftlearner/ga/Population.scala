package com.danylchuk.swiftlearner.ga

import com.danylchuk.swiftlearner.coll.SeqOp._
import scala.reflect.ClassTag

/** Selects parents and evolves the next generation */
class Population[A <: Individual[A] :ClassTag](
  val size: Int,
  tournamentSize: Int,
  givenIndividuals: Option[Vector[A]] = None
) {
  val individuals: Vector[A] = givenIndividuals getOrElse
    Vector.tabulate(size)(_ => implicitly[ClassTag[A]].runtimeClass.newInstance.asInstanceOf[A])

  def getFittest: A = individuals.minBy(_.fitness)

  /** Tournament selection: pick the fittest one out of a random sample */
  def tournamentSelect(): A = individuals.randomSample(tournamentSize).minBy(_.fitness)

  def evolve: Population[A] = {
    val nextGen = (1 until size).map { _ =>  // skip one for the elite
      val parent1 = tournamentSelect()
      val parent2 = tournamentSelect()
      parent1.crossover(parent2).mutate()
    }
    val withElite = (getFittest +: nextGen).toVector
    new Population(size, tournamentSize, Some(withElite))
  }
}