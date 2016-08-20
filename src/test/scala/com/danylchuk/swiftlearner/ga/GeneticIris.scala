package com.danylchuk.swiftlearner.ga

import com.danylchuk.swiftlearner.data.FisherIris
import com.danylchuk.swiftlearner.math.VectorOp

import scala.util.Random


/**
  * Example user domain implementation of Individual.
  *
  * Usage: val result = new Genetic[GeneticIris](50, 10).optimize(100, 10000L)
  *
  * Ref.: http://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
  */
class GeneticIris(override val genome: Option[Seq[Double]]) extends Individual[GeneticIris, Seq[Double]] {
  val genomeVal = genome.get

  /** Random default constructor */
  def this() = this(Some(Seq.fill(GeneticIris.genomeLength)(Random.nextDouble * 6)))

  /** Fitness function: sum of differences, squared to penalize severe defects */
  override val fitness: Double = {
    val trainingSet = GeneticIris.trainingSet
    val diffs = for ((classIdx, params) <- GeneticIris.trainingSet) yield {
      val classGenes = genomeVal.grouped(4).toVector(classIdx)
      VectorOp.squaredDistance(params, classGenes.toVector)
    }
    diffs.sum
  }

  /** Crossover: take the head of one genome and the tail of another */
  override def crossover(that: GeneticIris): GeneticIris = {
    val pos = Random.nextInt(GeneticIris.genomeLength)
    val newGenome = this.genomeVal.take(pos) ++ that.genomeVal.drop(pos)
    new GeneticIris(Some(newGenome))
  }

  /** Mutate: slightly modify at a random location */
  override def mutate(): GeneticIris = {
    val pos = Random.nextInt(GeneticIris.genomeLength)
    val change = Random.nextDouble() * 0.2 + 0.9  // [-10% .. +10%]
    val newGenome = this.genomeVal.updated(pos, this.genomeVal(pos) * change)
    new GeneticIris(Some(newGenome))
  }
}
object GeneticIris {
  // We need the training set here for use in the fitness function.
  // We also store the corresponding test set for use in the tests,
  // because FisherIris splits the data randomly every time.
  lazy val (trainingSet, testSet) = FisherIris.trainingAndTestData(Some(0L))

  val genomeLength = 12  // 3 classes * 4 params

  /** Predicts the most likely class given the parameters. */
  def predict(genome: Seq[Double], params: Seq[Double]): Int = {
    val diffs = for (classIdx <- 0 to 2) yield {
      val classGenes = genome.grouped(4).toSeq(classIdx)
      params.zip(classGenes).map(Function.tupled(_ - _)).map(x => x * x).sum
    }
    diffs.zipWithIndex.minBy(_._1)._2
  }

}