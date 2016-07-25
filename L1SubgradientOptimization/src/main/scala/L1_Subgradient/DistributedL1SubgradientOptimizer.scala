package main.scala.L1_Subgradient

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.log4j.Level

/**
  * Created by nico on 25.07.2016.
  */

object DistributedL1SubgradientOptimizer {
  def main(args: Array[String]): Unit = {
    val noIter = 100
    val noFeatures = 1000


    // initialize spark
    val conf = new SparkConf()
      .setAppName("Distributed L1 Subgradient Optimizer")
      .setMaster("local[8]")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // sample X, Y
    val r = scala.util.Random
    r.setSeed(1003010L)
    val X = sc.parallelize(Array.fill(noFeatures)(2 * r.nextDouble() - 1))
    val bAst = r.nextDouble()
    val Y = X.map(xi => bAst * xi)

    // sample b0
    val b0 = r.nextDouble()

    // initialize optimizer
    val optimizer = new DistributedL1SubgradientOptimizer(X, Y)
    var bnext = b0
    println("0 f(" + bnext + ")=" + optimizer.functionValue(bnext))
    var min_f = optimizer.functionValue(bnext)
    var min_b = bnext

    // iterate
    for (i <- 1 to noIter) {
      val ak: Double = 1. / i // diminishing step size rule
      val bold = bnext
      bnext = optimizer.step(bnext, ak)
      val fb = optimizer.functionValue(bnext)
      if (i % 10 == 0) {
        println(i + " " + ak + " " + Math.sqrt(Math.pow(bnext - bold, 2)) + " => f(" + bnext + ")=" + fb)
      }
      if (fb < min_f) {
        min_f = fb
        min_b = bnext
      }
    }
    // print final result
    println("\n***")
    println("* " + " f*(" + bAst + ")=" + optimizer.functionValue(bAst))
    println("* " + " f(" + min_b + ")=" + min_f)
  }
}

class DistributedL1SubgradientOptimizer(X: RDD[Double], Y: RDD[Double]) {
  val XzY = X zip Y

  // single iteration
  def step(bk: Double, ak: Double): Double = {
    bk - ak * subgradient(bk).sum
  }

  // compute subgradient of
  // f(b) = ||b*X - Y||_1 wrt. b
  def subgradient(b: Double): RDD[Double] = {
    XzY.map( xy => {
      if( (b*xy._1 - xy._2) < 0) {
        -1 * xy._1
      } else {
        xy._1
      }
    } )
  }

  // compute function value of
  // f(b) = ||b*X - Y||_1
  def functionValue(b: Double): Double = {
    XzY.map{ case (xi: Double,yi: Double) => Math.abs(b*xi - yi) }.sum

  }
}
