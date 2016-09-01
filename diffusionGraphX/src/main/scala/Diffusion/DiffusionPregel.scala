package Diffusion

import java.security.InvalidParameterException

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx._
import org.apache.spark.{Accumulator, AccumulatorParam, SparkConf, SparkContext}
import org.jblas.DoubleMatrix

import scala.collection.mutable


/**
  * reduction of Min-Sum Diffusion algorithm to Pregel/Pagerank Distributed Computation Pattern
  *
  * @author Nico Hoffmann, Benjamin Naujoks
  */

object DiffusionPregel {

  def main(args: Array[String]): Unit = {
    // toggle debug output
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf()
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setAppName("PREGEL Min-Sum Diffusion")
      .setMaster(args(2))
    val sc = new SparkContext(conf)
    val benchmark = args(0)
    println("Dataset: " + benchmark)

    // load edge data
    val pwPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/pwFactors.csv")
    val noLabelsOfEachVertex: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/nrLabels.csv")
    val unaryPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/uFactors.csv")
    val gridWidth = scala.io.Source.fromFile("benchmark/" + benchmark + "/lcid.txt").getLines().next().toInt

    // create graph structure
    val graph = GraphLoader.edgeListFile(sc, "benchmark/" + benchmark + "/edgeListFile.txt", false, args(1).toInt).cache()

    // initialize diffusion data structures
    val noIterPerBndComputation = 100
    val noIter = noIterPerBndComputation
    val conv = 1e-4
    val boundArray = sc.accumulator[Array[Double]](Array.ofDim[Double](Math.floor((noIter.toDouble - 1) / 2).toInt))(new DoubleArrayAccumulator)
    val energyArray = sc.accumulator[Array[Double]](Array.ofDim[Double](Math.floor((noIter.toDouble - 1) / 2).toInt))(new DoubleArrayAccumulator)

    // run distributed inference algorithm
    val diffInference = new DiffusionPregel(graph, noLabelsOfEachVertex, unaryPotentials, pwPotentials, gridWidth, boundArray, energyArray, noIter, noIterPerBndComputation, conv)
    diffInference.apply()
  }
}

class DiffusionPregel(graph: Graph[Int, Int], noLabelsOfEachVertex: DoubleMatrix, unaryPotentials: DoubleMatrix, pwPotentials: DoubleMatrix, lastColumnId: Integer, boundArray: Accumulator[Array[Double]], energyArray: Accumulator[Array[Double]], noIter: Int, noIterPerBndComputation: Int, conv: Double) extends java.io.Serializable {
  if (noIterPerBndComputation % 2 == 1)
    throw new InvalidParameterException("due to the parallelization strategy **noIterPerBndComputation** must be set to any even number")

  val SHOW_LABELING_IN_ITERATION: Boolean = false
  val t_final = System.currentTimeMillis()

  def apply() = {

    var bound = 0.0

    // create hashmap of vertex data (unary factors)
    var g_t = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
    var sel = 0
    for (vertexId <- 0 to noLabelsOfEachVertex.rows - 1) {
      val noLabels = noLabelsOfEachVertex.get(vertexId)

      //TODO WHY DO WE ACTUALLY NEED THIS CONTAINER?
      var g_t_xt: DoubleMatrix = DoubleMatrix.zeros(noLabels.toInt)

      //TODO WHY DO WE ACTUALLY NEED THIS CONTAINER?
      // construct g_t_xt array for the current vertex
      for (label <- 0 to noLabels.toInt - 1) {
        g_t_xt.put(label, unaryPotentials.get(sel))
        sel += 1
      }
      // add g_t_x_t to hashmap (to assign later the right vertexdata to the corresponding vertex)
      g_t += ((vertexId, g_t_xt))
    }

    // Graph initialization
    val new_graph = graph.mapVertices((vid, data) => {
      new PregelVertexData(g_t.getOrElse(vid.toInt, DoubleMatrix.zeros(noLabelsOfEachVertex.get(vid.toInt).toInt)), pwPotentials, 0, lastColumnId)
    }
    )

    // add number of neighbours to each node
    val next_graph = new_graph.outerJoinVertices(new_graph.outDegrees) { (vid, data, out_degree) => mapNode(data, out_degree.getOrElse(0), vid.toInt) }

    // initialize local phi_tt' data structure given ids of neighbour nodes
    var preprocessed_graph = next_graph.outerJoinVertices(new_graph.collectNeighborIds(EdgeDirection.Out)) {
      (vid, data, idsNeighbours) => {
        data.initPhiTT(idsNeighbours.get.map(v => v.toInt))
        data
      }
    }

    // *****
    // Start Min-Sum Diffusion Iteration
    // *****
    println("*************")
    println("GraphX/PREGEL Min-Sum Diffusion")
    println("*************")
    println("Graph loaded successfully:")
    println("noEdges: " + graph.numEdges)
    println("noVertices: " + graph.numVertices)
    val pregelRuns = (Math.ceil(noIter.toDouble / noIterPerBndComputation.toDouble) - 1).toInt
    val initialMessage = mutable.HashMap[Int, DoubleMatrix]()
    val continuationMessage = mutable.HashMap[Int, DoubleMatrix]()
    continuationMessage(-1) = DoubleMatrix.EMPTY
    var t_start = System.currentTimeMillis()

    for (i <- 0 to pregelRuns) {
      println("chunk " + i + "/" + pregelRuns)
      // pregel/msd

      preprocessed_graph = preprocessed_graph.cache().pregel(initialMessage, noIterPerBndComputation - 1, EdgeDirection.Out)(vprog, sendMsg, mergeMsg)


      // inform pregel that we are about to continue our computations
      // while the next iterations
      if (i == 0) {
        initialMessage(-1) = DoubleMatrix.EMPTY
      }

      // compute statistics
      val bnd: Array[Double] = boundArray.value
      val en: Array[Double] = energyArray.value
      var firstZero = bnd.indexOf(0.0)
      var eps = 0.
      firstZero match {
        case x if x > 1 => eps = bnd(firstZero - 1) - bnd(firstZero - 2)
        case -1 => {
          firstZero = bnd.length
          eps = bnd(firstZero - 1) - bnd(firstZero - 2)
        }
        case _ => eps = Double.MaxValue
      }
      println("I " + firstZero + " B: " + bnd(firstZero - 1).toString() + " eps: " + eps)
      println("E " + energyArray.value.toSeq.toString())
      val idx = (0 to firstZero - 1).map(f => f.toDouble)
      val yval = bnd.toList.slice(0, idx.last.toInt + 1)
      val yval2 = en.toList.slice(0, idx.last.toInt + 1)

      val visualizer = Figure()
      visualizer.clear()
      val p = visualizer.subplot(0)
      p += plot(idx, yval)
      p += plot(idx, yval2)

      //TODO: test if conv bnd reached

      if (SHOW_LABELING_IN_ITERATION) {
        val labeling = compute_grid_labeling(preprocessed_graph)
        val labelVisualizer = Figure()
        labelVisualizer.subplot(0) += image(labeling)
        labelVisualizer.subplot(0).title = "Primal solution of iteration " + i.toString
        labelVisualizer.subplot(0).xaxis.setTickLabelsVisible(false)
        labelVisualizer.subplot(0).yaxis.setTickLabelsVisible(false)
      }
    }
    println("runtime " + (System.currentTimeMillis() - t_start) + " ms")

    println()
    // compute primal solution


    /*
    // visualize (final) primal solution
    val labeling = compute_grid_labeling(temp_graph)
    val labelVisualizer = Figure()
    labelVisualizer.subplot(0) += image(labeling)
    labelVisualizer.subplot(0).title = "Primal solution"
    labelVisualizer.subplot(0).xaxis.setTickLabelsVisible(false)
    labelVisualizer.subplot(0).yaxis.setTickLabelsVisible(false)*/
  }

  // update phi_tt given phi_tt messages of neighbouring nodes
  def vprog(vertexId: VertexId, data: PregelVertexData, phi_tt_neighbours: mutable.HashMap[Int, DoubleMatrix]): PregelVertexData = {
    val newData: PregelVertexData = new PregelVertexData(data)
    val isActive: Boolean = isWhite(vertexId.toInt, newData.gridWidth, newData.white)

    // key -1 indicates that pregel was paused for bound/energy computation
    // and is about to continue its computations
    if (phi_tt_neighbours.contains(-1)) {
      newData.white = 0
      return newData
    }

    if (isActive == true) {
      // **********
      // update g_tt_phi for each neighbour
      // **********
      for ((neighbourId, phi_tt) <- newData.phi_tt) {
        // phi_tt_g_tt = g_tt' + phi_tt' + phi_t't
        val phi_tt_g_tt = newData.g_tt.addColumnVector(phi_tt).addRowVector(phi_tt_neighbours.getOrElse(neighbourId, DoubleMatrix.zeros(newData.noLabels)))
        newData.g_tt_phi(neighbourId) = phi_tt_g_tt
        //println(vertexId.toInt + "-" + neighbourId.toInt + "-" + isActive + " :" + phi_tt_g_tt)
      }
    }

    //*******
    // compute primal energy (vertex)
    //*******
    val enArraySz = energyArray.localValue.length
    val enUpd = Array.ofDim[Double](enArraySz)
    if (newData.iteration > 1 && ((newData.iteration - 1) % 2) == 1) {
      val bndUpdIdx = Math.floor((newData.iteration - 1) / 2).toInt
      // GTT PHI UPDATE
      for ((neighbourId, phi_tt) <- newData.phi_tt) {
        // phi_tt_g_tt = g_tt' + phi_tt' + phi_t't
        val phi_tt_g_tt = newData.g_tt.addColumnVector(phi_tt).addRowVector(phi_tt_neighbours.getOrElse(neighbourId, DoubleMatrix.zeros(newData.noLabels)))
        data.g_tt_phi(neighbourId) = phi_tt_g_tt
      }

      // AT UPDATE
      val lAt = DoubleMatrix.zeros(newData.noLabels)
      if (newData.iteration < 2) {
        lAt.putColumn(0, data.g_t) // g_t is empty from the 2nd iteration on
      }
      for ((k, v) <- data.g_tt_phi) {
        lAt.addiColumnVector(v.rowMins())
      }
      val lbl = lAt.argmin()
      newData.labelForEnergyCompuytation = lbl
      enUpd(bndUpdIdx) += data.g_t.get(lbl)
      energyArray += enUpd
    }

    if (isActive == false) {
      // update white/black state
      newData.white = (newData.white + 1) % 2
      newData.iteration += 1
      return newData
    }

    // **********
    // update A_t
    // **********
    // Calculate the sum of the minimum pairwise dual variables g_tt_phi_tt
    if (newData.iteration < 2) {
      newData.At.putColumn(0, newData.g_t) // g_t is empty from the 2nd iteration on
    } else {
      newData.At.putColumn(0, DoubleMatrix.zeros(newData.noLabels)) // g_t is empty from the 2nd iteration on
    }
    for ((neighbourId, phi_tt) <- newData.g_tt_phi) {
      newData.At.addiColumnVector(phi_tt.rowMins())
    }

    newData.label = newData.At.argmin()


    // **********
    // update phi_tt'
    // **********
    for ((neighbourId, g_tt_phi) <- newData.g_tt_phi) {
      newData.phi_tt += ((neighbourId,
        newData.phi_tt(neighbourId)
          .subColumnVector(g_tt_phi.rowMins())
          .addColumnVector(newData.At.dup().div(newData.out_degree.toDouble)).dup()
        ))
    }

    //**********
    //compute bound
    //**********
    val arraySz = boundArray.localValue.length
    val bndUpd = Array.ofDim[Double](arraySz)
    if (newData.iteration > 1 && ((newData.iteration - 1) % 2) == 1) {
      val bndUpdIdx = Math.floor((newData.iteration - 1) / 2).toInt
      for ((k, v) <- newData.g_tt_phi) {
        bndUpd(bndUpdIdx) += v.rowMins().min()
      }
      boundArray += bndUpd
    }

    // update white/black state
    newData.white = (newData.white + 1) % 2
    newData.iteration += 1

    // **********
    // return updated vertex
    // **********
    newData
  }

  // send updated phi_tt' to neighbouring nodes t'
  def sendMsg(triplet: EdgeTriplet[PregelVertexData, Int]): Iterator[(VertexId, mutable.HashMap[Int, DoubleMatrix])] = {
    val srcVtx: PregelVertexData = triplet.srcAttr
    val dstVtx: PregelVertexData = triplet.dstAttr
    val dstId: VertexId = triplet.dstId
    val srcId: VertexId = triplet.srcId
    val activeNextRound: Boolean = isWhite(srcId.toInt, srcVtx.gridWidth, srcVtx.white)

    //TODO commented out for debugging reasons
    // we only send new phi_tt if its actually updated
    // this behaviour is required to guarantee convergence in case of checker board parallelism
    /*if (activeNextRound == true) {
      // PREGEL forces us to send at least one message to each node that shall be used during next round
      val maxKey = triplet.dstAttr.phi_tt.keySet.max.toInt
      if (srcId.toInt == maxKey) {
        val dstData = mutable.HashMap[Int, DoubleMatrix]((triplet.srcId.toInt, DoubleMatrix.zeros(srcVtx.noLabels)))
        return Iterator((triplet.dstId, dstData))
      } else {
        return Iterator()
      }
    }*/


    val enArraySz = energyArray.localValue.length
    val enUpd = Array.ofDim[Double](enArraySz)
    val bndUpdIdx = Math.floor((srcVtx.iteration - 1) / 2).toInt - 1
    if (srcId < dstId && srcVtx.iteration > 2 && (((srcVtx.iteration - 1) % 2) == 0) && (bndUpdIdx < enArraySz)) {
      enUpd(bndUpdIdx) = dstVtx.g_tt.get(srcVtx.labelForEnergyCompuytation, dstVtx.labelForEnergyCompuytation)
      energyArray += enUpd
    }

    // send updated phi_tt to neighbouring nodes
    val dstData = mutable.HashMap[Int, DoubleMatrix]((triplet.srcId.toInt, srcVtx.phi_tt(dstId.toInt)))
    Iterator((triplet.dstId, dstData))
  }

  def isWhite(srcId: Int, gridWidth: Int, weiss: Int): Boolean = {
    ((((srcId % gridWidth) + (srcId / gridWidth)) % 2)) == weiss
  }

  // merge multiple messages arriving at the same vertex at the start of a superstep
  // before applying the vertex program vprog
  def mergeMsg(msg1: mutable.HashMap[Int, DoubleMatrix], msg2: mutable.HashMap[Int, DoubleMatrix]): mutable.HashMap[Int, DoubleMatrix] = {
    for ((k, v) <- msg2) {
      msg1(k) = v
    }
    msg1
  }

  def mapNode(data: PregelVertexData, out_degree: Int, vid: Int): PregelVertexData = {
    data.out_degree = out_degree
    data.vid = vid
    data
  }

  def compute_grid_labeling(g: Graph[PregelVertexData, Int]): DenseMatrix[Double] = {
    val vertexArray = g.mapVertices[Integer]((vid, vertexData) => vertexData.label
    ).vertices.sortByKey().map(elem => elem._2.toDouble).collect()
    val noRows = vertexArray.size / lastColumnId
    DenseVector(vertexArray).toDenseMatrix.reshape(lastColumnId, noRows)
  }

  def printPhiTT(vertId: Int, phi_tt: mutable.HashMap[Int, DoubleMatrix]): Unit = {
    for ((neighbourId, phi_tt) <- phi_tt) {
      println("phi_" + vertId + "," + neighbourId + " -> " + phi_tt)
    }
  }

}

class DoubleArrayAccumulator extends AccumulatorParam[Array[Double]] {
  override def addInPlace(r1: Array[Double], r2: Array[Double]): Array[Double] = {
    r1.zip(r2).map(r1r2 => r1r2._1 + r1r2._2)
  }

  override def zero(initialValue: Array[Double]): Array[Double] = {
    initialValue.map(v => 0.)
  }
}
