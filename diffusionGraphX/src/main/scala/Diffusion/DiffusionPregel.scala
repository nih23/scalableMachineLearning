package Diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx._
import org.apache.spark.{SparkConf, SparkContext}
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
      //.set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setAppName("GraphX Min-Sum Diffusion")
      .setMaster(args(2))
    val sc = new SparkContext(conf)

    val benchmark = args(0)
    println("Benchmark: " + benchmark)


    // load edge data
    val pwPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/pwFactors.csv")
    val noLabelsOfEachVertex: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/nrLabels.csv")
    val unaryPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/uFactors.csv")
    val gridWidth = scala.io.Source.fromFile("benchmark/" + benchmark + "/lcid.txt").getLines().next().toInt

    // create graph structure
    val graph = GraphLoader.edgeListFile(sc, "benchmark/" + benchmark + "/edgeListFile.txt", false, args(1).toInt).cache()

    // initialize and run distributed inference algorithm
    val diffInference = new DiffusionPregel(graph, noLabelsOfEachVertex, unaryPotentials, pwPotentials, gridWidth)
    diffInference.apply()
  }
}

class DiffusionPregel(graph: Graph[Int, Int], noLabelsOfEachVertex: DoubleMatrix, unaryPotentials: DoubleMatrix, pwPotentials: DoubleMatrix, lastColumnId: Integer) extends java.io.Serializable {
  val USE_DEBUG_PSEUDO_BARRIER: Boolean = false
  val SHOW_LABELING_IN_ITERATION: Boolean = false
  val noMaxIter = 5
  val conv_bound = 0.001
  val t_final = System.currentTimeMillis()

  def apply() = {

    var bound = 0.0

    // create hashmap of vertex data (unary factors)
    var g_t = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
    var sel = 0
    for (vertexId <- 0 to noLabelsOfEachVertex.rows - 1) {
      val noLabels = noLabelsOfEachVertex.get(vertexId)
      var g_t_xt: DoubleMatrix = DoubleMatrix.zeros(noLabels.toInt)

      // construct g_t_xt array for the current vertex
      for (label <- 0 to noLabels.toInt - 1) {
        g_t_xt.put(label, unaryPotentials.get(sel))
        sel += 1
      }
      // add g_t_x_t to hashmap (to assign later the right vertexdata to the corresponding vertex)
      g_t += ((vertexId, g_t_xt))
    }

    //Initialization
    var energy = 0.0
    val edges = new EdgeData(pwPotentials)
    println("Graph loaded successfully:")
    println("noEdges: " + graph.numEdges)
    println("noVertices: " + graph.numVertices)

    val new_graph = graph.mapVertices((vid, data) => {
      new PregelVertexData(g_t.getOrElse(vid.toInt, DoubleMatrix.zeros(noLabelsOfEachVertex.get(vid.toInt).toInt)), pwPotentials, 0, lastColumnId)
    }
    )

    // add number of neighbours to each node
    val next_graph = new_graph.outerJoinVertices(new_graph.outDegrees) { (vid, data, out_degree) => mapNode(data, out_degree.getOrElse(0), vid.toInt) }

    // initialize local phi_tt' data structure given ids of neighbour nodes
    val preprocessed_graph = next_graph.outerJoinVertices(new_graph.collectNeighborIds(EdgeDirection.Out)) {
      (vid, data, idsNeighbours) => {
        data.initPhiTT(idsNeighbours.get.map(v => v.toInt))
        data
      }
    }.mapEdges(e => edges)

    // *****
    // Start Min-Sum Diffusion Iteration
    // *****

    val initialMessage = mutable.HashMap[Int, DoubleMatrix]()

    val t_start = System.currentTimeMillis()
    val min = preprocessed_graph.pregel(initialMessage, noMaxIter, EdgeDirection.Both)(vprog, sendMsg, mergeMsg)
    val t_conv = System.currentTimeMillis()
    println("runtime " + (t_conv - t_start) + " ms")

    // compute primal solution

    /*val labeling = compute_grid_labeling(temp_graph)
    val labelVisualizer = Figure()
    labelVisualizer.subplot(0) += image(labeling)
    labelVisualizer.subplot(0).title = "Primal solution"
    labelVisualizer.subplot(0).xaxis.setTickLabelsVisible(false)
    labelVisualizer.subplot(0).yaxis.setTickLabelsVisible(false)*/
  }

  // update local phi_tt data structure and re-compute all dependent variables
  def vprog(vertexId: VertexId, data: PregelVertexData, phi_tt_neighbours: mutable.HashMap[Int, DoubleMatrix]): PregelVertexData = {
    // TODO maybe rewrite phi_tt in order to reflect actual structure: ((int, int), DoubleMatrix) ((t,t'),(phi_tt'))
    var newData: PregelVertexData = data

    // TODO remove debug output
    if (vertexId.toInt == 0) {
      println()
    }


    // **********
    // update phi_tt_g_tt for each neighbour locally
    // **********
    if (isWhite(vertexId.toInt, newData.gridWidth, newData.white)) {
      for ((neighbourId, phi_tt) <- newData.phi_tt) {
        // phi_tt_g_tt = g_tt' + phi_tt' + phi_t't
        val phi_tt_g_tt = newData.g_tt.addColumnVector(phi_tt).addColumnVector(phi_tt_neighbours.getOrElse(neighbourId, DoubleMatrix.zeros(newData.noLabels)))
        newData.phi_tt_g_tt(neighbourId) = phi_tt_g_tt
      }
    }

    // **********
    // update A_t
    // **********
    // Calculate the sum of the minimum pairwise dual variables g_tt_phi_tt
    if (isWhite(vertexId.toInt, newData.gridWidth, newData.white)) {
      newData.At.putColumn(0, newData.g_t) // g_t is empty from the 2nd iteration on
      for ((neighbourId, phi_tt) <- newData.phi_tt) {
        newData.At.addiColumnVector(phi_tt.rowMins())
      }

      // clear g_t as it is already contained in the floating messages
      // TODO find improved expression
      newData.g_t.subiColumnVector(newData.g_t)
    }

    // **********
    // update phi_tt'
    // **********
    if (isWhite(vertexId.toInt, newData.gridWidth, newData.white)) {
      for ((neighbourId, phi_tt) <- newData.phi_tt) {
        newData.phi_tt += ((neighbourId,
          newData.phi_tt(neighbourId)
            .subColumnVector(phi_tt_neighbours.getOrElse(neighbourId, DoubleMatrix.zeros(newData.noLabels)).rowMins())
            .addColumnVector(newData.At.dup().div(newData.out_degree.toDouble))
          ))
      }
    }

    // update white/black state
    newData.white = (newData.white + 1) % 2
    newData.touched += 1

    // TODO remove debug output
    if (vertexId.toInt == 0) {
      printPhiTT(vertexId.toInt, newData.phi_tt)
    }

    // **********
    // compute bound
    // **********
    // TODO: optimize bound computation -> seems that we need additional messages w/ phi_tt_g_tt

    // **********
    // return updated vertex
    // **********
    newData
  }

  def printPhiTT(vertId: Int, phi_tt: mutable.HashMap[Int, DoubleMatrix]): Unit = {
    for ((neighbourId, phi_tt) <- phi_tt) {
      println("phi_" + vertId + "," + neighbourId + " -> " + phi_tt)
    }
  }

  // user defined function to determine the messages to send out for the next iteration and where to send it to.
  def sendMsg(triplet: EdgeTriplet[PregelVertexData, EdgeData]): Iterator[(VertexId, mutable.HashMap[Int, DoubleMatrix])] = {
    val srcVtx: PregelVertexData = triplet.srcAttr
    val dstId: VertexId = triplet.dstId
    // send empty in case we didn't alter the data structure due to grid partitioning
    if (srcVtx.vid == 0) {
      println("test")
    }
    // TODO check if some key is not found.. if so this indicates that the graph aint initialized properly
    val dstData = mutable.HashMap[Int, DoubleMatrix]((triplet.srcId.toInt, srcVtx.phi_tt(dstId.toInt)))
    Iterator((triplet.dstId, dstData))
  }

  // merge multiple messages arriving at the same vertex at the start of a superstep
  // before applying the vertex program vprog
  // Here: Compute At
  def mergeMsg(msg1: mutable.HashMap[Int, DoubleMatrix], msg2: mutable.HashMap[Int, DoubleMatrix]): mutable.HashMap[Int, DoubleMatrix] = {
    for ((k, v) <- msg2) {
      msg1(k) = v
    }
    msg1
  }

  def mapNode(data: PregelVertexData, out_degree: Int, vid: Int): PregelVertexData = {
    data.out_degree = out_degree
    data.vid = vid
    if (vid == 0) {
      println()
    }
    data
  }

  def compute_grid_labeling(g: Graph[PregelVertexData, EdgeData]): DenseMatrix[Double] = {
    val vertexArray = g.mapVertices[Integer]((vid, vertexData) => {
      vertexData.At.argmin() // compute primal solution
    }).vertices.sortByKey().map(elem => elem._2.toDouble /* we only care about our label */).collect()
    val noRows = vertexArray.size / lastColumnId
    DenseVector(vertexArray).toDenseMatrix.reshape(lastColumnId, noRows)
  }

  def isWhite(srcId: Int, gridWidth: Int, weiss: Int): Boolean = {
    ((((srcId % gridWidth) + (srcId / gridWidth)) % 2)) == weiss
  }

}