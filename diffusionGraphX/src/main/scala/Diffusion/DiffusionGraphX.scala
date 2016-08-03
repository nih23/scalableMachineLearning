package main.scala.Diffusion

import Diffusion.{EdgeData, VertexData}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx._
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * first draft of scalable diffusion algorithm based on Apache Spark GraphX
  *
  * @author Nico Hoffmann, Benjamin Naujoks
  */

object DiffusionGraphX
{

  def main(args: Array[String]): Unit = {

    // turn off dbg output
    Logger.getRootLogger.setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf()
      //.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setAppName("Distributed Diffusion")
      .setMaster("local[2]")
    val sc = new SparkContext(conf)


    // TODO: import data of opengm's hdf5 file
    val benchmark = "snail" // triplepoint4-plain-ring
    //val benchmark = "triplepoint4-plain-ring"


    // load edge data (experimental)
    val pwPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/pwFactors.csv")
    val noLabelsOfEachVertex: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/nrLabels.csv")
    var unaryPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/uFactors.csv")
    val lcid = scala.io.Source.fromFile("benchmark/" + benchmark + "/lcid.txt").getLines().next().toInt

    // create graph structure
    val graph = GraphLoader.edgeListFile(sc, "benchmark/" + benchmark + "/edgeListFile.txt")

    // initialize and run distributed inference algorithm
    val diffInference = new DiffusionGraphX(graph, noLabelsOfEachVertex, unaryPotentials, pwPotentials, lcid)
    diffInference.apply()
  }
}

class DiffusionGraphX(graph: Graph[Int, Int], noLabelsOfEachVertex: DoubleMatrix, unaryPotentials: DoubleMatrix, pwPotentials: DoubleMatrix, lastColumnId: Integer) extends java.io.Serializable {

  val maxIt = 200
  val conv_bound = 0.001

  def apply() = {

    var bound = 0.0

    // Make hashmap of vertex data (unary factors)
    var g_t = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
    var sel = 0
    for (vertexId <- 0 to noLabelsOfEachVertex.rows - 1)
    {
      val noLabels = noLabelsOfEachVertex.get(vertexId)
      var g_t_xt: DoubleMatrix = DoubleMatrix.zeros(noLabels.toInt)

      //Construct g_t_xt array for the current vertex
      for (label <- 0 to noLabels.toInt - 1)
      {
        g_t_xt.put(label, unaryPotentials.get(sel))
        sel += 1
      }
      //Add to hashmap (to assign later the right vertexdata to the corresponding vertex)
      g_t += ((vertexId, g_t_xt))
    }

    //Initialization
    val edges = new EdgeData(pwPotentials)
    println("noEdges: " + graph.numEdges)
    println("noVertices: " + graph.numVertices)

    val new_graph = graph.mapVertices((vid, data) =>
      new VertexData(g_t.getOrElse(vid.toInt, DoubleMatrix.zeros(noLabelsOfEachVertex.get(vid.toInt).toInt))))



    val next_graph = new_graph.outerJoinVertices(new_graph.outDegrees){(vid,data,out_degree) => mapNode(data,out_degree.getOrElse(0)) }
    val final_graph = next_graph.mapEdges(e => edges)

    var energy = 0.0
    var temp_graph = final_graph //.mapTriplets( triplet => new VertexEdgeData(triplet.attr, triplet.srcAttr, triplet.dstAttr))

    //val col_edges_1 = final_graph.collectEdges(EdgeDirection.Either).collect()

    // Start iteration
    for (i <- 0 to maxIt)
    {
      //TODO: use aggregateMessage for compute and send min g_tt_phi

      //++++++Black++++++
      //compute min_g_tt_phi

      val black_min_graph = temp_graph.mapTriplets(triplet => {
        compute_min(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0)
      }
      )

      //send mins to hashmap
      val black_send_graph = black_min_graph.mapTriplets(triplet =>
        send_mins(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))

      // update phis
      val black_graph = black_send_graph.mapTriplets(triplet =>
        compute_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0, i))

      // persist computation
      black_graph.triplets.count()

      //val col_edges_2 = black_graph.collectEdges(EdgeDirection.Either).collect()

      //++++White*****
      val white_min_graph = black_graph.mapTriplets(triplet =>
        compute_min(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1))

      val white_send_graph = white_min_graph.mapTriplets(triplet =>
        send_mins(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1))

      val white_graph = white_send_graph.mapTriplets(triplet =>
        compute_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1, i))

      // persist computation
      white_graph.triplets.count()

      //val col_edges_3 = white_graph.collectEdges(EdgeDirection.Either).collect()

      //+++bound++++
      val bound_min_triplets = white_graph.mapTriplets(triplet => compute_min(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))
      val aggregate_vertices = bound_min_triplets.aggregateMessages[Double](triplet => {
        if ((((triplet.srcId.toInt % lastColumnId) + (triplet.srcId.toInt / lastColumnId)) % 2) == 0) {
          val datatoSend = triplet.srcAttr.min_gtt.get(triplet.dstId.toInt).get.min()
          //println("Sending => " + datatoSend)
          triplet.sendToSrc(triplet.srcAttr.min_gtt.get(triplet.dstId.toInt).get.min())
        }
        else // Do not count the edges twice.
          triplet.sendToSrc(0.)
      },
        (a, b) => a + b
      )

      // Finally sum up for the bound
      bound = aggregate_vertices.aggregate[Double] (zeroValue = 0.0) ((id, data) => data._2, (a,b) => a+b )

      // Aggregate energies (use sum of mins of previous state to save computation, as it only is a heuristic)
      val aggregate_vertices_energy = white_graph.aggregateMessages[Double](triplet => {
        if ((((triplet.srcId.toInt % lastColumnId) + (triplet.srcId.toInt / lastColumnId)) % 2) == 0) {
          triplet.sendToSrc(compute_energy(triplet.srcAttr,triplet.dstAttr,triplet.attr))
        }
      },
        (a,b) => a + b
      )
      // Finally sum up energy
      energy = aggregate_vertices_energy.aggregate[Double] (zeroValue = 0.0) ((id, data) => data._2, (a,b) => a+b)

      println("\nenergy: " + energy)
      println("bound: " + bound + "\n")

      // Start the next iteration with newest graph and reset phi_tt_g_tt first
      temp_graph = white_graph.mapVertices((vid,data) =>{
        data.phi_tt_g_tt = data.phi_tt_g_tt.empty
        data
      })

    }


  }

  def mapNode(data: VertexData, out_degree: Int): VertexData = {
    data.out_degree = out_degree
    data
  }


  /////////////////////////////////
  // OLD IMPLEMENTATION (DEPRECATED)
  /////////////////////////////////
  //TODO: BROKEN!!! VERTEX DATA IS IMMUTABLE
  def send_mins(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): EdgeData = {
    if (((((srcId.toInt % lastColumnId) + (srcId.toInt / lastColumnId)) % 2) + weiss) == 0) {
      //src_data.min_gtt += ((dstId.toInt, attr.min_gtt_phi.rowMins()))
      //println("row mins " + srcId + "->" + dstId + "= " + src_data.phi_tt_g_tt.get(dstId.toInt).get.rowMins())
      src_data.min_gtt += ((dstId.toInt, src_data.phi_tt_g_tt.get(dstId.toInt).get.rowMins()))
      // Reinitialize g_tt_phi temp array
      //src_data.phi_tt_g_tt += ((dstId.toInt, DoubleMatrix.zeros(attr.attr.rows,attr.attr.columns)))
      // Compute g_t_phi (Every edge adds g_t/out_degree to g_t_phi)
      src_data.g_t_phi.putColumn(0, src_data.g_t.div(src_data.out_degree.toDouble).subColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))))
    }
    attr
  }

  //TODO: BROKEN!!! VERTEX DATA IS IMMUTABLE
  def compute_min(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): EdgeData = {
    if (((((srcId.toInt % lastColumnId) + (srcId.toInt / lastColumnId)) % 2) + weiss) == 0) {
      src_data.phi_tt_g_tt += ((dstId.toInt,
        src_data.phi_tt_g_tt.getOrElse(dstId.toInt,
          DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns)).add(attr.g_tt.div(2.0).
          addColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows))))))
    }
    else {
      dst_data.phi_tt_g_tt += ((srcId.toInt,
        dst_data.phi_tt_g_tt.getOrElse(srcId.toInt,
          DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns)).add(attr.g_tt.div(2.0).
          addRowVector(attr.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)).transpose()))))
    }
    attr
  }

  //TODO: BROKEN!!! VERTEX DATA IS IMMUTABLE
  def compute_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int, iter: Int): EdgeData = {
    if (((((srcId.toInt % lastColumnId) + (srcId.toInt / lastColumnId)) % 2) + weiss) == 0) {

      // compute sum of mins
      src_data.min_sum.fill(0.)
      for ((k, v) <- src_data.min_gtt) {
        src_data.min_sum = src_data.min_sum.addColumnVector(v)
      }

      // update phi_tt'
      var innerVec = src_data.min_sum

      if (iter == 0) // Add g_t in the first iteration
      {
        innerVec = innerVec.addColumnVector(src_data.g_t)
        //  attr.phi_tt += ((srcId.toInt, attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)).subiColumnVector(src_data.min_gtt.get(dstId.toInt).get.subColumnVector(src_data.min_sum.addColumnVector(src_data.g_t).div(src_data.out_degree.toDouble)))))
      }
      //else
      //{
      //  attr.phi_tt += ((srcId.toInt, attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)).subiColumnVector(src_data.min_gtt.get(dstId.toInt).get.subColumnVector(src_data.min_sum.div(src_data.out_degree.toDouble)))))
      //}
      attr.phi_tt += ((srcId.toInt,
        attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))
          .subColumnVector(src_data.min_gtt.get(dstId.toInt).get
            .subColumnVector(innerVec.div(src_data.out_degree.toDouble)))))
    }

    attr
  }

  def compute_energy(src_attr: VertexData, dst_attr: VertexData, attr: EdgeData): Double = {
    val src_label = src_attr.g_t_phi.argmin()
    val dst_label = dst_attr.g_t_phi.argmin()
    var energy: Double = src_attr.g_t.get(src_label)
    energy += dst_attr.g_t.get(dst_label)
    energy += attr.g_tt.get(src_label, dst_label)
    energy
  }
}
