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
    // turn off debug output
    Logger.getRootLogger.setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setAppName("GraphX Min-Sum Diffusion")
      .setMaster("local[16]")
    val sc = new SparkContext(conf)


    // TODO: import data of opengm's hdf5 file
    //val benchmark = "snail" // triplepoint4-plain-ring
    val benchmark = "triplepoint4-plain-ring"


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

    // create hashmap of vertex data (unary factors)
    var g_t = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
    var sel = 0
    for (vertexId <- 0 to noLabelsOfEachVertex.rows - 1)
    {
      val noLabels = noLabelsOfEachVertex.get(vertexId)
      var g_t_xt: DoubleMatrix = DoubleMatrix.zeros(noLabels.toInt)

      // construct g_t_xt array for the current vertex
      for (label <- 0 to noLabels.toInt - 1)
      {
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
      new VertexData(g_t.getOrElse(vid.toInt, DoubleMatrix.zeros(noLabelsOfEachVertex.get(vid.toInt).toInt)))
    }
    )
    val next_graph = new_graph.outerJoinVertices(new_graph.outDegrees){(vid,data,out_degree) => mapNode(data,out_degree.getOrElse(0)) }
    var temp_graph = next_graph.mapEdges(e => edges)


    // Start Min-Sum Diffusion Iteration
    for (i <- 0 to maxIt)
    {
      //+++ Black +++
      //compute min_g_tt_phi
      val newRdd = temp_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = compute_min(edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc(msg._1)
        edgeContext.sendToDst(msg._2)
      }
        , (msg1: VertexData, msg2: VertexData) => {
          msg1 + msg2
        }
      )
      val black_min_graph = Graph(newRdd, temp_graph.edges)

      //send mins to hashmap
      val newRdd2 = black_min_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = update_mins(edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc(msg._1)
        edgeContext.sendToDst(msg._2)
      }
        , (msg1: VertexData, msg2: VertexData) => {
          msg1 + msg2
        }
      )
      val black_send_graph = Graph(newRdd2, black_min_graph.edges)

      // update phis
      val black_graph = black_send_graph.mapTriplets(triplet =>
        compute_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0, i))


      //+++ White +++
      //compute min_g_tt_phi
      val newRdd3 = black_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = compute_min(edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 1)
        edgeContext.sendToSrc(msg._1)
        edgeContext.sendToDst(msg._2)
      }
        , (msg1: VertexData, msg2: VertexData) => {
          msg1 + msg2
        }
      )
      val white_min_graph = Graph(newRdd3, black_graph.edges)

      //send mins to hashmap
      val newRdd4 = white_min_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = update_mins(edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 1)
        edgeContext.sendToSrc(msg._1)
        edgeContext.sendToDst(msg._2)
      }
        , (msg1: VertexData, msg2: VertexData) => {
          msg1 + msg2
        }
      )
      val white_send_graph = Graph(newRdd4, white_min_graph.edges)

      val white_graph = white_send_graph.mapTriplets(triplet =>
        compute_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1, i))

      //+++ COMPUTE BOUND +++
      //val bound_min_triplets = white_graph.mapTriplets(triplet => compute_min(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))
      val newRdd5 = white_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = compute_min(edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc(msg._1)
        edgeContext.sendToDst(msg._2)
      }
        , (msg1: VertexData, msg2: VertexData) => {
          msg1 + msg2
        }
      )
      val bound_min_triplets = Graph(newRdd5, white_graph.edges)


      val aggregate_vertices = bound_min_triplets.aggregateMessages[Double](triplet => {
        if (isWhite(triplet.srcId.toInt, 0)) {
          val mv = triplet.srcAttr.min_gtt_phi.get(triplet.dstId.toInt).get.min()
          if (triplet.srcId < 10) {
            //println(triplet.srcId.toInt + " => " + triplet.srcAttr.phi_tt_g_tt(triplet.dstId.toInt).toString + " **MIN** " + triplet.srcAttr.phi_tt_g_tt(triplet.dstId.toInt).rowMaxs().toString)
          }
          triplet.sendToSrc(triplet.srcAttr.min_gtt_phi.get(triplet.dstId.toInt).get.min())
        }
      },
        (a, b) => a + b
      )

      // sum up for bound computation
      bound = aggregate_vertices.aggregate[Double] (zeroValue = 0.0) ((id, data) => data._2, (a,b) => a+b )

      // aggregate energies (use sum of mins of previous state to save computation, as it only is a heuristic)
      val aggregate_vertices_energy = white_graph.aggregateMessages[Double](triplet => {
        if (isWhite(triplet.srcId.toInt, 0)) {
          //if ((((triplet.srcId.toInt % lastColumnId) + (triplet.srcId.toInt / lastColumnId)) % 2) == 0) {
          triplet.sendToSrc(compute_energy(triplet.srcAttr, triplet.dstAttr, triplet.attr))
        }
      },
        (a,b) => a + b
      )
      // sum up for energy computation
      energy = aggregate_vertices_energy.aggregate[Double] (zeroValue = 0.0) ((id, data) => data._2, (a,b) => a+b)
      println(i + " -> E " + energy + " B " + bound)

      // reset phi_tt_g_tt for fresh compuation in next round
      temp_graph = white_graph.mapVertices((vid,data) =>{
        data.phi_tt_g_tt = data.phi_tt_g_tt.empty
        data
      })

    }


  }



  def update_mins(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): Tuple2[VertexData, VertexData] = {
    if (isWhite(srcId, weiss)) {
      src_data.min_gtt_phi += ((dstId.toInt, src_data.phi_tt_g_tt.get(dstId.toInt).get.rowMins()))
      src_data.g_t_phi.putColumn(0, src_data.g_t.div(src_data.out_degree.toDouble).subColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))))
    }
    (src_data, dst_data)
  }

  def compute_min(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): Tuple2[VertexData, VertexData] = {
    //TODO: there seems to be a flaw in this computation ..
    if (isWhite(srcId, weiss)) {
      src_data.phi_tt_g_tt += ((dstId.toInt,
        src_data.phi_tt_g_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
          .add(attr.g_tt.div(2.0))
          .addColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)))))
    }
    else {
      dst_data.phi_tt_g_tt += ((srcId.toInt,
        dst_data.phi_tt_g_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
          .add(attr.g_tt.div(2.0))
          .addRowVector(attr.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)).transpose())))
    }
    (src_data, dst_data)
  }


  def compute_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int, iter: Int): EdgeData = {
    if (isWhite(srcId, weiss)) {
      // compute sum of mins
      //src_data.At.fill(0.)

      var At = DoubleMatrix.zeros(src_data.g_t.length)
      if (iter == 0) // Add g_t in the first iteration
      {
        At = src_data.g_t
      }

      // add gtt_phi of neighbouring grid elements
      for ((k, v) <- src_data.min_gtt_phi) {
        At = At.addColumnVector(v)
      }

      // update phi_tt'
      attr.phi_tt += ((srcId.toInt,
        attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))
          .subColumnVector(src_data.min_gtt_phi.get(dstId.toInt).get)
          .subColumnVector(At.div(src_data.out_degree.toDouble))
        ))
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

  def isWhite(srcId: VertexId, weiss: Int): Boolean = {
    ((((srcId.toInt % lastColumnId) + (srcId.toInt / lastColumnId)) % 2) + weiss) == 0
  }

  def mapNode(data: VertexData, out_degree: Int): VertexData = {
    data.out_degree = out_degree
    data
  }

}
