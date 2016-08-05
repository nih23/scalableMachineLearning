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
    val benchmark = "snail" // triplepoint4-plain-ring
    //val benchmark = "triplepoint4-plain-ring"
    //val benchmark = "toy"


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
      // Firstly, set A_t to g_t after the first iteration At == 0
      val temp_graph2 = temp_graph.mapVertices( (vid,data) => set_a_t( data, i ) )

      //+++ Black +++

      // Compute g_tt_phi
      val black_graph1 = temp_graph2.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))

      //compute sum of min_g_tt_phi
      val newRdd = black_graph1.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc( msg )
      },
        (msg1: VertexData, msg2: VertexData ) => {
          msg1.min_sum( msg2 )
        }
      )

      val black_min_graph = Graph(newRdd, temp_graph.edges)

      // update phis
      val black_graph = black_min_graph.mapVertices( (vid,data) => compute_phi(vid, data, 0) )


      //+++ White +++

      // Compute g_tt_phi
      val white_graph1 = black_graph.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1))

      //compute sum of min_g_tt_phi
      val newRdd1 = white_graph1.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 1)
        edgeContext.sendToSrc( msg )
      },
        (msg1: VertexData, msg2: VertexData ) => {
          msg1.min_sum( msg2 )
        }
      )

      val white_min_graph = Graph(newRdd1, temp_graph.edges)
      // update phis
      temp_graph = white_min_graph.mapVertices( (vid,data) => compute_phi(vid, data, 1) )

      //+++ COMPUTE BOUND +++
      // Compute g_tt_phi
      val bound_graph = temp_graph.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))

      val aggregate_v = bound_graph.aggregateMessages[Double]( triplet => {
        if ( isWhite(triplet.srcId.toInt, 0)){
          val minimum = triplet.attr.g_tt_phi.rowMins().min()
          triplet.sendToSrc( minimum )
        }
      },
        (a,b) => a+b )


      // sum up for bound computation
      bound = aggregate_v.aggregate[Double] (zeroValue = 0.0) ((double, data) => double + data._2, (a,b) => a+b )

      // sum up for energy computation
      // Energy of pairwise factors
      // Compute new g_tt_phi
      temp_graph = temp_graph.mapTriplets(triplet =>
                      compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))
      energy =  temp_graph.edges.aggregate[Double](zeroValue = 0.0 ) ((double,data) => compute_edge_energy(double,data ), (a,b) => a+b )
      energy += temp_graph.vertices.aggregate[Double] (zeroValue = 0.0) ((double,data) => compute_vertice_energy(double,data._2), (a,b) => a+b)
      println(i + " -> E " + energy  + " B " + bound)

      // reset phi_tt_g_tt for fresh compuation in next round
      /*temp_graph = white_graph.mapVertices((vid,data) =>{
        data.phi_tt_g_tt = data.phi_tt_g_tt.empty
        data
      })*/

    }


  }

  def compute_edge_energy( double : Double, data : Edge[EdgeData] ) : Double = {
    if (isWhite(data.srcId, 0)) double + data.attr.g_tt_phi.rowMins().min() else double
  }

  def compute_vertice_energy( double : Double, data: VertexData ) : Double = {
    var g_t_phi = data.g_t
    for ( (k,v) <- data.phi_tt) {
      g_t_phi.subColumnVector( v )
    }
    //println( data.g_t.get(g_t_phi.argmin()) )
    double + data.g_t.get( g_t_phi.argmin() )
  }

  def set_a_t( data: VertexData, i: Int ) : VertexData = {
    if ( i == 0 ) {
      data.At = data.g_t
    }
    else {
      data.At = DoubleMatrix.zeros(data.At.rows)
    }
    data
  }

  def compute_g_tt_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int) : EdgeData = {
    if (isWhite(srcId, weiss)) {
       attr.g_tt_phi = attr.g_tt.addColumnVector( src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)) )
                                .addColumnVector( dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)).transpose())
      //println(attr.g_tt_phi)
      //print(srcId)
    }
    attr
  }

  def update_mins(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): Tuple2[VertexData, VertexData] = {
    if (isWhite(srcId, weiss)) {
      src_data.min_gtt_phi += ((dstId.toInt, src_data.phi_tt_g_tt.get(dstId.toInt).get.rowMins()))
      src_data.g_t_phi.putColumn(0, src_data.g_t.div(src_data.out_degree.toDouble).subColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))))
    }
    (src_data, dst_data)
  }

  def send_g_tt_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): VertexData = {
    //TODO: there seems to be a flaw in this computation ..
    if (isWhite(srcId, weiss)) {
      src_data.phi_tt_g_tt += ( (dstId.toInt, attr.g_tt_phi ) )
      //src_data.phi_tt_g_tt += ((dstId.toInt,
      //  src_data.phi_tt_g_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
      //    .add(attr.g_tt.div(2.0))
      //    .addColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)))))
    }
    /*else {
      dst_data.phi_tt_g_tt += ((srcId.toInt,
        dst_data.phi_tt_g_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
          .add(attr.g_tt.div(2.0))
          .addRowVector(attr.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)).transpose())))
    }*/
    src_data
  }

  def compute_phi( srcId: VertexId, data : VertexData, weiss: Int ) : VertexData = {
    if ( isWhite(srcId, weiss) ) {
      for ((v,k) <- data.phi_tt_g_tt ){
        data.phi_tt += ((v, data.phi_tt.getOrElse(v, DoubleMatrix.zeros(data.g_t.rows))
                                        .subColumnVector( k.rowMins() )
                                        .addColumnVector( data.At.div( data.out_degree.toDouble ) )) )
        //println(data.phi_tt)
      }
    }
    data
  }

/*
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
  }*/

  def compute_energy(src_attr: VertexData, dst_attr: VertexData, attr: EdgeData): Double = {
    val src_label = src_attr.g_t_phi.argmin()
    val dst_label = dst_attr.g_t_phi.argmin()
    var energy: Double = src_attr.g_t.get(src_label)
    energy += dst_attr.g_t.get(dst_label)
    energy += attr.g_tt.get(src_label, dst_label)
    energy
  }

  def isWhite(srcId: VertexId, weiss: Int): Boolean = {
    ((((srcId.toInt % lastColumnId) + (srcId.toInt / lastColumnId)) % 2) ) == weiss
  }

  def mapNode(data: VertexData, out_degree: Int): VertexData = {
    data.out_degree = out_degree
    data
  }

}
