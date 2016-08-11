package main.scala.Diffusion

import Diffusion.{EdgeData, VertexData}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx._
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * first draft of scalable diffusion algorithm based on Apache Spark GraphX
  *
  * @author Benjamin Naujoks, Nico Hoffmann
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
    //val benchmark = "toy2"


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

  val maxIt = 10
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
    val next_graph = new_graph.outerJoinVertices(new_graph.outDegrees){(vid,data,out_degree) => mapNode(data,out_degree.getOrElse(0), vid.toInt) }
    var temp_graph = next_graph.mapEdges(e => edges)



    // Start Min-Sum Diffusion Iteration
    for (i <- 0 to maxIt)
    {
      // Firstly, set A_t to g_t after the first iteration At == 0
      /*if ( i == 0 )  {
        temp_graph = temp_graph.mapVertices( (vid,data) => {
          data.At.putColumn(0,data.g_t )
          data
        } )
      }
      else {
        temp_graph = temp_graph.mapVertices(  (vid,data) => {
          data.At.putColumn(0, DoubleMatrix.zeros(data.At.rows))
          data
        } )
      }*/
      //+++ Black +++

      // Compute g_tt_phi
      val black_graph1 = temp_graph.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))

      //compute sum of min_g_tt_phi
      val newRdd = black_graph1.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
          msg1
        }
      )
      println( black_graph1.vertices.count() )

      val black_min_graph = Graph(newRdd, black_graph1.edges)
      println( black_min_graph.vertices.count() )
      val black_min_graph2 = black_min_graph.mapVertices( (vid,data) => {
        if ( isWhite(vid.toInt,0) ) {
          if ( i != 0 )  data.At.putColumn(0,data.At.fill(0.))
          for ((k, v) <- data.phi_tt_g_tt) {
            //println( "rowmins : " + v.rowMins() + " vid " + vid)
            data.At.addiColumnVector(v.rowMins())
          }
          //println( "A_t: " + data.At + " vid: " + vid.toInt)
        }

          data
      }  )
      //println( black_min_graph2.vertices.count() )
      // update phis
      val black_graph = black_min_graph2.mapVertices( (vid,data) => compute_phi(vid, data, 0) )
      //println( black_graph.vertices.count() )
      //+++ White +++

      // Compute g_tt_phi
      val white_graph1 = black_graph.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 1))
      println( white_graph1.vertices.count() )
      //compute sum of min_g_tt_phi
      val newRdd1 = white_graph1.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 1)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
          //println(" mergfunction: msg2 phitt_gtt" + msg2.phi_tt_g_tt + " vid " + msg2.vid + " msg1 phitt_gtt " + msg1.phi_tt_g_tt + " msg1 vid" + msg1.vid )
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
          //println( "merged msg1 phittgt " + msg1.phi_tt_g_tt + "vid " + msg1.vid )
          msg1
        }
      )
      println( white_graph1.vertices.count() )

      val white_min_graph = Graph(newRdd1, white_graph1.edges)
      println( white_min_graph.vertices.count() )
      val white_min_graph2 = white_min_graph.mapVertices( (vid,data) => {
        if (isWhite(vid.toInt, 1)) {
          //println( "At compuitation phi_tt_gtt: "  + data.phi_tt_g_tt )
          if ( i!= 0 )  data.At.putColumn(0,data.At.fill(0.))
          for ((k, v) <- data.phi_tt_g_tt) {
            //if ( vid == 3 ) println( " key of 3: " + k + " value " + v )
            data.At.addiColumnVector(v.rowMins())
          }
          //println( "A_t: " + data.At + " vid: " + vid.toInt )

        }
        data
      }  )
      println( white_min_graph2.vertices.count() )
      // update phis
      temp_graph = white_min_graph2.mapVertices( (vid,data) => compute_phi(vid, data, 1) )

      println( temp_graph.vertices.count() )
      //+++ COMPUTE BOUND +++

      // Compute g_tt_phi
      val bound_graph = temp_graph.mapTriplets(triplet =>
        compute_g_tt_phi(triplet.srcId, triplet.dstId, triplet.srcAttr, triplet.dstAttr, triplet.attr, 0))

      val newRddBound = bound_graph.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 2)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
        //  println(" mergfunction: msg2 phitt_gtt" + msg2.phi_tt_g_tt + " vid " + msg2.vid + " msg1 phitt_gtt " + msg1.phi_tt_g_tt + " msg1 vid" + msg1.vid )
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
        //  println( "merged msg1 phittgt " + msg1.phi_tt_g_tt + "vid " + msg1.vid )
          msg1
        }
      )
      println( bound_graph.vertices.count() )

      val bound_graph2 = Graph(newRddBound, bound_graph.edges)

      val temp_graph2 = bound_graph2.mapVertices((vid, data) => {
        data.At.putColumn(0, data.At.fill(0.))
       // println( "Before energy: phi_tt_gtt_of " + vid +" " + data.phi_tt_g_tt + " phi_tt " + data.phi_tt)
        for ((k, v) <- data.phi_tt_g_tt) {
          //println("rowmins : " + v.rowMins() + " vid " + vid)
          data.At.addiColumnVector(v.rowMins())
        }
        data.label = data.At.argmin()
       // println("Before energy computation: A_t: " + data.At + " vid: " + vid.toInt)
        data
      }
      )

      val vertice_energy = temp_graph2.vertices.aggregate[Double] (zeroValue = 0.0) ((double,data) => {
        //  println( "Compute vertice energy: " + " vid " + data._1 +  " label " + data._2.label + " At " + data._2.At)
          double + data._2.g_t.get( data._2.label )
      }, (a,b) => a+b)


      val aggregate_v = bound_graph2.aggregateMessages[Double]( triplet => {
        var minimum = 0.0
        if ( isWhite(triplet.srcId.toInt, 0)){
          val new_gtt_phi = triplet.attr.g_tt.addColumnVector( triplet.srcAttr.phi_tt.getOrElse(triplet.dstId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)) )
            .addRowVector( triplet.dstAttr.phi_tt.getOrElse(triplet.srcId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)).transpose())
          minimum = new_gtt_phi.rowMins().min()
        //  println(" minimum g_tt_phi of id " + triplet.srcId.toInt + " "+ minimum + " gttphi " + new_gtt_phi )
        }
        triplet.sendToSrc( minimum )
      },
        (a,b) => a+b )
      println(aggregate_v.count())
      // sum up for bound computation
      bound = aggregate_v.aggregate[Double] (zeroValue = 0.0) ((double, data) => {
       // println( "double: " + double + " data_2 " + data._2 )
       double + data._2}, (a,b) => a+b )
      /*bound = bound_graph.edges.aggregate[Double](zeroValue = 0.0) ((double, data) => {
        var result = 0.
        if ( isWhite(data.srcId.toInt,0 ) ){
          result = data.attr.g_tt_phi.rowMins().min()
        }
        double + result
      }, (a,b) => a+b ) */

      // sum up for energy computation
      // Energy of pairwise factors




      println( temp_graph2.triplets.count() )
      println( bound_graph2.vertices.count())
      energy = temp_graph2.triplets.aggregate[Double](  zeroValue =0.0  )((double,triplet) => {
       //   println( "srcattr at " + triplet.srcAttr.At +"----------------------")
          triplet.attr.g_tt.get( triplet.srcAttr.At.argmin(),triplet.dstAttr.At.argmin() )
      },
        (a,b)=> a+b )
     // println( "other engergy: " + energy)
      val temp_graph3 = temp_graph2.mapTriplets(  triplet => {

         // println(" id: " + triplet.srcId + "argmin: " + triplet.srcAttr.At.argmin() + " At: " + triplet.srcAttr.At + " label " + triplet.srcAttr.label)
         // println(" id: " + triplet.dstId + "argmin: " + triplet.dstAttr.At.argmin() + " At: " + triplet.dstAttr.At + " label " + triplet.dstAttr.label)
          triplet.attr.src_label = triplet.srcAttr.At.argmin()
          triplet.attr.dst_label = triplet.dstAttr.At.argmin()

        triplet.attr
      } )

      energy = temp_graph3.edges.aggregate[Double](zeroValue = 0.0 ) ((double,data) => double + compute_edge_energy(double,data ), (a,b) => a+b )
    //  println("energy edge: " + energy)
      println( temp_graph3.edges.collect() )
     // println("energy edge: " + energy)
      println( temp_graph3.edges.collect() )
      energy += vertice_energy
      println(i + " -> E " + energy +" B " + bound + "---------------------------------------------")

      // reset phi_tt_g_tt for fresh compuation in next round
      /*temp_graph = white_graph.mapVertices((vid,data) =>{
        data.phi_tt_g_tt = data.phi_tt_g_tt.empty
        data
      })*/

    }


  }

  def compute_edge_energy( double : Double, data : Edge[EdgeData] ) : Double = {
   // println( "g_tt energy " + data.attr.g_tt + "src und dstlabel: " + data.attr.src_label + data.attr.dst_label )
    var result = 0.
    if (isWhite(data.srcId.toInt, 0)) {
      result = data.attr.g_tt.get(data.attr.src_label, data.attr.dst_label)
    }
    //println( "result edge:" + result + "srcid: " + data.srcId.toInt)
    result
  }

  def compute_vertice_energy( double : Double, data: VertexData ) : Double = {
 //   println( " Label: " + data.At.argmin())
    val result = double + data.g_t.get( data.At.argmin() )
   // println( " vertice energy: " + result + " of id : " + data.vid)
    result
  }

  def set_a_t( data: VertexData, i: Int ) : VertexData = {
    if ( i == 0 ) {
      data.At.putColumn(0,data.g_t )
    }
    else {
      data.At.fill(0.)
    }
    data
  }

  def compute_g_tt_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int) : EdgeData = {
    //println( " compute_g_tt_phi: old g_tt_phi: " + attr.g_tt_phi + " srcid: " + srcId.toInt  + " dtsid: " + dstId.toInt + " weiss: " + weiss)
    if (isWhite(srcId.toInt, weiss) || weiss == 2 ) {
      //if (srcId.toInt == 1 && dstId.toInt == 0) println( "g_tt before: " + attr.g_tt)
       attr.g_tt_phi.copy( attr.g_tt )
       attr.g_tt_phi.addiColumnVector(src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)) )
                    .addiRowVector(dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(dst_data.g_t.rows)).transpose())
         //= attr.g_tt.addColumnVector( src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)) )
         //                       .addColumnVector( dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(dst_data.g_t.rows)).transpose() )

      /*if ( srcId.toInt == 1 && dstId.toInt == 0 )
        {
          println( "g_tt: " + attr.g_tt)
          println( "phi_tt' " + src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)) )
          println( "phi_t't " + dst_data.phi_tt(srcId.toInt) )
          println( "g_tt_phi of 1: " + attr.g_tt_phi)
        }*/
      //println( "compute_g_tt_phi: phi_tt' " + src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)) + " weiss: " + weiss)
      //println( "compute_g_tt_phi: phi_t't " + dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(dst_data.g_t.rows)).transpose() + " weiss: " + weiss )

      //if ( srcId.toInt == 0) println(attr.g_tt_phi)
    }
 //   println( " compute g_tt_phi: new_gtt_phi"  + attr.g_tt_phi + " srcid: " + srcId.toInt  + " dtsid: " + dstId.toInt + " weiss: " + weiss )

    attr
  }

  def update_mins(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): Tuple2[VertexData, VertexData] = {
    if (isWhite(srcId.toInt, weiss)) {
      src_data.min_gtt_phi += ((dstId.toInt, src_data.phi_tt_g_tt.get(dstId.toInt).get.rowMins()))
      src_data.g_t_phi.putColumn(0, src_data.g_t.div(src_data.out_degree.toDouble).subColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows))))
    }
    (src_data, dst_data)
  }

  def send_g_tt_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): VertexData = {
    //TODO: there seems to be a flaw in this computation ..
    if (isWhite(srcId.toInt, weiss) || weiss == 2 ) {
      //println( "added to vertex: " + srcId.toInt + " key: " + dstId.toInt)
      //src_data.phi_tt_g_tt += ( (dstId.toInt, attr.g_tt_phi.dup() ) )

      val new_gtt_phi = attr.g_tt.addColumnVector( src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)) )
        .addRowVector( dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)).transpose())
  //    println( "send_g_tt_phi: added to vertex: " + srcId.toInt + " key: " + dstId.toInt + " with value: " + attr.g_tt_phi.dup() + " new calc " + new_gtt_phi )
      src_data.phi_tt_g_tt += ( (dstId.toInt, new_gtt_phi ) )
  //    println( " phittgtt of " + srcId.toInt + " : " + src_data.phi_tt_g_tt )

      //src_data.phi_tt_g_tt += ((dstId.toInt,
      //  src_data.phi_tt_g_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
      //    .add(attr.g_tt.div(2.0))
      //    .addColumnVector(attr.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)))))
    }
    //if ( srcId.toInt == weiss ) println( src_data.phi_tt_g_tt )
    /*else {
      dst_data.phi_tt_g_tt += ((srcId.toInt,
        dst_data.phi_tt_g_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(attr.g_tt.rows, attr.g_tt.columns))
          .add(attr.g_tt.div(2.0))
          .addRowVector(attr.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(attr.g_tt.rows)).transpose())))
    }*/
    src_data
  }

  def compute_phi( srcId: VertexId, data : VertexData, weiss: Int ) : VertexData = {
    if ( isWhite(srcId.toInt, weiss) ) {
      //if ( srcId.toInt == 0 ) println( "At of 0: " + data.At )
      //if ( srcId.toInt == 1 ) println( "At")
      for ((k,v) <- data.phi_tt_g_tt ){
  //      if ( srcId.toInt == 0 ) println( "compphi: key: " + k + " old phi: " +data.phi_tt.getOrElse(k, DoubleMatrix.zeros(data.g_t.rows)) + " gttphimins " + v.rowMins() + " At/w " + data.At.dup().div( data.out_degree.toDouble )  )
        data.phi_tt += ((k, data.phi_tt.getOrElse(k, DoubleMatrix.zeros(data.g_t.rows))
                                        .subColumnVector( v.rowMins() )
                                        .addColumnVector( data.At.dup().div( data.out_degree.toDouble ) )) )

      }

 //     println( " phitt of srcid: "  + srcId.toInt + " " + data.phi_tt + "" )
      //if ( srcId.toInt == 0 ) println("phi_tt: " + data.phi_tt)
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

  def isWhite(srcId: Int, weiss: Int): Boolean = {
    ((((srcId % lastColumnId) + (srcId / lastColumnId)) % 2) ) == weiss
  }

  def mapNode(data: VertexData, out_degree: Int, vid: Int): VertexData = {
    data.out_degree = out_degree
    data.vid = vid
    data
  }

}
