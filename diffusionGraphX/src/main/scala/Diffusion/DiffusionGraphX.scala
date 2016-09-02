package main.scala.Diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot._
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

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf()
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setAppName("GraphX Min-Sum Diffusion")
      .setMaster(args(2))
    val sc = new SparkContext(conf)

    println("Benchmark: " + args(0))
    val benchmark = args(0)
    // TODO: import data of opengm's hdf5 file


    // load edge data (experimental)
    val pwPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/pwFactors.csv")
    val noLabelsOfEachVertex: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/nrLabels.csv")
    var unaryPotentials: DoubleMatrix = DoubleMatrix.loadCSVFile("benchmark/" + benchmark + "/uFactors.csv")
    val lcid = scala.io.Source.fromFile("benchmark/" + benchmark + "/lcid.txt").getLines().next().toInt

    // create graph structure
    val graph = GraphLoader.edgeListFile(sc, "benchmark/" + benchmark + "/edgeListFile.txt",false,args(1).toInt).cache()

    // initialize and run distributed inference algorithm
    val diffInference = new DiffusionGraphX(graph, noLabelsOfEachVertex, unaryPotentials, pwPotentials, lcid)
    diffInference.apply()
  }
}

class DiffusionGraphX(graph: Graph[Int, Int], noLabelsOfEachVertex: DoubleMatrix, unaryPotentials: DoubleMatrix, pwPotentials: DoubleMatrix, lastColumnId: Integer) extends java.io.Serializable {
  val USE_DEBUG_PSEUDO_BARRIER: Boolean = false
  val SHOW_LABELING_IN_ITERATION: Boolean = false


  val maxIt = 49
  val conv_bound = 0.001
  val t_final = System.currentTimeMillis()

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


    // *****
    // Start Min-Sum Diffusion Iteration
    // *****
    val t_start = System.currentTimeMillis()
    for (i <- 0 to maxIt)
    {
      val t_start_iter = System.currentTimeMillis()

      // *****
      // Black
      // *****
      // Compute g_tt_phi
      val newRdd = temp_graph.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 0)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
          msg1
        }
      ).cache()

      if (USE_DEBUG_PSEUDO_BARRIER) println(temp_graph.vertices.count())
      temp_graph = Graph(newRdd, temp_graph.edges).cache()
      //temp_graph = temp_graph.joinVertices(newRdd)( (vid, vd1, vd2) => vd2 ).cache()
      if (USE_DEBUG_PSEUDO_BARRIER) println(temp_graph.vertices.count())

      // Calculate the sum of the minimum pairwise dual variables g_tt_phi_tt
      temp_graph = temp_graph.mapVertices((vid, data) => {
        if ( isWhite(vid.toInt,0) ) {
          if ( i != 0 )  data.At.putColumn(0,data.At.fill(0.))
          for ((k, v) <- data.phi_tt_g_tt) {
            data.At.addiColumnVector(v.rowMins())
          }
        }
        val dat = compute_phi( vid, data, 0 )
        dat
      }  )


      // *****
      // White
      // *****
      // Compute g_tt_phi
      //compute sum of min_g_tt_phi
      val newRdd1 = temp_graph.aggregateMessages[VertexData](edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 1)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
          msg1
        }
      ).cache()

      //TODO Optimize integration of new vertices
      temp_graph = Graph(newRdd1, temp_graph.edges).cache()
      //temp_graph = temp_graph.joinVertices(newRdd1)( (vid, vd1, vd2) => vd2 ).cache()


      if (USE_DEBUG_PSEUDO_BARRIER) println(temp_graph.vertices.count())
      // update phis
      temp_graph = temp_graph.mapVertices((vid, data) => {
        if (isWhite(vid.toInt, 1)) {
          if ( i!= 0 )  data.At.putColumn(0,data.At.fill(0.))
          for ((k, v) <- data.phi_tt_g_tt) {
            data.At.addiColumnVector(v.rowMins())
          }

        }
        val dat = compute_phi( vid, data,1 )
        dat
      }  )

      if (USE_DEBUG_PSEUDO_BARRIER) println(temp_graph.vertices.count())

      // *****
      // BOUND
      // *****
      // Compute g_tt_phi
      val newRddBound = temp_graph.aggregateMessages[VertexData]( edgeContext => {
        val msg = send_g_tt_phi( edgeContext.srcId, edgeContext.dstId, edgeContext.srcAttr, edgeContext.dstAttr, edgeContext.attr, 2)
        edgeContext.sendToSrc( msg )
      },
        (msg1, msg2) => {
          msg1.phi_tt_g_tt.++=(  msg2.phi_tt_g_tt  )
          msg1
        }
      ).cache()

      val bound_graph2 = Graph(newRddBound, temp_graph.edges).cache()

      // Recompute At
      val temp_graph2 = bound_graph2.mapVertices((vid, data) => {
        data.At.putColumn(0, data.At.fill(0.))
        for ((k, v) <- data.phi_tt_g_tt) {
          data.At.addiColumnVector(v.rowMins())
        }
        data.label = data.At.argmin()
        data
      }
      ).cache()
      /*
            // Compute bound as min g_tt_phi (only for black nodes to avoid duplicated edge energy )
            val aggregate_v = bound_graph2.aggregateMessages[Double](triplet => {
              var minimum = 0.0
              if (isWhite(triplet.srcId.toInt, 0)) {
                val new_gtt_phi = triplet.attr.g_tt.addColumnVector(triplet.srcAttr.phi_tt.getOrElse(triplet.dstId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)))
                  .addRowVector(triplet.dstAttr.phi_tt.getOrElse(triplet.srcId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)).transpose())
                minimum = new_gtt_phi.rowMins().min()
                //  println(" minimum g_tt_phi of id " + triplet.srcId.toInt + " "+ minimum + " gttphi " + new_gtt_phi )
              }
              triplet.sendToSrc(minimum)
            },
              (a, b) => a + b).cache()
            if (USE_DEBUG_PSEUDO_BARRIER) println(aggregate_v.count())
            // sum up for bound computation
            bound = aggregate_v.aggregate[Double](zeroValue = 0.0)((double, data) => {
              // println( "double: " + double + " data_2 " + data._2 )
              double + data._2
            }, (a, b) => a + b)*/

      // *****
      // PRIMAL ENERGY
      // *****
      // Compute vertice energy as g_t( argmin A_t )
      val vertice_energy = temp_graph2.vertices.aggregate[Double] (zeroValue = 0.0) ((double,data) => {
          double + data._2.g_t.get( data._2.label )
      }, (a,b) => a+b)


      // Compute edge energy as g_tt'( argmin At, argmin At' )
      val edge_energy = temp_graph2.triplets.aggregate[(Double, Double)](zeroValue = (0.0, 0.0))((double, triplet) => {
        var result = double._1
        var minimum = double._2
        if (isWhite(triplet.srcId.toInt, 0)) {
          result += triplet.attr.g_tt.get(triplet.srcAttr.label, triplet.dstAttr.label) //+ triplet.srcAttr.g_t.get(triplet.srcAttr.label) ///2 + triplet.dstAttr.g_t.get(triplet.dstAttr.label)/2
          val new_gtt_phi = triplet.attr.g_tt.addColumnVector(triplet.srcAttr.phi_tt.getOrElse(triplet.dstId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)))
            .addRowVector(triplet.dstAttr.phi_tt.getOrElse(triplet.srcId.toInt, DoubleMatrix.zeros(triplet.srcAttr.g_t.rows)).transpose())
          minimum += new_gtt_phi.rowMins().min()
        }
        (result, minimum)
      }, (a, b) => (a._1 + b._1, a._2 + b._2))
      energy = edge_energy._1 + vertice_energy
      bound = edge_energy._2

      if (USE_DEBUG_PSEUDO_BARRIER) println(temp_graph2.triplets.count())
      if (USE_DEBUG_PSEUDO_BARRIER) println(bound_graph2.vertices.count())

      println(i + " -> E " + energy + " B " + bound + " dt " + (System.currentTimeMillis() - t_start_iter) + "ms ---------------------------------------------")

      if (SHOW_LABELING_IN_ITERATION) {
        val labeling = compute_grid_labeling(temp_graph)
        val labelVisualizer = Figure()
        labelVisualizer.subplot(0) += image(labeling)
        labelVisualizer.subplot(0).title = "Primal solution of iteration " + i.toString
        labelVisualizer.subplot(0).xaxis.setTickLabelsVisible(false)
        labelVisualizer.subplot(0).yaxis.setTickLabelsVisible(false)
      }
    }
    println("runtime " + (System.currentTimeMillis() - t_start) + " ms")

    /*val labeling = compute_grid_labeling(temp_graph)
    val labelVisualizer = Figure()
    labelVisualizer.subplot(0) += image(labeling)
    labelVisualizer.subplot(0).title = "Primal solution"
    labelVisualizer.subplot(0).xaxis.setTickLabelsVisible(false)
    labelVisualizer.subplot(0).yaxis.setTickLabelsVisible(false)*/
  }

  def compute_grid_labeling(g: Graph[VertexData, EdgeData]): DenseMatrix[Double] = {
    val vertexArray = g.mapVertices[Integer]((vid, vertexData) => {
      vertexData.At.argmin() // compute primal solution
    }).vertices.sortByKey().map(elem => elem._2.toDouble /* we only care about our label */).collect()
    val noRows = vertexArray.size / lastColumnId
    DenseVector(vertexArray).toDenseMatrix.reshape(lastColumnId, noRows)
  }

  def send_g_tt_phi(srcId: VertexId, dstId: VertexId, src_data: VertexData, dst_data: VertexData, attr: EdgeData, weiss: Int): VertexData = {
    if (isWhite(srcId.toInt, weiss) || weiss == 2 ) {
      val new_gtt_phi = attr.g_tt.addColumnVector(src_data.phi_tt.getOrElse(dstId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)))
        .addRowVector(dst_data.phi_tt.getOrElse(srcId.toInt, DoubleMatrix.zeros(src_data.g_t.rows)).transpose())
      src_data.phi_tt_g_tt += ((dstId.toInt, new_gtt_phi))
    }
    src_data
  }

  def isWhite(srcId: Int, weiss: Int): Boolean = {
    ((((srcId % lastColumnId) + (srcId / lastColumnId)) % 2)) == weiss
  }

  def compute_phi( srcId: VertexId, data : VertexData, weiss: Int ) : VertexData = {
    if ( isWhite(srcId.toInt, weiss) ) {
      for ((k,v) <- data.phi_tt_g_tt ){
        data.phi_tt += ((k, data.phi_tt.getOrElse(k, DoubleMatrix.zeros(data.g_t.rows))
          .subColumnVector(v.dup().rowMins())
                                        .addColumnVector( data.At.dup().div( data.out_degree.toDouble ) )) )
      }
    }
    data
  }

  def mapNode(data: VertexData, out_degree: Int, vid: Int): VertexData = {
    data.out_degree = out_degree
    data.vid = vid
    data
  }
}
