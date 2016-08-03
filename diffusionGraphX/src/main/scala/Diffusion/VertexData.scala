package Diffusion

import org.jblas.DoubleMatrix

/**
  * Created by nico on 26.07.2016.
  */
class VertexData(var g_t: DoubleMatrix) extends java.io.Serializable {
  //var data: EdgeData = new EdgeData(g_t, nr_labels, 4) //here: phi_attr = phi_tt', attr = g_t
  //var g_tt = mutable.HashMap.empty[Int,DoubleMatrix]
  //var phi_tt_other = mutable.HashMap.empty[Int,DoubleMatrix]
  //var g_t = g_t_c
  //var phi_tt = mutable.HashMap.empty[Int,DoubleMatrix]
  var noLabels = g_t.getRows()
  var min_sum = DoubleMatrix.zeros(noLabels)
  var min_gtt = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
  // min ( g_tt^phi )
  var out_degree: Int = 0
  //var neighbour_ids = new Array[VertexId](4)
  var phi_tt_g_tt = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
  // phi_tt of edges + half of the edge attribute
  var g_t_phi: DoubleMatrix = DoubleMatrix.zeros(noLabels)
  //for bound computation
  var weight: Double = 0.0
  // var primal: Int = 0
  //def get_min_g_t_phi() : Double =
  //{
  //  this.g_t_phi.min()
  //}
}