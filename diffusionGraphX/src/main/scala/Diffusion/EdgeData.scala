package Diffusion

import org.jblas.DoubleMatrix

/**
  * Created by nico on 26.07.2016.
  */
class EdgeData(attrc: DoubleMatrix, nr_labels: Int) extends java.io.Serializable
{
  var attr: DoubleMatrix = attrc  //equals g_tt(x,x')
  var min_gtt_phi: DoubleMatrix = DoubleMatrix.zeros(attrc.rows,attrc.columns) //equals min of g_tt_phi
  var phi_tt = scala.collection.mutable.HashMap.empty[Int,DoubleMatrix]  // has phi_tt' and phi_t't
}