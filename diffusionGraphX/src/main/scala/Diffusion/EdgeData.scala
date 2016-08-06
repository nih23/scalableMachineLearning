package Diffusion

import org.jblas.DoubleMatrix

/**
  * Created by nico on 26.07.2016.
  */
class EdgeData(val g_tt: DoubleMatrix) extends java.io.Serializable
{
  // g_tt equals g_tt(x,x')
  var min_gtt_phi: DoubleMatrix = DoubleMatrix.zeros(g_tt.rows, g_tt.columns)
  //equals min of g_tt_phi
  var phi_tt = scala.collection.mutable.HashMap.empty[Int,DoubleMatrix]  // has phi_tt' and phi_t't
}