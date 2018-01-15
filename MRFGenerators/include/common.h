#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/decomposition/graphicalmodeldecomposer.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

	namespace TST{
	typedef opengm::DiscreteSpace<> Space;
	typedef opengm::ExplicitFunction<double> Function;
	typedef opengm::GraphicalModel<double,
				       opengm::Adder,
				       OPENGM_TYPELIST_5(opengm::ExplicitFunction<double> ,
							 opengm::PottsFunction<double>,
							 opengm::PottsGFunction<double>,
							 opengm::TruncatedSquaredDifferenceFunction<double>,
							 opengm::TruncatedAbsoluteDifferenceFunction<double>) , Space> GraphicalModel;

	}
#endif
