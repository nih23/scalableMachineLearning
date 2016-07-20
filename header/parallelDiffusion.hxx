#include <iostream>
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx>

#include <limits>
#include <ctime>
#include <map>
#include "common.h"

template<class GM, class ACC>
class ParallelDiffusion : public opengm::Inference<GM, ACC>
{
public:
	using FactorType = typename GM::FactorType;
	using IndexType = typename GM::IndexType;
	using ValueType = typename GM::ValueType;
	using LabelType = typename GM::LabelType;


	using uIterator = double*;
	using UnaryFactor = std::vector<double>;
	using VecUnaryFactors = std::vector<UnaryFactor>;
	using VecFactors = std::vector<FactorType>;
	using VecFactorIndices = std::vector<IndexType>;
	using VecNeighbourIndices = std::vector<IndexType>;
	using MapFactorindexDualvariable = std::map<IndexType, UnaryFactor>;

	using MapVariableValueType = std::map<IndexType, UnaryFactor>;
	using VariablePair = std::pair<IndexType, IndexType>;
	using MapVariablePairValueType = std::map<VariablePair, UnaryFactor>;


	ParallelDiffusion(GM& gm, size_t maxIterations, ValueType convBound, int nx, int ny);

	std::string name() const override;
	const GM& graphicalModel() const override;
	void updateAllStates();
	ValueType computeEnergy();

	// opengm API functions
	opengm::InferenceTermination infer() override;
	opengm::InferenceTermination arg(std::vector<LabelType>&, const size_t=1) const override;
	ValueType energy();
	ValueType bound();

	int _mpi_myRank;
	int _mpi_commSize;

private:
	GM& _gm;

	std::pair<IndexType, IndexType> computePartitionBounds();
	std::pair<IndexType, IndexType> computePartitionBounds(int rank);
	void diffusionIteration(bool isBlack);
	void diffusionIterationNew(bool onBlackNodes);
	void receiveMPIUpdatesOfDualVariables(bool isBlack);

	IndexType _lastColumnId;
	size_t _maxIterations;
	ValueType _energy;
	ValueType _bound;
	ValueType _convBound;
	std::string _name;

	IndexType _griddx;        //_variableToFactors.insert( std::pair<IndexType, VecFactorIndices>(leftVar, got));
	IndexType _griddy;

	std::vector<LabelType> _states;
	std::vector<ValueType> _gPhis;
	std::vector<ValueType> _weights;

	std::vector<FactorType> _factors;

	/*
	 * new
	 */
	MapVariableValueType _nAt; // Variable x Value*
	MapVariablePairValueType _ndualVars; // (Variable, Variable) x Value*

	/*
	 * new new
	 */
	std::map<IndexType,VecFactorIndices> _variableToFactors;
	std::map<IndexType,VecNeighbourIndices> _variableToNeighbours;
	//std::vector<VecUnaryFactors> _dualVars; //phi_tt
	std::map<IndexType, MapFactorindexDualvariable> _dualVars; // VariableIndex x (FactorIndex x ValueType)
	MapFactorindexDualvariable _phiTT; // HashMap with Layout: FactorIndex x double*

	ValueType computeBound();
	//ValueType computeBoundMPI();
	void computeWeights();
	void computePhi(IndexType factorIndex, IndexType varIndex, uIterator begin, uIterator end);
	void updateState(IndexType varIndex);

	int convertVariableIndexToMPIRank(IndexType variableIndex);
	void outputDualvars();
	bool blackAndWhite(IndexType index);


	/*************************************
	 *************************************
	 *************************************
	 * NEW DIFFUSION IMPLEMENTATION
	 *************************************
	 *************************************
	 *************************************/
	UnaryFactor computeAt(IndexType variableIndex);
	ValueType computeGttPhi(IndexType factorIndex, LabelType xt, LabelType xtt, bool switch_xt_xtt);
	ValueType computeMinGttPhi(IndexType varIdx, IndexType factorIndex, LabelType xt);
	double SumNeighboursMinGttPhi(IndexType variableIndex, LabelType xt);
	double getValueOfUnaryFactor(IndexType variableId, LabelType xt);
	/*************************************
	 *************************************
	 *************************************
	 * END OF NEW DIFFUSION IMPLEMENTATION
	 *************************************
	 *************************************
	 *************************************/
	template<class Iterator>
	    typename GM::ValueType getFactorValue(IndexType factorIndex, IndexType varIndex, Iterator it) {
	        FactorType factor = _gm[factorIndex];
	        ValueType result = 0;
	        if (factor.numberOfVariables()>1)
	        {
	            result = factor(it);
	            for (IndexType varId=0; varId<factor.numberOfVariables(); ++varId)
	            {
	                IndexType globalVarId = _gm[factorIndex].variableIndex(varId);
	                UnaryFactor uf = _dualVars[globalVarId][factorIndex];
	                result+=uf[*(it+varId)]; // TODO: the access to the unary factors seems to be broken.
	            }
	        } else {
	            result = getVariableValue(varIndex,*it);

	        }

	        return result;
	    }

	    //g^{phi}_{tt} - switched to g_{t}^{phi}
	    typename GM::ValueType getVariableValue(IndexType variableIndex, LabelType label)
	    {
	        ValueType result = 0.;
	        for (IndexType i=0; i<_gm.numberOfFactors(variableIndex); ++i)
	        {
	            IndexType factorIndex = _gm.factorOfVariable(variableIndex,i);
	            if (_gm[factorIndex].numberOfVariables()==1)
	            {
	                result += _gm[factorIndex](&label);
	                continue;
	            }
	            result -= _dualVars[variableIndex][factorIndex][label]; //switched to plus
	        }

	        return result;
	    }
};
