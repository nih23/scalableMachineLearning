#include <iostream>
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx>

#include <boost/serialization/utility.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <boost/thread.hpp>

#include <limits>
#include <ctime>
#include <map>
#include <unordered_map>
#include "common.h"

template<class GM, class ACC>
class Diffusion_MPI_Rewrite : public opengm::Inference<GM, ACC>
{
public:
    using FactorType = typename GM::FactorType;
    using IndexType = typename GM::IndexType;
    using ValueType = typename GM::ValueType;
    using LabelType = typename GM::LabelType;

    using uIterator = ValueType*;
    using UnaryFactor = std::vector<ValueType>;
    using VecUnaryFactors = std::vector<UnaryFactor>;
    using VecFactors = std::vector<FactorType>;
    using VecFactorIndices = std::vector<IndexType>;
    using VecNeighbourIndices = std::vector<IndexType>;
    using MapFactorindexDualvariable = std::map<IndexType, UnaryFactor>;

    Diffusion_MPI_Rewrite(GM& gm, size_t maxIterations, ValueType convBound, int nx, int ny);

    std::string name() const override;
    const GM& graphicalModel() const override;
    void updateAllStates();
    ValueType computeEnergy();
    ValueType computeEnergySerial();

    // opengm API functions
    opengm::InferenceTermination infer() override;
    opengm::InferenceTermination arg(std::vector<LabelType>&, const size_t=1) const override;
    ValueType energy();
    ValueType bound();

    int _mpi_myRank;
    int _mpi_commSize;

private:
    boost::mpi::communicator _mpi_world_comm;    

    GM& _gm;

    std::pair<IndexType, IndexType> computePartitionBounds();
    std::pair<IndexType, IndexType> computePartitionBounds(int rank);
    void diffusionIteration(bool isBlack);
    void receiveMPIUpdatesOfDualVariables(bool isBlack);
    void receiveMPIUpdatesOfDualVariablesBOOST(bool isBlack);
    void sendMPIUpdatesOfDualVariables(bool isBlack);

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
    std::unordered_map<IndexType,VecFactorIndices> _variableToFactors;
    std::unordered_map<IndexType,VecNeighbourIndices> _variableToNeighbours;
    //std::vector<VecUnaryFactors> _dualVars; //phi_tt
    std::unordered_map<IndexType, MapFactorindexDualvariable> _dualVars; // VariableIndex x (FactorIndex x ValueType)

    ValueType computeBound();
    ValueType computeBoundMPI();
    void computeWeights();
    void computePhi(IndexType factorIndex, IndexType varIndex, uIterator begin, uIterator end);
    void updateState(IndexType varIndex);

    //int convertVariableIndexToMPIRank(IndexType variableIndex);
    void outputDualvars();

    int convertVariableIndexToMPIRank(IndexType variableIndex) {
        int partitionSz = _gm.numberOfVariables() / _mpi_commSize;
        return variableIndex / (partitionSz);
    }


    bool blackAndWhite(IndexType index)
    {
        return (((index%_lastColumnId)+(index/_lastColumnId))%2==0);      
    }

    template<class Iterator>
    typename GM::ValueType 
    getFactorValue(IndexType factorIndex, IndexType varIndex, Iterator it) {
        const typename GM::FactorType& factor = _gm[factorIndex];
        ValueType result = 0;
        if (factor.numberOfVariables()>1)
        {
            result = factor(it);
            for (IndexType varId=0; varId<factor.numberOfVariables(); ++varId)
            {
                IndexType globalVarId = _gm[factorIndex].variableIndex(varId);
                UnaryFactor uf = _dualVars[globalVarId][factorIndex];
                result += uf[*(it+varId)];
            }
        } else {
            result = getVariableValue(factor.variableIndex(0),*it);
        }

        return result;
    }

    //g^{phi}_{tt} - switched to g_{t}^{phi}
    typename GM::ValueType 
    getVariableValue(IndexType variableIndex, LabelType label)
    {
        ValueType result = 0.;
        ValueType tmpRes = 0.;
        for (IndexType i=0; i<_gm.numberOfFactors(variableIndex); ++i)
        {
            IndexType factorIndex = _gm.factorOfVariable(variableIndex,i);
            if (_gm[factorIndex].numberOfVariables()==1)
            {
                result += _gm[factorIndex](&label);
                continue;
            }
          
            tmpRes = result;
            result -= _dualVars[variableIndex][factorIndex][label]; //switched to plus

        }
        return result;
    }
};
