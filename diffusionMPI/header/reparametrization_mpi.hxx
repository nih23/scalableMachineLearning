#ifndef REPARAMETRISATION_STORAGE_HXX
#define REPARAMETRISATION_STORAGE_HXX


#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx>


//dual_variables = phi_tt
//factor = //g^{phi}_{t}
//variable = //g^{phi}_{tt}

template<class GM>
class ReparametrisationStorageMPI {
public:
    using GraphicalModelType = GM;
    using ValueType = typename GM::ValueType;
    using FactorType = typename GM::FactorType;
    using IndexType = typename GM::IndexType;
    using LabelType = typename GM::LabelType;

    using PairwiseFactor = std::vector<FactorType>;
    using UnaryFactor = std::vector<ValueType>;
    using uIterator = ValueType*;
    using VecUnaryFactors = std::vector<UnaryFactor>;
    //using VecPairwiseFactors = std::vector<FactorType>;
    using VecPairwiseFactors = FactorType;
    using VarIdMapType = std::map<IndexType,IndexType>;

    //static const IndexType InvalidIndex;
    ReparametrisationStorageMPI(const GM& gm);

    const UnaryFactor& get(IndexType factorIndex, IndexType relativeVarIndex) const
    {
        OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
        OPENGM_ASSERT(relativeVarIndex < dualVars_[factorIndex].size());
        return dualVars_[factorIndex][relativeVarIndex];
    }

    std::pair<uIterator,uIterator> getIterators(IndexType factorIndex, IndexType relativeVarIndex)
    {
        OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
        OPENGM_ASSERT(relativeVarIndex < dualVars_[factorIndex].size());
        UnaryFactor& uf = dualVars_[factorIndex][relativeVarIndex];
        uIterator begin = &uf[0];
        return std::make_pair(begin, begin+uf.size());
    }

    // it -> instance of label
    //g^{phi}_{t} - switched to g_{tt}^{phi}
    template<class Iterator>
    ValueType getFactorValue(IndexType factorIndex, LabelType it) const
    {
        OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
        const typename GM::FactorType& factor = gm_[factorIndex];
        ValueType result = 0;
        if (factor.numberOfVariables()>1)
        {
            result = factor(it);
            for (IndexType varId=0; varId<factor.numberOfVariables(); ++varId) // iterate over neighbours?
            {
                OPENGM_ASSERT(varId < dualVars_[factorIndex].size());
                OPENGM_ASSERT(*(it+varId) < dualVars_[factorIndex][varId].size());
		//if(factorIndex == 14480)
		//  std::cout << "!" << gm_[factorIndex].variableIndex(varId) << std::endl;
                result+=dualVars_[factorIndex][varId][*(it+varId)]; //switched to minus
                //result+=dualVars_[factorIndex][varId][*(it)];;
                //if(factor.variableIndex(0) == 4800)
		//  std::cout << "SEQ " << factorIndex << " " << factor.variableIndex(varId) << " " << result << std::endl;
                
            }
        }
        else
        {
            result = getVariableValue(factor.variableIndex(0),*it);
        }
        //if( (factorIndex == 14479) && (factor.variableIndex(0) == 4800))
        //std::cout << "FactorValue: " << result << std::endl;
        return result;
    }

    //g^{phi}_{tt} - switched to g_{t}^{phi}
    ValueType getVariableValue(IndexType variableIndex, LabelType label) const
    {
        OPENGM_ASSERT(variableIndex < gm_.numberOfVariables());
/*
        if( variableIndex == 10998 ) {

            std::cout << "**** gVV @ " << variableIndex << std::endl;
        }
     */ ValueType result = 0.;
        for (IndexType i=0; i<gm_.numberOfFactors(variableIndex); ++i)
        {
            IndexType factorIndex = gm_.factorOfVariable(variableIndex,i);
            /*if( variableIndex == 10998 && factorIndex == 33225 ) {
                std::cout << "Starting update of 10998 33225" << std::endl;
            }*/
            OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
            if (gm_[factorIndex].numberOfVariables()==1)
            {
                result += gm_[factorIndex](&label);
               // if( variableIndex == 10998 ) std::cout << "res1v " << factorIndex << " " << result << std::endl;
                continue;
            }
            OPENGM_ASSERT( factorIndex < dualVars_.size() );
            OPENGM_ASSERT(label < dualVars_[factorIndex][localIndex(factorIndex,variableIndex)].size());
    /*        if(variableIndex == 10998 && factorIndex == 33225 ) {
                std::cout << "dv for " << variableIndex << " " << factorIndex << " ";
                for(ValueType vt : dualVars_[factorIndex][localIndex(factorIndex,variableIndex)])
                    std::cout << vt << " ";
                std::cout << std::endl;
            }*/
            result -= dualVars_[factorIndex][localIndex(factorIndex,variableIndex)][label]; //switched to plus
  //          if( variableIndex == 10998 ) std::cout << "resMultV " << factorIndex << " " << result << std::endl;

        }
/*
        if( variableIndex == 10998 ) {

            std::cout << " ===> " << result << std::endl;
        }*/
        return result;
    }

    IndexType localIndex(IndexType factorIndex, IndexType variableIndex)const
    {
        typename VarIdMapType::const_iterator it = localIdMap_[factorIndex].find(variableIndex);
        //typename trws_base::exception_check(it!= localIdMap_[factorIndex].end(),"ReparametrisationStorage:localIndex() - factor and variable are not connected!");
        return it->second;
    };

    const GM& graphicalModel()const {
        return gm_;
    }

private:
    const GM& gm_;
    std::vector<VecPairwiseFactors> factors_; //g_tt'
    std::vector<VecUnaryFactors> dualVars_; //phi_tt'
    std::vector<VarIdMapType> localIdMap_;
};

//template <class GM>
//const typename ReparametrisationStorage<GM>::IndexType ReparametrisationStorage<GM>::InvalidIndex=std::numeric_limits<IndexType>::max();


template<class GM>
ReparametrisationStorageMPI<GM>::ReparametrisationStorageMPI(const GM& gm) : gm_(gm), localIdMap_(gm.numberOfFactors())
{
    factors_.resize(gm_.numberOfFactors());
    dualVars_.resize(gm_.numberOfFactors());
    // for all factors with order >1
    for (IndexType factorIndex=0; factorIndex<gm_.numberOfFactors(); ++factorIndex)
    {
        IndexType numVars = gm_[factorIndex].numberOfVariables();
        VarIdMapType& mapFactorIndex = localIdMap_[factorIndex];
        if (numVars >= 2)
        {
            //factors_[factorIndex].resize(numVars);
            dualVars_[factorIndex].resize(numVars);
            std::vector<IndexType> v(numVars);
            gm_[factorIndex].variableIndices(&v[0]);

            //TODO remove debug output
            const typename GM::FactorType& factor = gm_[factorIndex];
            IndexType noVars = factor.numberOfVariables();

            factors_[factorIndex] = factor;

            for (IndexType i=0; i<numVars; ++i)
            {

                //std::cout << v[i] << std::endl;
                dualVars_[factorIndex][i].resize(gm_.numberOfLabels(v[i]));

                //factors_[factorIndex][i].resize(gm_.numberOfLabels(v[i]));
                mapFactorIndex[v[i]] = i;
            }

            //std::cout << factorIndex << " " << factor(0) << " " << factor(1) << " " <<  noVars << std::endl;
        }
    }
}


#endif // REPARAMETRISATION_STORAGE_HXX