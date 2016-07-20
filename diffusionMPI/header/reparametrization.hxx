#ifndef REPARAMETRISATION_STORAGE_HXX
#define REPARAMETRISATION_STORAGE_HXX


#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx>


//dual_variables = phi_tt
//factor = //g^{phi}_{t}
//variable = //g^{phi}_{tt}  

template<class GM>
class ReparametrisationStorage{
public:
    using GraphicalModelType = GM;
    using ValueType = typename GM::ValueType;
    using FactorType = typename GM::FactorType;
    using IndexType = typename GM::IndexType;
    using LabelType = typename GM::LabelType;
    
    using UnaryFactor = std::vector<ValueType>;
    using uIterator = ValueType*;
    using VecUnaryFactors = std::vector<UnaryFactor>;
    using VarIdMapType = std::map<IndexType,IndexType>;
    
    //static const IndexType InvalidIndex;
    ReparametrisationStorage(const GM& gm);
    
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
    
    //g^{phi}_{t} - switched to g_{tt}^{phi}
    template<class Iterator>
    ValueType getFactorValue(IndexType factorIndex, Iterator it) const
    {
        OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
        const typename GM::FactorType& factor = gm_[factorIndex];
        
        ValueType result = 0;
        if (factor.numberOfVariables()>1)
        {
            result = factor(it);
            for (IndexType varId=0;varId<factor.numberOfVariables();++varId)
            {
                OPENGM_ASSERT(varId < dualVars_[factorIndex].size());
                OPENGM_ASSERT(*(it+varId) < dualVars_[factorIndex][varId].size());
                result+=dualVars_[factorIndex][varId][*(it+varId)]; //switched to minus
            }
        }
        else
        {
            result = getVariableValue(factor.variableIndex(0),*it);
        }
        return result;
    }
    
    //g^{phi}_{tt} - switched to g_{t}^{phi}
    ValueType getVariableValue(IndexType variableIndex, LabelType label) const
    {
        OPENGM_ASSERT(variableIndex < gm_.numberOfVariables());
        ValueType result = 0.;
        for (IndexType i=0;i<gm_.numberOfFactors(variableIndex);++i)
        {
            IndexType factorIndex = gm_.factorOfVariable(variableIndex,i);
            OPENGM_ASSERT(factorIndex < gm_.numberOfFactors());
            if (gm_[factorIndex].numberOfVariables()==1)
            {
                result += gm_[factorIndex](&label);
                continue;
            }
            OPENGM_ASSERT( factorIndex < dualVars_.size() );
            OPENGM_ASSERT(label < dualVars_[factorIndex][localIndex(factorIndex,variableIndex)].size());
            result -= dualVars_[factorIndex][localIndex(factorIndex,variableIndex)][label]; //switched to plus
        }
        return result;
    }
    
    IndexType localIndex(IndexType factorIndex, IndexType variableIndex)const
    {
        typename VarIdMapType::const_iterator it = localIdMap_[factorIndex].find(variableIndex);
        //typename trws_base::exception_check(it!= localIdMap_[factorIndex].end(),"ReparametrisationStorage:localIndex() - factor and variable are not connected!");
        return it->second;
    };
    
    const GM& graphicalModel()const {return gm_;}
    
    /**
     * 
     * private Variables
     * 
     * */
private:
    const GM& gm_;
    std::vector<FactorType> factors;
    std::vector<VecUnaryFactors> dualVars_; //phi_tt
    std::vector<VarIdMapType> localIdMap_;
};

//template <class GM>
//const typename ReparametrisationStorage<GM>::IndexType ReparametrisationStorage<GM>::InvalidIndex=std::numeric_limits<IndexType>::max();


template<class GM>
ReparametrisationStorage<GM>::ReparametrisationStorage(const GM& gm) : gm_(gm), localIdMap_(gm.numberOfFactors())
{
    dualVars_.resize(gm_.numberOfFactors());
    factors = {};
    
    //size_t vec_both_nodes[] = {0,1}; // would mean label 0 of left and label 1 of right node
    size_t vec_local[] = {0};
    // for all factors with order >1
    for (IndexType factorIndex=0; factorIndex<gm_.numberOfFactors();++factorIndex)
    {
        IndexType numVars = gm_[factorIndex].numberOfVariables();
        VarIdMapType& mapFactorIndex = localIdMap_[factorIndex];
        if (numVars >= 2)
        {
            dualVars_[factorIndex].resize(numVars);
            std::vector<IndexType> v(numVars);
            gm_[factorIndex].variableIndices(&v[0]);
	    
	    const typename GM::FactorType& factor = gm_[factorIndex];
	    IndexType noVars = factor.numberOfVariables();
	    //IndexType left = factor.variableIndicesBegin();

	    //const IndexType noLabels = factor.numberOfLabels();
	    
	    //std::cout << "Factor " << factorIndex << " ";
	    //for (IndexType j = 0; j < noLabels; j++) {
	    std::cout << factorIndex << " " << factor(0) << " " << factor(1) << " " <<  noVars;
	    //}
	    std::cout << std::endl;
	    //factors.push_back( gm_[factorIndex] );
            for (IndexType i=0;i<numVars;++i)
            {
                dualVars_[factorIndex][i].resize(gm_.numberOfLabels(v[i]));
                mapFactorIndex[v[i]] = i;
            }
        }
    }
    std::cout << "no factors: " + factors.size() << std::endl;
}


#endif // REPARAMETRISATION_STORAGE_HXX