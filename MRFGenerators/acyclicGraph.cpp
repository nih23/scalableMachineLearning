#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <cmath>

typedef opengm::SimpleDiscreteSpace<> Space; 
typedef opengm::meta::TypeListGenerator<opengm::ExplicitFunction<double>>::type FunctionTypelist;
typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelist, Space> Model;
 
double unaryPotential(size_t label)
{
   return static_cast<double>(rand()) / RAND_MAX;
}

double pairwisePotential(size_t l1, size_t l2, double lambda)
{
   return lambda * abs(static_cast<double>(l1-l2));
}

int main() {
   // construct a label space with numberOfVariables many variables,
   // each having numberOfLabels many labels
   const size_t numberOfVariables = 40; 
   const size_t numberOfLabels = 5;
   const double lambda_pairwise = 0.01;
   Space space(numberOfVariables, numberOfLabels);

   // construct a graphical model
   Model gm(space);
  
   /*
    * add 1st order functions (unary potential) 
    */
  for(size_t v = 0; v < numberOfVariables; ++v) {
      const size_t shape[] = {numberOfLabels};
      opengm::ExplicitFunction<double> f(shape, shape + 1);
      for(size_t s = 0; s < numberOfLabels; ++s) {
         f(s) = unaryPotential(s);
      }
      Model::FunctionIdentifier fid = gm.addFunction(f);
      size_t variableIndices[] = {v};
      gm.addFactor(fid, variableIndices, variableIndices + 1);
   }
   /*
    * add 2nd order function
    */
   const size_t shape2[] = {numberOfLabels, numberOfLabels};
   opengm::ExplicitFunction<double> f2(shape2, shape2 + 2, 1.0);
   for(size_t s = 0; s < numberOfLabels; ++s) {
      for(size_t l = 0; l < numberOfLabels; ++l) {
         f2(s,l) = pairwisePotential(s,l,lambda_pairwise);
      }
   }
   Model::FunctionIdentifier fid2 = gm.addFunction(f2);
   for(size_t v = 0; v < numberOfVariables - 1; ++v) {
      size_t variableIndices[] = {v, v + 1};
      gm.addFactor(fid2, variableIndices, variableIndices + 2);
   }   

   opengm::hdf5::save(gm, "gm_acyclic.h5", "acyclic-toy-gm"); 

   // loopy belief propagation
   std::cout << "Belief Propagation" << std::endl;
   typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
   typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
   const size_t maxNumberOfIterations = numberOfVariables * 2;
   const double convergenceBound = 1e-7;
   const double damping = 0.0;
   BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
   BeliefPropagation bp(gm, parameter);

   // optimize (approximately)
   BeliefPropagation::VerboseVisitorType visitor;
   bp.infer(visitor);

   // obtain the (approximate) argmin
   std::vector<size_t> labeling(numberOfVariables);
   bp.arg(labeling);


   // TRWS
   typedef opengm::TRWSi<Model,opengm::Minimizer> TRWSi;
   TRWSi::Parameter para(size_t(100));
   para.precision_=1e-12;
   // optimize
   TRWSi trws(gm,para);
   trws.infer();
   // obtain the (approximate) argmin
   std::vector<size_t> l_TRWS;
   trws.arg(l_TRWS);
   std::cout << std::endl << std::endl << "TRWS" << std::endl;
   std::cout << "Energy :  "<<trws.value() << std::endl; 
   std::cout << "State :  ";
}


