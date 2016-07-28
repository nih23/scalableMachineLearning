#include <cstdlib>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>

#include <common.h>

int main(int argc, char* argv[])
{
  using Model = TST::GraphicalModel;
  Model gm;
  opengm::hdf5::load(gm, argv[1], "gm");
  
  std::ofstream nrLabels("nrLabels.csv");
  std::ofstream uFactors("uFactors.csv");
  std::ofstream pwFactors("pwFactors.csv");
  std::ofstream edgeFile("edgeListFile.txt");

  bool notSaved = true;
  size_t oneLabel[1];
  size_t twoLabels[2];  

  IndexType numberFac = _gm.numberOfFactors(0);
  for (auto i=1; i<_gm.numberOfVariables(); ++i)
  {
    if (_gm.numberOfFactors(i) == numberFac)
    {
        _lastColumnId = i+1;
        break;
    }
  }
  std::cout << "LCid: " << _lastColumnId << std::endl;

  for ( auto var=0; var < gm.numberOfVariables(); ++var)
  {
      for (auto factor = 0; factor < gm.numberOfFactors(var); ++factor)
      {
        auto facId = gm.factorOfVariable(var,factor);
        if ( gm.numberOfVariables(facId) == 1) //unary Factor
        {
            for (auto label = 0; label < gm.numberOfLabels(var); ++label)
            {
                oneLabel[0] = label;
            uFactors << gm[facId](oneLabel) << std::endl;
            }
            nrLabels << gm.numberOfLabels(var) << std::endl;
        }
        else
        {
            int var2;
            for (auto nrVar = 0; nrVar < gm.numberOfVariables(facId); ++nrVar)
            {
            var2 = gm.variableOfFactor(facId,nrVar);
            if (var != var2)
              break;
            }
            if ( notSaved ) //only for equal pw functions, yet
            {
              for (auto label = 0; label < gm.numberOfLabels(var); ++label)
              {
              twoLabels[0] = label;
              for (auto label2 = 0; label2 < gm.numberOfLabels(var2); ++label2)
              {
                twoLabels[1] = label2;
                if ( label2 == (gm.numberOfLabels(var2)-1) )
                pwFactors << gm[facId](twoLabels) << std::endl;
                else
                pwFactors << gm[facId](twoLabels) << ",";
              }
              }
              notSaved = false;
            }
            edgeFile << var << " " << var2 << std::endl;
        }
      }
  }

  nrLabels.close();
  uFactors.close();
  pwFactors.close();
  edgeFile.close();
}
