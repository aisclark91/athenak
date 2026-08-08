#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "athena.hpp"

struct Block {
  Real th;
  std::vector<Real> time;
  std::vector<Real> v_infty;
  std::vector<Real> mdot;
  std::vector<Real> temperature;
};

class NumericalEjectaData {
  private:
   std::vector<Block> data;

  public:

    NumericalEjectaData(std::string filename, const size_t &n_th, const size_t n_tm) {

      std::ifstream file(filename);
      if (!file) {
        std::cerr << "Could not open file\n";
        std::exit(EXIT_FAILURE);
      }

        // Discard the first 9 metadata lines
      std::string dummy;
      for (size_t i = 0; i < 6; i++) {
        if (!std::getline(file, dummy)) {
          std::cerr << "File ended before metadata was finished\n";
          std::exit(EXIT_FAILURE);
        }
      }

      auto readArray = [&file](std::vector<double>& v, size_t n_tm) -> bool {
        v.resize(n_tm);
        for (size_t i = 0; i < n_tm; ++i) {
            if (!(file >> v[i])) return false;
        }
        return true;
      };

      while (true) {
        Block b;
        std::string leftover;
        std::getline(file, leftover);
        if (!(file >> b.th)) break;
        if (!readArray(b.time, n_tm)) break;
        if (!readArray(b.v_infty, n_tm)) break;
        if (!readArray(b.mdot, n_tm)) break;
        if (!readArray(b.temperature, n_tm)) break;
        data.push_back(std::move(b));
      }
  };

  std::vector<Block> ComputeBlocks() {
    return data;
  }
};