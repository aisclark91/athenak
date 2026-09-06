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
   size_t n_th;
   size_t n_tm;

  public:

    NumericalEjectaData(std::string filename) {

      std::ifstream file(filename);
      if (!file) {
        std::cerr << "Could not open file\n";
        std::exit(EXIT_FAILURE);
      }

      std::string dummy;

      // Discard the "#thsize" and "#tsize" label lines, then read the two
      // integers that follow them: the total number of blocks (n_th) and
      // the number of time samples per block (n_tm).
      for (size_t i = 0; i < 2; i++) {
        if (!std::getline(file, dummy)) {
          std::cerr << "File ended before metadata was finished\n";
          std::exit(EXIT_FAILURE);
        }
      }

      if (!(file >> n_th) || !(file >> n_tm)) {
        std::cerr << "Could not read n_th and n_tm from file\n";
        std::exit(EXIT_FAILURE);
      }

      // Consume the rest of the n_tm line, then discard the remaining
      // 5 column-label lines ("#theta", "#time", "#mdot", "#vinfty",
      // "#temperature").
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

  size_t thsize() {
    return n_th;
  }

  size_t tsize() {
    return n_tm;
  }
};