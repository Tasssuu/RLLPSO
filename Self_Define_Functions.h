/*
This file contains the main self-defined functions for the implementation of the proposed algorithm in the following paper:

Qiang Yang, Wei-Neng Chen, Jeremiah D. Deng, Yun Li, Tianlong Gu and Jun Zhang, "Level-based Learning Swarm Optimizer for Large Scale Optimization",
IEEE Transactions on Evolutionary Computation, conditionally accepted, 2017.
*/


#ifndef SELF_DEFINE_FUNCTIONS_H_INCLUDED
#define SELF_DEFINE_FUNCTIONS_H_INCLUDED


#include "./CEC2010/Header.h"
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include "matplotlibcpp.h"
#include <unordered_map>
#include <utility>
#include <boost/functional/hash.hpp>
// Hash function for pair<int, int> to use as a key in unordered_map
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return boost::hash_value(pair);
    }
};


const int dim = 1000; // the dimension size
const int timesOfRun = 1;// the number of independent runs

//the main parameter settings

const int Population_size = 500; // the population size

const int rand_level_set[] = {4,6,8,10,20,50};   //the pool of the number of levels
const int rand_level_size = 6;

const int MAX_FV= 3000*dim; // the maximum number of fitness evaluations
//const int MAX_FV= 3000*dim; // the maximum number of fitness evaluations

const double phi = 0.4; // the control parameter

// Define Q-table and other parameters
const double alpha = 0.4;  // learning rate
const double discount_factor = 0.8;  // gamma
const double epsilon = 0.1;  // exploration rate

struct NewType
{
    double data;
    int id;
};


Benchmarks* generateFuncObj(int funcID);
void Fitness( double &results, double *particle,  int &FV, Benchmarks* fp );

void Fitness_Computation( double *results, double **population, int &gbest, int num, int dim, int &FV, Benchmarks* fp );

void Ranking( double *results, int **levels, int level_num, int level_size, int last_level_size, int NP );

void Update_Particle( double *particle, double *speed, double *exemplar1, double *exemplar2, int dim, double phi, Benchmarks *fp );

int Select_Level_Num (  double* group_prob, int num );

void Compute_Probablity( double *record, double *prob, int num );

double Compute_Relative_Performance( double pre_result, double current_result );
// グラフプロット関数の宣言
void plot_level_data(const vector<vector<double>>& data1, const vector<vector<double>>& data2, const vector<vector<int>>& selections, const string& title, const string& xlabel, const string& ylabel, const string& filename, const string& dir_path);

double get_Q_value(int state, int action);

int select_action(int state, int level_size);

void initialize_Q_table(int level_size);

void update_Q_table(int state, int action, double reward, int level_size);

#endif // SELF_DEFINE_FUNCTIONS_H_INCLUDED
