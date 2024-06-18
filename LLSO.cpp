#include "./CEC2010/Header.h"
#include "Self_Define_Functions.h"
#include <sys/time.h>
#include <cstdio>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <string>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <time.h>
#include <filesystem>
#include "matplotlibcpp.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace std;

void Compute_Probabilities_Uniform(double* probability_for_levels, int size) {
    double uniform_value = 1.0 / size;
    for (int i = 0; i < size; ++i) {
        probability_for_levels[i] = uniform_value;
    }
}

void Plot_Fitness_Comparison(vector<double>& fitness_current, vector<double>& fitness_uniform, vector<double>& fitness_rl, const string& func_name) {
    size_t min_size = min({fitness_current.size(), fitness_uniform.size(), fitness_rl.size()});

    if (fitness_current.size() > min_size) {
        fitness_current.resize(min_size);
    }
    if (fitness_uniform.size() > min_size) {
        fitness_uniform.resize(min_size);
    }
    if (fitness_rl.size() > min_size) {
        fitness_rl.resize(min_size);
    }

    vector<double> x(min_size);
    for (size_t i = 0; i < min_size; ++i) {
        x[i] = i + 1;
    }

    plt::figure();
    plt::named_loglog("Current Method", x, fitness_current);
    plt::named_loglog("Uniform Method", x, fitness_uniform);
    plt::named_loglog("Reinforcement Learning Method", x, fitness_rl);
    plt::xlabel("Iteration");
    plt::ylabel("Fitness");
    plt::title("Fitness Comparison for " + func_name);
    plt::legend();
    plt::save("../Results/prob_comp_rl/fitness_comparison_" + func_name + ".png");
}

int main(int argc, char *argv[]) {
    // 必要なディレクトリが存在しない場合に作成する
    fs::create_directories("../Results/level_data/");
    fs::create_directories("../Results/values/");
    fs::create_directories("../Results/images/");

    Benchmarks *fp = NULL;

    int funToRun[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};  // function set 
    int funNum = 20; // the total number of functions to be optimized in CEC'2010 benchmark set

    int i,j,k,t;
    int gl_best = 0;

    double *result_all_run = new double[timesOfRun]; // to record the global best fitness of each run
    double *time_results = new double[timesOfRun]; // to record the total time of each run
    double *final_global_best = new double[dim]; // the final global best solution
    double final_val;
    double pre_final_val;
    double **population = new double*[Population_size];
    double **speed = new double*[Population_size];

    for (i = 0; i < Population_size; ++i) {
        population[i] = new double[dim];
        speed[i] = new double[dim];
    }
    
    double *results = new double[Population_size]; // the fitness results for the whole population
    int level_size;
    int last_level_size;
    int temp_size;
    double *record_for_levels = new double[rand_level_size];
    double *probability_for_levels = new double[rand_level_size];

    double *uniform_probability_for_levels = new double[rand_level_size];
    int FV = 0;
    int selected_particle1, selected_particle2, selected_level1, selected_level2;
    int level_num_index;
    int level_num;
    int **levels = new int*[Population_size];
    for (i = 0; i < Population_size; ++i) {
        levels[i] = new int[Population_size];
    }

    for (k = 0; k < funNum; k++) {
        cout << "Function " << funToRun[k] << " Begins!" << endl;
        char fun[10];
        snprintf(fun, 10, "%d", funToRun[k]);

        /*
        string function_dir = "../Results/level_data/Function_" + string(fun);
        fs::create_directories(function_dir);

        string filename_fitness = "../Results/values/Fitness_result_for_function_" + string(fun) + ".txt";
        ofstream out_fitness(filename_fitness.c_str());

        if (!out_fitness) {
            cerr << "Can not open the file " << filename_fitness << endl;
            exit(1);
        }
        */

        vector<double> fitness_current_method;
        vector<double> fitness_uniform_method;
        vector<double> fitness_rl_method;

        int iteration = 0;

        for (int use_current_method = 0; use_current_method < 3; use_current_method++) {
            fp = generateFuncObj(funToRun[k]);
            boost::mt19937 generator(time(0) * rand());

            boost::uniform_real<> uniform_real_generate_x(fp->getMinX(), fp->getMaxX());
            boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random_real_num_x(generator, uniform_real_generate_x);

            boost::uniform_real<> uniform_real_generate_r(0, 1);
            boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random_real_num_r(generator, uniform_real_generate_r);
            FV = 0;

            // Initialize Q-table if using reinforcement learning method
            if (use_current_method == 2) {
                initialize_Q_table(rand_level_size);
            }

            for (t = 0; t < timesOfRun; t++) {
                cout << "Running the " << t << "th run with method " << use_current_method << endl;

                // initialize the population
                for (i = 0; i < Population_size; ++i){
                    for (j = 0; j < dim; ++j) {
                        population[i][j] = random_real_num_x();
                        speed[i][j] = 0;
                    }
                }

                Fitness_Computation(results, population, gl_best, Population_size, dim, FV, fp);
                pre_final_val = final_val = results[gl_best];
                memcpy(final_global_best, population[gl_best], sizeof(double) * dim);

                for (i = 0; i < rand_level_size; ++i) {
                    record_for_levels[i] = 1;
                }

                iteration = 0;
                int state = 0;
                int action_next;
                while (FV < MAX_FV) {
                    if (use_current_method == 0) {
                        Compute_Probablity(record_for_levels, probability_for_levels, rand_level_size);
                        double* prob = probability_for_levels;
                        for (int i = 1; i < rand_level_size; ++i) {
                            prob[i] += prob[i - 1];
                        }
                        level_num_index = Select_Level_Num(prob, rand_level_size);
                    }
                    else if(use_current_method == 1){
                        Compute_Probabilities_Uniform(uniform_probability_for_levels, rand_level_size);
                        double* prob = uniform_probability_for_levels;
                        for (int i = 1; i < rand_level_size; ++i) {
                            prob[i] += prob[i - 1];
                        }
                        level_num_index = Select_Level_Num(prob, rand_level_size);
                    }
                    else if(use_current_method == 2){
                        action_next = select_action(state, rand_level_size);
                        level_num_index = action_next;
                    }

                    level_num = rand_level_set[level_num_index];

                    // compute the level size of each level and the last level
                    level_size = Population_size / level_num;
                    last_level_size = level_size + Population_size % level_num;

                    // sort the swarm
                    Ranking(results, levels, level_num, level_size, last_level_size, Population_size);

                    // update particles from the lowest level to the third level
                    for (i = level_num - 1; i > 1; --i) {
                        boost::mt19937 generator2(time(0) * rand());
                        boost::uniform_int<> uniform_int_particle(0, level_size - 1);
                        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > random_int_particle(generator2, uniform_int_particle);

                        boost::uniform_int<> uniform_int_level(0, i - 1);
                        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > random_int_level(generator2, uniform_int_level);

                        if (i == level_num - 1)
                            temp_size = last_level_size;
                        else
                            temp_size = level_size;

                        for (j = 0; j < temp_size; ++j) {
                            selected_level1 = random_int_level();
                            selected_level2 = random_int_level();
                            while (selected_level2 == selected_level1) {
                                selected_level2 = random_int_level();
                            }
                            if (selected_level1 > selected_level2) {
                                swap(selected_level1, selected_level2);
                            }
                            selected_particle2 = random_int_particle();
                            selected_particle1 = random_int_particle();
                            Update_Particle(population[levels[i][j]], speed[levels[i][j]], population[levels[selected_level1][selected_particle1]], population[levels[selected_level2][selected_particle2]], dim, phi, fp);
                            Fitness(results[levels[i][j]], population[levels[i][j]], FV, fp);
                            if (results[levels[i][j]] < final_val) {
                                final_val = results[levels[i][j]];
                                memcpy(final_global_best, population[levels[i][j]], sizeof(double) * dim);
                            }
                        }
                    }

                    boost::mt19937 generator3(time(0) * rand());
                    boost::uniform_int<> uniform_int_particle2(0, level_size - 1);
                    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > random_int_particle2(generator3, uniform_int_particle2);

                    for (j = 0; j < level_size; ++j) {
                        selected_particle1 = random_int_particle2();
                        do {
                            selected_particle2 = random_int_particle2();
                        } while (selected_particle1 == selected_particle2);
                        if (results[levels[0][selected_particle2]] < results[levels[0][selected_particle1]])
                            swap(selected_particle1, selected_particle2);
                        Update_Particle(population[levels[i][j]], speed[levels[i][j]], population[levels[0][selected_particle1]], population[levels[0][selected_particle2]], dim, phi, fp);
                        Fitness(results[levels[i][j]], population[levels[i][j]], FV, fp);
                    }
                    record_for_levels[level_num_index] = Compute_Relative_Performance(pre_final_val, final_val);
                    pre_final_val = final_val;

                    if (use_current_method == 0) {
                        fitness_current_method.push_back(final_val);
                    } else if (use_current_method == 1) {
                        fitness_uniform_method.push_back(final_val);
                    } else {
                        double reward = Compute_Relative_Performance(pre_final_val, final_val);
                        update_Q_table(state, action_next, reward, rand_level_size);
                        state = level_num_index;
                        fitness_rl_method.push_back(final_val);
                    }
                    iteration++;
                }
            }
        }
        Plot_Fitness_Comparison(fitness_current_method, fitness_uniform_method, fitness_rl_method, "Function_" + string(fun));
    }

    delete[] result_all_run;
    delete[] time_results;
    delete[] final_global_best;
    for (i = 0; i < Population_size; ++i) {
        delete[] population[i];
        delete[] speed[i];
    }
    delete[] population;
    delete[] speed;
    delete[] results;
    delete[] record_for_levels;
    delete[] probability_for_levels;
    delete[] uniform_probability_for_levels;
    for (i = 0; i < Population_size; ++i) {
        delete[] levels[i];
    }
    delete[] levels;

    return 0;
}