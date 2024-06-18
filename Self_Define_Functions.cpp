
#include "Self_Define_Functions.h"
#include "matplotlibcpp.h"
#include <filesystem>
#include <unordered_map>

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

std::unordered_map<std::pair<int, int>, double, pair_hash> Q_table;

// create new object of class for different functions
Benchmarks* generateFuncObj(int funcID)
{
	Benchmarks *fp;
	// run each of specified function in "configure.ini"
	if (funcID==1){
		fp = new F1();
	}else if (funcID==2){
		fp = new F2();
	}else if (funcID==3){
		fp = new F3();
	}else if (funcID==4){
		fp = new F4();
	}else if (funcID==5){
		fp = new F5();
	}else if (funcID==6){
		fp = new F6();
	}else if (funcID==7){
		fp = new F7();
	}else if (funcID==8){
		fp = new F8();
	}else if (funcID==9){
		fp = new F9();
	}else if (funcID==10){
		fp = new F10();
	}else if (funcID==11){
		fp = new F11();
	}else if (funcID==12){
		fp = new F12();
	}else if (funcID==13){
		fp = new F13();
	}else if (funcID==14){
		fp = new F14();
	}else if (funcID==15){
		fp = new F15();
	}else if (funcID==16){
		fp = new F16();
	}else if (funcID==17){
		fp = new F17();
	}else if (funcID==18){
		fp = new F18();
	}else if (funcID==19){
		fp = new F19();
	}else if (funcID==20){
		fp = new F20();
	}else{
		cerr<<"Fail to locate Specified Function Index"<<endl;
		exit(-1);
	}
	return fp;
}



bool Compare_NewType( NewType data1, NewType data2 )
{
    return data1.data < data2.data;
}




//calculate the fitness of one particle
void Fitness( double &results, double *particle,  int &FV, Benchmarks* fp )
{
    results =  fp->compute( particle );
    FV++;
}


//calculate the fitness of one swarm
void Fitness_Computation( double *results, double **population, int &gbest, int num, int dim, int &FV, Benchmarks* fp )
{// num is the population size and dim is the size of dimensions
    int i;
    double best = results[0] = fp->compute( population[0] );
    gbest = 0;
    for( i = 1; i < num; ++i )
    {
        results[i] = fp->compute( population[i]);

        if( results[i] < best )
        {
            best = results[i];
            gbest = i;

        }
    }

    FV += num;
}



// sort the swarm and then divide the whole population into different levels based on ranking
void Ranking( double *results, int **levels, int level_num, int level_size, int last_level_size, int NP )
{
    int i,j,k;

    NewType *temp = new NewType [NP];
    for( i = 0; i < NP; ++i )
    {
        temp[i].data = results[i];
        temp[i].id = i;
    }

    sort( temp, temp+NP, Compare_NewType );

    k = 0;
    for( i = 0; i < level_num-1; ++i )
    {
        for( j = 0; j < level_size; ++j )
        {
            levels[i][j] = temp[k].id;
            ++k;
        }
    }

    for( j = 0; j < last_level_size; ++j )
    {
        levels[i][j] = temp[k].id;
        ++k;
    }

    delete []temp;
}


//update one particle
void Update_Particle( double *particle, double *speed, double *exemplar1, double *exemplar2, int dim, double phi, Benchmarks *fp )
{
    int i;
    double r1,r2,r3;
    boost::mt19937 generator(time(0)*rand());
    boost::uniform_real<> uniform_real_generate_r( 0, 1 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_r( generator, uniform_real_generate_r );


    for( i = 0; i < dim; ++i )
    {
        r1 = random_real_num_r();
        r2 = random_real_num_r();
        r3 = random_real_num_r();

        speed[i] = r1 * speed[i] + r2 *( exemplar1[i] - particle[i] ) + phi * r3 * ( exemplar2[i] - particle[i] );
        particle[i] += speed[i];

        if( particle[i] < fp->getMinX() )
            particle[i] = fp->getMinX();

        if( particle[i] > fp->getMaxX() )
            particle[i] = fp->getMaxX();
    }
}


//calculate the relative performance improvement
double Compute_Relative_Performance( double pre_result, double current_result )
{
    if(pre_result < 1e-100 )
        return 1;
    else
        return abs( current_result - pre_result ) / abs( pre_result );
}



//compute the probability of each level number
void Compute_Probablity( double *record, double *prob, int num )
{
    int i;
    double sum = 0;
    for( i = 0; i < num; ++i )
    {
        prob[i] = exp(7*record[i]);
        sum += prob[i];
    }

    for( i = 0; i < num; ++i )
    {
        prob[i] /= sum;

    }

}


//randomly select one number using Roulette Wheel Selection
int Select_Level_Num (  double* group_prob, int num )
{

    boost::mt19937 gen(time(0)*rand());
    boost::uniform_real<> unif( 0,1 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > unif_dist( gen, unif );

    double temp = unif_dist();

    int i ;
    if( temp >=0 && temp <=group_prob[0] )
        return 0;
    else
    {
        for( i = 1; i < num; ++i )
            if( temp >group_prob[i-1] && temp <=group_prob[i] )
            {
                break;
            }

        return i;

    }

}

// グラフプロット関数の実装
void plot_level_data(const vector<vector<double>>& data1, const vector<vector<double>>& data2, const vector<vector<int>>& selections, const string& title, const string& xlabel, const string& ylabel, const string& filename, const string& dir_path) {
    for (size_t i = 0; i <= 5; ++i) {
        plt::figure_size(3200, 600);  // Create a new figure with specified size (width, height)
        plt::plot(data1[i], {{"label", "Record Level " + to_string(i)}});
        plt::plot(data2[i], {{"label", "Probability Level " + to_string(i)}});
        for (int sel : selections[i]) {
            plt::scatter(vector<double>{static_cast<double>(sel)}, vector<double>{data1[i][sel]}, 10.0, {{"color", "red"}});
        }
        plt::xlabel(xlabel);
        plt::ylabel(ylabel);  // Set y-axis label for each subplot
        plt::title(title + " - Level " + to_string(i));
        plt::ylim(0.0, 0.5);  // Set y-axis limit with both arguments as double
        plt::legend({{"fontsize", "small"}});  // Set smaller font size for legend
        string filename_with_level = filename + "_level_" + to_string(i) + ".png";
        plt::save(filename_with_level);  // Save the figure
        plt::close();  // Close the figure

        // イテレーションをテキストファイルに保存
        string text_filename = dir_path + "/Level_" + to_string(i) + "_iterations.txt";
        ofstream out_iterations(text_filename.c_str());
        for (int sel : selections[i]) {
            out_iterations << sel << endl;
        }
        out_iterations.close();
    }
}





// Helper function to get Q-value
double get_Q_value(int state, int action) {
    std::pair<int, int> state_action = std::make_pair(state, action);
    if (Q_table.find(state_action) == Q_table.end()) {
        Q_table[state_action] = 0.0;
    }
    return Q_table[state_action];
}

// Helper function to select action using epsilon-greedy policy
int select_action(int state, int level_size) {
    if (static_cast<double>(rand()) / RAND_MAX < epsilon) {
        return rand() % level_size;
    } else {
        int best_action = 0;
        double best_value = get_Q_value(state, 0);
        for (int action = 0; action < level_size; ++action) {
            double q_value = get_Q_value(state, action);
            if (q_value > best_value) {
                best_value = q_value;
                best_action = action;
            }
        }
        return best_action;
    }
}

// Initialize Q-table
void initialize_Q_table(int level_size) {
    for (int state = 0; state < level_size; ++state) {
        for (int action = 0; action < level_size; ++action) {
            Q_table[std::make_pair(state, action)] = 0.0;
        }
    }
}

// Update Q-table
void update_Q_table(int state, int action, double reward,int level_size) {
    double best_next_Q = get_Q_value(action, 0);
    for (int next_action = 0; next_action < level_size; ++next_action) {
        double next_Q = get_Q_value(action, next_action);
        if (next_Q > best_next_Q) {
            best_next_Q = next_Q;
        }
    }
    Q_table[std::make_pair(state, action)] += alpha * (reward + discount_factor * best_next_Q - get_Q_value(state, action));
}




