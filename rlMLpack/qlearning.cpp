#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/replay/random_replay.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
using namespace ens;

int main() {
    FFN<MeanSquaredError<>, GaussianInitialization> simpleModel(MeanSquaredError<>(), GaussianInitialization(0,0.001));
    simpleModel.Add<Linear<>>(4,32);
    simpleModel.Add<ReLULayer<>>();
    simpleModel.Add<Linear<>>(32,24);
    simpleModel.Add<ReLULayer<>>();
    simpleModel.Add<Linear<>>(24,8);
    simpleModel.Add<ReLULayer<>>();
    simpleModel.Add<Linear<>>(8,2);

    GreedyPolicy<CartPole> policy(1.0,500,0.15);
    //You can tune memory batch size to see the effect on reward
    int replay_batch_size;
    cout << "replay batch size: ";
    cin >> replay_batch_size;
    int reploy_memory_size = 25000;
    RandomReplay<CartPole> replayMethod(replay_batch_size,reploy_memory_size);

    TrainingConfig config;
    config.StepSize() = 0.01;
    config.Discount() = 0.95;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 150;
    config.DoubleQLearning() = false;
    config.StepLimit() = 300;

    QLearning<CartPole, decltype(simpleModel), ens::AdamUpdate, decltype(policy)> agent(move(config),
    move(simpleModel),move(policy),move(replayMethod));

    arma::running_stat<double> stats;
    size_t current_episode = 0;
    size_t max_iter = 2000;
    size_t requirement;

    cout << "minimun reward: ";
    cin >> requirement;
    while ((current_episode <= max_iter))
    {
       double episode_reward = agent.Episode();
       stats(episode_reward);
       current_episode++;
       cout << "Average episode return: " << stats.mean() << " Episode return: " << episode_reward << endl;
       if (stats.mean() > requirement) {
           agent.Deterministic() = true;
           arma::running_stat<double> test_stats;
           for(size_t i = 0; i<20;i++) {
               test_stats(agent.Episode());
           }
           cout << "Converged with return: " << test_stats.mean() << " after: " << current_episode << " episodes" <<endl;
           break;
       }
    }
    if (current_episode > max_iter) {
        cout << "Fail to converge" << endl;
    }
    return 0;
}