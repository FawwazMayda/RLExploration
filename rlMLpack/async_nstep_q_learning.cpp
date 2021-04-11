#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/reinforcement_learning/async_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/aggregated_policy.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
using namespace ens;

int main() {
    FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(), GaussianInitialization(0, 0.001));
    model.Add<Linear<>>(4, 128);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(128, 128);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(128, 2);

    //Aggregated Policy
    AggregatedPolicy<GreedyPolicy<CartPole>> policy_pool({
        GreedyPolicy<CartPole>(0.7,5000,0.1),
        GreedyPolicy<CartPole>(0.7,5000,0.02),
        GreedyPolicy<CartPole>(0.7,5000,0.5)
    },arma::colvec("0.4 0.3 0.3"));

    TrainingConfig config;
    config.StepSize() = 0.01;
    config.Discount() = 0.9;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 200;

    NStepQLearning<CartPole,decltype(model),ens::VanillaUpdate,decltype(policy_pool)> 
    agent(move(config),move(model),move(policy_pool));

    arma::vec rewards(20,arma::fill::zeros);
    size_t pos = 0;
    size_t episode = 0;
    auto measure = [&rewards,  &pos, &episode](double reward) {
        size_t max_episode = 5000;
        if (episode > max_episode) return true;
        episode++;
        rewards[pos++] = reward;
        pos %= rewards.n_elem;
        double averageReward = arma::mean(rewards);
        cout << "Episode: " << episode << " Average return: " << averageReward << " Episode return: " << reward << endl;
    };

    for(size_t i = 0; i<2000;i++) {
        agent.Train(measure);
    }
    return 0;
}