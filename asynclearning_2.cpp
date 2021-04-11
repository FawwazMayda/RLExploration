#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/reinforcement_learning/async_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/policy/aggregated_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <ensmallen.hpp>
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
int main()
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(), GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);
  AggregatedPolicy<GreedyPolicy<CartPole>> policy({GreedyPolicy<CartPole>(0.7, 5000, 0.1),
                                                   GreedyPolicy<CartPole>(0.7, 5000, 0.01),
                                                   GreedyPolicy<CartPole>(0.7, 5000, 0.5)},
                                                   arma::colvec("0.4 0.3 0.3"));
  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.9;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;
  OneStepQLearning<CartPole, decltype(model), ens::VanillaUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy));
  arma::vec returns(20, arma::fill::zeros);
  size_t position = 0;
  size_t episode = 0;
  auto measure = [&returns, &position, &episode](double episodeReturn)
  {
    if(episode > 10000) return true;
    returns[position++] = episodeReturn;
    position = position % returns.n_elem;
    episode++;
    std::cout << "Episode No.: " << episode
        << "; Episode Return: " << episodeReturn
        << "; Average Return: " << arma::mean(returns) << std::endl;
  };
  for(size_t i = 0;i < 2000;i++) {
    agent.Train(measure);
  }
  std::cout << "Total episode: " << episode << std::endl;
}