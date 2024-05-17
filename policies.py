from stable_baselines3 import PPO, DQN
import random

def run_random_policy(init_cash, eval_env, episode_length,  eval_iterations):    
    stock_allocation = []
    stock_price = []
    cummulative_pnl = 0
    for _ in range(episode_length * eval_iterations):
        action = [random.randint(0, 1) for _ in range(len(eval_env.asset_list) + 1)]
        _, _, done, _, _ = eval_env.step(action)
        stock_allocation.append(eval_env.allocation)
        stock_price.append(eval_env.last_prices)
        if done:
            cummulative_pnl += eval_env.current_portfolio_value - init_cash
            obs, info = eval_env.reset()

    return stock_allocation, stock_price, cummulative_pnl

def run_long_and_hold(init_cash, eval_env, episode_length,  eval_iterations, init_distribution=[0.25, 0.25, 0.25, 0.25, 0]):    
    stock_allocation = []
    stock_price = []
    cummulative_pnl = 0
    action = init_distribution
    for _ in range(episode_length * eval_iterations):
        _, _, done, _, _ = eval_env.step(action)
        # do not change current distribution of assets 
        action = eval_env.get_current_allocation()
        stock_allocation.append(eval_env.allocation)
        stock_price.append(eval_env.last_prices)
        if done:
            cummulative_pnl += eval_env.current_portfolio_value - init_cash
            obs, info = eval_env.reset()

    return stock_allocation, stock_price, cummulative_pnl

def run_ppo(train_env, eval_env, logger_callback, init_cash, steps, episode_length, eval_iterations, id):
    model = PPO("MlpPolicy",train_env, n_steps=512, batch_size=128, ent_coef=0.0025, learning_rate=0.001, seed=27,  verbose=1, tensorboard_log=f"runs/{id}")
    model = model.learn(total_timesteps=steps, callback=logger_callback)
    obs, info = eval_env.reset()
    cummulative_pnl = 0


    stock_allocation = []
    stock_price = []

    for _ in range(episode_length * eval_iterations):
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = eval_env.step(action)
        print(f"action: {action}")
        print(f"observation: {obs}\n\n")
        stock_allocation.append(eval_env.allocation)
        stock_price.append(eval_env.last_prices)
        if done:
            cummulative_pnl += eval_env.current_portfolio_value - init_cash
            obs, info = eval_env.reset()

    return stock_allocation, stock_price, cummulative_pnl

def run_dqn(train_env, eval_env, logger_callback, init_cash, steps, episode_length, eval_iterations, id):
    model = DQN("MlpPolicy", train_env, learning_starts=20*episode_length,  verbose=1, tensorboard_log=f"runs/{id}", exploration_final_eps=0.08, exploration_fraction=0.4)
    model = model.learn(total_timesteps=steps, callback=logger_callback)
    obs, info = eval_env.reset()
    cummulative_pnl = 0


    stock_allocation = []
    stock_price = []

    for _ in range(episode_length * eval_iterations):
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = eval_env.step(action)
        print(f"action: {action}")
        print(f"observation: {obs}\n\n")
        stock_allocation.append(eval_env.allocation)
        stock_price.append(eval_env.last_prices)
        if done:
            cummulative_pnl += eval_env.current_portfolio_value - init_cash
            obs, info = eval_env.reset()

    return stock_allocation, stock_price, cummulative_pnl