import gymnasium as gym
import numpy as np
import random
import math
import utils

class SimpleTradingEnv(gym.Env):
    def __init__(self, asset_list, asset_data, episode_length, seed, cash=100000, eval=False, log=False):
        self.log = log
        self.asset_names = asset_list
        self.n_assets = len(asset_list)
        self.data = asset_data
        self.data_length = len(self.data[asset_list[0]])
        self.episode_length = episode_length
        self.cash = cash
        self.init_cash = cash
        self.transaction_cost = 0.002
        self.seed = seed
        # spaces
        self.action_space = gym.spaces.Discrete(2 ** (self.n_assets))
        self.observation_space = self._set_observation_space()
        # self.reward_range = (-1*episode_length, episode_length + 100)

        # multiobjective
        self.reward_dim = 2
        self.reward_space = gym.spaces.Box(low=np.array([-1*episode_length, -5]), high=np.array([1*episode_length, 0]), shape=(2,))


        self._set_env_variables()

    def _set_env_variables(self):
        random.seed(self.seed)
        self.data_i = random.randrange(self.data_length - self.episode_length - 1)
        self.initial_value = self.init_cash
        self.portfolio_value = self.init_cash
        self.cash = self.init_cash
        self.assets_allocation = [0]*self.n_assets
        self.shares = [0]*self.n_assets
        self.episode_step = 0
        self.low = [np.inf]*self.n_assets
        self.high = [0]*self.n_assets
        self.pnl = 0
        self.daily_pnl = 0
        self.daily_pnl_history = []
        # for differential sharpe ratio
        self.log_returns = []
        self.prev_A = 0
        self.prev_B = 0
    
    def reset(self, seed=13, **kwargs):
        self._set_env_variables()
        obs = self._get_observation()
        info = {}
        return obs, info

    def _set_observation_space(self):
        # trend indicator 0 - no trend, 1 - positive, 2 - negative
        dim = self.n_assets
        # low/high asset prices
        dim += 2*self.n_assets
        # current asset price
        dim += self.n_assets

        observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(dim,), dtype=np.float32)

        return observation_space
    
    def _return_trend(self, data):
        if sorted(data) == data:
            return 1
        if sorted(data, reverse=True) == data:
            return 2
        
        return 0
    
    def _get_observation(self):
        trend_window = 3
        obs = []
        for i in range(self.n_assets):
            index = self.data_i + self.episode_step
            asset_data = self.data[self.asset_names[i]].iloc[index-trend_window:index,0].tolist()
            trend = self._return_trend(asset_data[-trend_window:])
            current_price = self.get_asset_price(i)
            if self.low[i] > current_price:
                self.low[i] = current_price
            if self.high[i] < current_price:
                self.high[i] = current_price
            obs += [trend, current_price, self.low[i], self.high[i]]

        obs = np.asarray(obs)

        return obs
    
    def _get_differential_sharpe_ratio(self):
        time_decay = 0.004
        delta_A = time_decay*(self.log_returns[-1] - self.prev_A)
        delta_B = time_decay*(self.log_returns[-1]**2 - self.prev_B)
        Dt = (self.prev_B*delta_A - 0.5*self.prev_A*delta_B)/pow((self.prev_B - self.prev_A**2), 3/2)
        self.prev_A = self.prev_A + delta_A
        self.prev_B = self.prev_B + delta_B

        A = np.mean(np.asarray(self.log_returns[:-1]))
        B = np.mean(np.asarray(self.log_returns[:-1])**2)
        if A == 0 and B == 0:
            return 0
        
        delta_A = self.log_returns[-1] - A
        delta_B = self.log_returns[-1]**2 - B
        Dt = (B*delta_A - 0.5*A*delta_B) / (B-A**2)**(3/2)

        
        Dt*= time_decay

        #print(f"Dt: {Dt}")

        return Dt
    
    def _set_sharpe_ratio_vars(self):
        returns = np.asarray(self.log_returns[:-1])
        self.prev_A = np.mean(returns)
        self.prev_B = np.mean(returns**2)
    
    def _get_reward_sharpe(self):
        if self.episode_step < 5:
            return self.daily_pnl
        elif self.episode_step == 5:
            self._set_sharpe_ratio_vars()

        return self._get_differential_sharpe_ratio()
    
    def _get_reward(self, terminated):
        # if making money for couple of days in a row, give additional reward
        if terminated:
            if self.pnl > 0:
                return (((self.portfolio_value/self.init_cash) - 1) * 100) ** 2
            else:
                return -10

        return self.daily_pnl*100
    
    def compute_portfolio_volatility(self):

        assets_data = {}
        obs_window_in_days = 30
        df_i = self.data_i + self.episode_step
        diff = obs_window_in_days - df_i
    
        if diff <= 0:
            for asset_name in self.asset_names:
                assets_data[asset_name] = np.take(self.data[asset_name].iloc[df_i - obs_window_in_days : df_i].values, 0, axis=1).tolist()
        elif df_i == 0:
            assets_data[asset_name]= [0]*obs_window_in_days*len(self.n_assets)
        else:
            for asset_name in self.asset_names:
                assets_data[asset_name] = [0]*diff + np.take(self.data[asset_name].iloc[0:df_i].values, 0, axis=1).tolist()

        #1 create covariance matrix
        cov_matrix = utils.get_covariance_matrix(assets_data)
        
        #2 compute variance
        w = self.assets_allocation # without cash
        p_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

        #3 take a square root of it
        p_sd = np.sqrt(p_var)

        # annualize
        annual_p_sd = p_sd * 16

        return annual_p_sd
    
    def _get_risk_reward(self):
        return -10 * self.compute_portfolio_volatility()

    def _get_morl_reward(self, terminated):
        profit_reward = self._get_reward(terminated)
        risk_reward = self._get_risk_reward()

        return np.array([profit_reward, 0.01 * risk_reward])
    
    def get_asset_price(self, asset_id):
        asset = self.asset_names[asset_id]
        df_i = self.data_i + self.episode_step
        return self.data[asset].iloc[df_i].values[0]
    
    def _action_to_vec(self, action):
        action = [int(x) for x in bin(action)[2:].zfill(self.n_assets)]
        if sum(action) == 0:
            return [0] * self.n_assets
        else:
            return [x/sum(action) for x in action]
    
    def _sell(self, stocks):
        # stocks is a list of tuples [(index, diff)]
        # diff is in %
        for s_tuple in stocks:
            i, new_shares = s_tuple
            shares_to_sell = self.shares[i] - new_shares
            money_inflow = shares_to_sell * self.get_asset_price(i) * (1 - self.transaction_cost)
            self.shares[i] -= shares_to_sell
            self.cash += money_inflow
        
    def _buy(self, stocks):
        for s_tuple in stocks:
            i, new_shares = s_tuple
            shares_to_buy = new_shares - self.shares[i]
            money_outflow = shares_to_buy * self.get_asset_price(i) * (1 + self.transaction_cost)
            self.shares[i] += shares_to_buy
            self.cash -= money_outflow

    def _update_portfolio_value(self):
        # compute current portfolio value
        self.portfolio_value = 0
        for stock_index in range(self.n_assets):
            self.portfolio_value += self.shares[stock_index] * self.get_asset_price(stock_index)

        self.portfolio_value += self.cash

    def _update_current_allocation(self):
        # compute current allocation based on new prices
        for stock_index in range(self.n_assets):
            s_allocation = (self.shares[stock_index] * self.get_asset_price(stock_index))/self.portfolio_value
            self.assets_allocation[stock_index] = s_allocation

    def step(self, action):
        new_allocation = self._action_to_vec(action)
        prev_portfolio_value = self.portfolio_value

        new_number_of_shares = []
        for i, allocation in enumerate(new_allocation):
            n_shares = int((allocation * self.portfolio_value)/(self.get_asset_price(i)*(1 + self.transaction_cost)))
            new_number_of_shares.append(n_shares)

        stocks_to_sell = []
        stocks_to_buy = []

        for i, new_shares in enumerate(new_number_of_shares):
            if self.shares[i] > new_shares:
                stocks_to_sell.append((i, new_shares))
            elif self.shares[i] < new_shares:
                stocks_to_buy.append((i, new_shares))
        
        self._sell(stocks_to_sell)
        self._buy(stocks_to_buy)
        self._update_portfolio_value()
        self._update_current_allocation()

        self.episode_step += 1

        terminated = False
        truncated = False
        info = {}
        if self.episode_step == self.episode_length - 1:
            terminated = True

        self._update_portfolio_value()
        self._update_current_allocation()
        log_return = math.log(self.portfolio_value/prev_portfolio_value)
        self.daily_pnl = log_return
        self.log_returns.append(log_return)
        self.pnl += self.portfolio_value - prev_portfolio_value
        self.daily_pnl_history.append(self.daily_pnl)

        observation = self._get_observation()
        # reward = self._get_reward(terminated)
        # reward = self._get_reward_sharpe()
        reward = self._get_morl_reward(terminated)

        return observation, reward, terminated, truncated, info

        

