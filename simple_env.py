import gymnasium as gym
import numpy as np
import random
import math

class SimpleTradingEnv(gym.Env):
    def __init__(self, asset_list, asset_data, episode_length, cash=100000, eval=False, log=False):
        self.log = log
        self.asset_names = asset_list
        self.n_assets = len(asset_list)
        self.data = asset_data
        self.data_length = len(self.data[asset_list[0]])
        self.episode_length = episode_length
        self.cash = cash
        self.init_cash = cash
        self.transaction_cost = 0.002

        # spaces
        self.action_space = gym.spaces.Discrete(2 ** (self.n_assets))
        self.observation_space = self._set_observation_space()
        self.reward_range = (-1*episode_length, episode_length + 100)

        self._set_env_variables()

    def _set_env_variables(self):
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
    
    def _get_reward(self, terminated):
        # if making money for couple of days in a row, give additional reward
        if terminated:
            if self.pnl > 0:
                return (((self.portfolio_value/self.init_cash) - 1) * 100) ** 2
            else:
                return -10

        return self.daily_pnl*100
        # pnl_history_5_d = self.daily_pnl_history[-5:]
        # if sum(pnl_history_5_d) > 0:
        #     return 1
 
        # elif sum(pnl_history_5_d) == 0:
        # #     return 0
        # # else:
        # #     return -5
    
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
        self.daily_pnl = math.log(self.portfolio_value/prev_portfolio_value)
        self.pnl += self.portfolio_value - prev_portfolio_value
        self.daily_pnl_history.append(self.daily_pnl)

        observation = self._get_observation()
        reward = self._get_reward(terminated)

        return observation, reward, terminated, truncated, info

        

