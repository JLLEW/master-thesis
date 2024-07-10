import gymnasium as gym
import numpy as np
import random
import math

class SimpleTradingEnv(gym.Env):
    def __init__(self, asset_list, asset_data, episode_length, seed=None, cash=1000000, eval=False, log=False):
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
        
        self.eval = eval
        # spaces
        self.action_space = gym.spaces.Discrete(2 ** (self.n_assets))
        self.observation_space = self._set_observation_space()
        self.reward_range = (-1*episode_length, episode_length + 100)
        self.offset = 31
        self.episode_step = 0
        self.data_i = self.offset

        self._set_env_variables()

    def _set_env_variables(self):
        if self.eval:
            self.data_i += self.episode_step
        else:
            self.data_i = random.randrange(self.offset, self.data_length - self.episode_length - 1)
            print(self.data_i)
        self.initial_value = self.init_cash
        self.portfolio_value = self.init_cash
        self.cash = self.init_cash
        self.assets_allocation = [0]*self.n_assets
        self.shares = [0]*self.n_assets
        self.episode_step = 0
        self.low = [np.inf]*self.n_assets
        self.high = [0]*self.n_assets
        self.pnl = 0
        self.transaction_costs = 0
        self.daily_pnl = 0
        self.daily_pnl_history = []
        # for differential sharpe ratio
        self.log_returns = []
        self.pct_returns = []
        self.A = 0
        self.B = 0
    
    def reset(self, seed=None, **kwargs):
        if 'cash' in kwargs.keys():
            self.init_cash = kwargs['cash']
        self.seed = seed
        self._set_env_variables()
        obs = self._get_observation()
        info = {}
        return obs, info

    
    def _set_observation_space(self):
        # presence flag in current portfolio
        dim = self.n_assets
        # price lookback 30 days
        dim += self.n_assets * 30

        observation_space = gym.spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        return observation_space
    
    def _return_trend(self, data):
        if sorted(data) == data:
            return 1
        if sorted(data, reverse=True) == data:
            return 2
        
        return 0
    
    def _market_return(self):
        r = 0
        for i in range(self.n_assets):
            r += math.log(self.get_asset_price(i)/self.get_asset_price(i, -1))
        r /= self.n_assets

        # returns average log return
        return r
    
    def _get_observation(self):
        lookback_window = 31
        obs = []
        for i in range(self.n_assets):
            obs += [int(self.assets_allocation[i] > 0)]
            index = self.data_i + self.episode_step
            asset_data = self.data[self.asset_names[i]].iloc[index-lookback_window:index,0].tolist()
            it = iter(asset_data)
            next(it)
            log_returns = [math.log(next(it)/val) for val in asset_data[:-1]]
            obs += log_returns

        obs = np.asarray(obs)

        return obs
    
    def _get_differential_sharpe_ratio(self):
        time_decay = 0.001
        pct_return = self.pct_returns[-1]
        self.A = np.mean(self.pct_returns)
        self.B = np.mean(np.array(self.pct_returns)**2)
        delta_A = pct_return - self.A
        delta_B = pct_return**2 - self.B
        A_prev = self.A
        B_prev = self.B

        if (B_prev - A_prev**2) <= 0:
            return 0

        Dt = (B_prev*delta_A - 0.5*A_prev*delta_B)/(abs(B_prev - A_prev**2)**(3/2))

        return Dt
    
    def _get_reward_sharpe(self):
        return self._get_differential_sharpe_ratio()
    
    def _get_reward(self, terminated):
        # if making money for couple of days in a row, give additional reward
        if terminated:
            if self.pnl > 0:
                return (((self.portfolio_value/self.init_cash) - 1) * 100)
            else:
                return -10
        
        market_return = self._market_return()
        
        if self.daily_pnl == 0:
            return 0
        elif market_return == 0:
            return self.daily_pnl

        if self.daily_pnl < 0:

            return -1*self.daily_pnl/market_return
        
        return self.daily_pnl/market_return
    
    def _get_reward_profit(self, terminated):
        # if making money for couple of days in a row, give additional reward
        if terminated:
            if self.pnl > 0:
                return (((self.portfolio_value/self.init_cash) - 1) * 100) ** 2
            else:
                return -10

        return self.daily_pnl*100

    
    def get_asset_price(self, asset_id, index=None):
        asset = self.asset_names[asset_id]
        df_i = self.data_i + self.episode_step
        if index == -1:
            df_i -= 1
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
            self.transaction_costs = shares_to_sell * self.get_asset_price(i) * self.transaction_cost
            self.shares[i] -= shares_to_sell
            self.cash += money_inflow
        
    def _buy(self, stocks):
        for s_tuple in stocks:
            i, new_shares = s_tuple
            shares_to_buy = new_shares - self.shares[i]
            money_outflow = shares_to_buy * self.get_asset_price(i) * (1 + self.transaction_cost)
            self.transaction_costs += shares_to_buy * self.get_asset_price(i) * self.transaction_cost
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
        self.transaction_costs = 0
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
        self.pct_returns.append(self.portfolio_value/prev_portfolio_value - 1)
        self.pnl += self.portfolio_value - prev_portfolio_value
        self.daily_pnl_history.append(self.daily_pnl)
        self.daily_return = self.portfolio_value - prev_portfolio_value
        observation = self._get_observation()
        reward = self._get_reward_sharpe()

        return observation, reward, terminated, truncated, info

        

