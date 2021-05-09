# Databricks notebook source
# DBTITLE 1,Imports
import numpy as np
from copy import deepcopy as clone
import pandas as pd
import itertools
import random

# Interface model (ProbabilityModel) - each model that extends needs to have: predict_proba and partial_fit methods
class ProbabilityModel():
    def __init__(self):
        self.class_prior = [0.5, 0.5]  ### class_prior represents behavior without any data - [T, F] for True, False representation, should always sum to 1 when initiatlizaing
        # TODO: can be = null (not used)

    def predict_proba(X):  # Sample
        raise NotImplementedError("not implemented")

    def partial_fit(X, y):  # Update
        raise NotImplementedError("not implemented")


# BetaBernoulli - extends probability model, implements beta bernoulli
class BetaBernoulli(ProbabilityModel):  ### This class inheretes from ProbabilityModel
    """ This class is a representation of the BetaBernoulli distribution.
    This is one bandit."""
    __slots__ = ["T", "F", "prior_T", "prior_F"]

    def __init__(self, T, F):
        self.prior_T = T  # True cases - What we assume the distribution is when there is no data
        self.prior_F = F  # False cases - What we assume the distribution is when there is no data
        self.T = T
        self.F = F

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        # In case of Win / Lose, let's update our prior belief accordingly.
        y = np.array(y)
        self.T += (y == 1).sum()
        self.F += (y == 0).sum()
        return self

    def discount(self, r):
        # Let's update our True / False accordingly with r rate, while multiplying with a number
        self.T = (self.T + r * self.prior_T) / (1 + r)
        self.F = (self.F + r * self.prior_F) / (1 + r)
        return self

    def predict_proba(self, X):
        # sample out of the beta distribution. This is the posterior
        ret = []
        for _ in range(len(X)):
            p = np.random.beta(self.T, self.F)
            ret.append([1 - p, p])
        return np.array(ret)



class BiddingStrategy():
    ### This class represents a collection of bandits
    def __init__(self, n_bins, max_bid, priors, classifier, desired_win_rate):
        assert n_bins == len(priors)  # Just for sanity

        bins = np.linspace(0, max_bid, n_bins + 1)[1:]  # number of arms
        self.bid_model = {}  # dict of price: probability of it
        for b, p in zip(bins, priors):
            self.bid_model[b] = clone(classifier)

        self.desired_win_rate = desired_win_rate

    def discount(self, r):
        """per each of the arms that based on this, run discount mechanism
        The goal of discount is to lower the confidence of each model, such that it will be based on "more recent" data, thus it can learn."""
        for model in self.bid_model.values():
            model.discount(r)
        return self

    def parameters(self):
        """BetaBernoulli based - parameters of the model: alpha + beta"""
        """TODO: Generalize for general models"""
        return {price: model.T / (model.T + model.F) for price, model in self.bid_model.items()}

    def bid(self, context):
        """ This method will generate a bid based on a bidding strategy"""
        probabilities = [(model.predict_proba([context])[0][1]) for model in self.bid_model.values()]
        self.probabilities = probabilities
        probabilities = self.normalize_probs(probabilities)  # So we will have the confidence per each arm

        assert len(probabilities) == len(self.bid_model.keys())

        n_samples = 3  # Aggregation strategy
        bid_price = np.random.choice(a=list(self.bid_model.keys()), p=probabilities, size=n_samples).mean()
        # bid price is based on 3 sampling average from the probabilities model

        return bid_price

    def reward(self, bid_price, won, context):
        """Represents the reward that will be given per each one of the bandits, negative / positive
        # params:
        # bid_price : bid price that our "agent" got out with
        # context: mapping: 1 - X between our world: e.g. dnt=true, os=android, ... then 1, dnt=false, os =android then 2, etc - this is the weak thing in beta bernoulli, as we need to create this mapping based on our logic - e.g. only for placementType, placementType + dnt, ...
        # won - True / False for win / lose in auction"""

        reward_arr = []
        for price in self.bid_model.keys():
            r = self.specific_reward(price, bid_price, won)
            if r != 0:
                for _ in range(abs(r)):
                    self.bid_model[price].partial_fit([context], [r > 0])
            reward_arr.append(r)
        return reward_arr

    def specific_reward(self, bandit_price, bid_price, won):
        if won and bandit_price >= bid_price:
            return 1
        if (not won) and bandit_price <= bid_price:
            return -1
        return 0

    def normalize_probs(self, arr):
        """This function calculates log on arr, then multiplies by 2 and deducts the min of that product
        Calculated as: 2*np.log(arr) - np.min(2*np.log(arr))
        On top of that, exponent and that is the returned value
        The goal is to have the highest value a very high value, and the lowest - 1"""
        arr = np.array(arr)
        arr = np.abs(np.log(arr) - np.log(self.desired_win_rate)) # 1L1D
        arr = arr - np.min(arr)
        # The goal: The closer you are as a bandit to the win rate, meaning this bandit is more likely to be chosen

        # log_arr = np.log(arr)
        # normalized_log_arr = 2 * log_arr - np.min(2 * log_arr)
        exp_norm_arr = np.exp(arr)
        exp_norm_arr /= exp_norm_arr.sum()

        return exp_norm_arr

    def simulate_by_constant(self, constant, discount_perc, n_iteration=100, noise=2):
        """Going to simulate"""
        regret_arr = []

        for i in range(1, n_iteration):
            b = self.bid([i])  # provide a bid
            auction_bid = constant + (random.random() - 0.5) *2* noise
            win = b > auction_bid

            regret_arr.append(np.abs(b - auction_bid)) # log regret: ABS(bid - constant, which is the "win price by default")
            self.discount(discount_perc) # discount
            r = self.reward(bid_price=b, won=win, context=[1])  # Generate reward per auction fin.


        return np.mean(np.array(regret_arr))

    # TODO: We want a negative reward


#       def specific_reward_function(self, bandit_price, bid_price, won):
#         """This method will update per each bandit its reward based on similiarity logic (how close were they to win / lose)"""
#           # TODO: This should be decided
#         diff = bid_price - bandit_price
#         if won:
#           if diff == 0: # we won in that price!
#             return 70
#           elif diff > 0 and diff <= 1:
#             return 30
#           elif diff > 0 and diff <= 2:
#             return 10
#           elif diff < -3:
#             return -3
#           else:
#             return 0

#         else: # lose
#             if diff > 5:
#               return -1
#             elif diff < 0 and diff >= -2:
#               return 1
#             else:
#               return 0

# COMMAND ----------

cls = BetaBernoulli(1, 1000)  # Defining an instance of this BetaBernoulli with 1 win, 1000 loses as our prior

# COMMAND ----------

# COMMAND ----------

reward_dict = {}
max_data_point = 10  # max number of bins
best_regret = 100
best_params = {}
penalties = [-1000, -1, 0, 10, 1000]  # TODO: try to think on functions / many distinct values -

# TODO:
# (1) joblib - multiprocessing https://joblib.readthedocs.io/en/latest/parallel.html for the below
# (2):
from joblib import delayed, Parallel


@delayed
def run_iteration():
    for comb in itertools.product(range(1, max_data_point + 1), [-2, -1, 1, 2], [True, False]):
        reward_dict[comb] = random.choice(penalties)
    biddingStrategy = BiddingStrategy(n_bins=max_data_point, max_bid=max_data_point, priors=range(max_data_point),
                                      classifier=cls, desired_win_rate=0.6)
    regret = biddingStrategy.simulate_by_constant(4, 0.99)
    return regret, reward_dict


from datetime import datetime

format = "%m/%d/%Y, %H:%M:%S"
print("Start:", datetime.now().strftime(format))
lst = Parallel(n_jobs=8)(run_iteration() for i in range(1000))
best_regret, best_params = min(lst)
print("Best Regret:", best_regret)
# print("best params:", best_params)
reward_arr = []
for k, v in best_params.items():
    reward_arr.append((k[0], k[1], k[2], v))
pandas_best_params = pd.DataFrame(reward_arr, columns=["bandit_price", "diff", "won", "reward"]).astype(int)
pandas_best_params.to_csv("output.csv", index=None)
print("End:", datetime.now().strftime(format))

# COMMAND ----------

# best_regret = 1.4006734006734005 for diff binary
# best_regret = 1.1 with 4 directions diff (-2, -1, 1, 2)
# best_regret = 0.919 with diff (4 directions) / customized reward [-10,-5,-2,-1,0,1,2,5,10]
# best_regret = 0.39 with diff (4 directions) / customized super reward penalties = [-1000,-100,-50,-20,-1,0,1,20,50,100,1000] - didn't manage to reproduce that specific result, mostly due to randomness
best_regret

# COMMAND ----------

best_params
# (bid_price, diff, won)
# bid price was 1, bid_price > bandit_price, lost => up

# diff
# if 0 < bid_price - bandit_price <= 2:
#         diff = 1
#       elif -2 < bid_price - bandit_price <= 0:
#         diff = -1
#       elif bid_price - bandit_price > 2:
#         diff = 2
#       else:
#         diff = -2

# COMMAND ----------

# def specific_reward_by_dict(self, bid_price, bandit_price, won):
#       """This method generate a reward based on bid price, bandit price, won events, stored in a reward_dictionary outside"""
#       diff = (bid_price - bandit_price) > 0 # Boolean flag
#       return int(self.reward_dict.get(bid_price, diff, won), 0)


# COMMAND ----------


# dummy_X = np.random.randint(low=1, high=max_data_point, size=(10,1)) # Dummy data - represents the bids that were in many auctions
# print(dummy_X) # e.g. 3 = banner, dntTrue, iOS

# dummy_bids = np.random.randint(low=1, high=max_data_point, size=(10,)) #
# print(dummy_bids)

# dummy_bool_auction_results = np.random.rand((10)) > 0.5
# print(dummy_bool_auction_results)


# cls = SGDClassifier(loss='log')
# cls.partial_fit(dummy_X, )
# cls.fit(dummy_X, np.random.rand(dummy_X.shape[0]) > 0.5)
# cls.partial_fit()


# reward_arr = []
# bids_arr = []
# win_lose_arr = []
# probs_arr = []
# params_arr = []
# for i in range(1, 1000):
#   b = biddingStrategy.bid([i])
#   params_arr.append(list(biddingStrategy.parameters().values()))
#   print(f"My bid is: {b:0.3f}")
#
#   bids_arr.append(b)
#   win = b > 7
#   win_lose_arr.append(win)
#
#   biddingStrategy.discount(0.9)
#
#   r = biddingStrategy.reward(bid_price=b, won=win, context=[1])
#   probs_arr.append(biddingStrategy.probabilities)
#   reward_arr.append(r)
#
# # COMMAND ----------
#
# for i in range(0, 1000, 10):
#   plt.bar(biddingStrategy.bid_model.keys(),params_arr[i])
#   plt.show()
#
# # COMMAND ----------
#
# pd.DataFrame(bids_arr).rolling(10).mean().plot()
# # pd.DataFrame(reward_arr).astype(int).plot()
# pd.DataFrame(probs_arr).plot()
# plt.ylim(-1e-3, 1e-3)

# COMMAND ----------

# print(np.array(probs_arr).max())
# print(np.array(probs_arr).min())

# COMMAND ----------

# probs_arr

# COMMAND ----------

# probabilities = np.array([0.000456890760588994,
#   0.006465069776677389,
#   0.0011343878313113415,
#   0.0055366061701542955,
#   0.003659762936292841,
#   0.0025525817626757936,
#   0.003517560887148804,
#   0.0004356916625517895,
#   0.0009289473262418103,
#   0.0005443335700367749,
#   0.0011412011710087097,
#   0.0011297464941493651])

# COMMAND ----------

# prob_hund = np.array(probabilities)*100

# COMMAND ----------

# probabilities = customized_softmax(np.array(prob_hund))
# print(probabilities)

# COMMAND ----------

# n_samples = 1000
# print(list(zip(biddingStrategy.bid_model.keys(), probabilities/np.sum(probabilities))))
# print(np.random.choice(a=list(biddingStrategy.bid_model.keys()), p=probabilities/np.sum(probabilities), size=n_samples))
