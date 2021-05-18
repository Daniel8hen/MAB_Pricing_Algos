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

    def parameters(self, return_type=None):
        """BetaBernoulli based - parameters of the model: alpha + beta"""
        if return_type=="T":
            return {price: model.T  for price, model in self.bid_model.items()}
        if return_type=="F":
            return {price: model.F  for price, model in self.bid_model.items()}
        return {price: model.T / (model.T + model.F) for price, model in self.bid_model.items()}

    def bid(self, context):
        """ This method will generate a bid based on a bidding strategy"""
        probabilities = [(model.predict_proba([context])[0][1]) for model in self.bid_model.values()]
        self.probabilities = probabilities
        probabilities = self.normalize_probs(probabilities)  # So we will have the confidence per each arm

        assert len(probabilities) == len(self.bid_model.keys())

        # Aggregation strategy
        n_samples = 11
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
            if r > 0:
                self.bid_model[price].partial_fit([context], [True])
            if r < 0:
                self.bid_model[price].partial_fit([context], [False])
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
        # print("arr is:", arr)
        # arr = np.abs(np.log(arr) - np.log(self.desired_win_rate)) # 1L1D
        # overwrite for test
        # arr = 1/(np.round(np.abs(arr - self.desired_win_rate),1) + np.float(0.01)) # numerical stability
        arr = 1 / (np.abs(arr - self.desired_win_rate) + np.float(0.00001))  # numerical stability
        # print("arr after manipulation and distance:", arr)
        # print("arr for sum is:", arr/arr.sum())
        return arr/arr.sum()

        arr = arr - np.min(arr)
        # The goal: The closer you are as a bandit to the win rate, meaning this bandit is more likely to be chosen
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


    def simulate_by_real_dataset(self, df, discount_perc):
        """Going to simulate by real df"""
        regret_arr = []

        for i, row in df.iterrows():
            b = self.bid([i])  # provide a bid
            # b = 0.4
            auction_bid = row['bidPrice']
            win = b > auction_bid

            regret_arr.append(np.abs(b - auction_bid))  # log regret: ABS(bid - constant, which is the "win price by default")
            self.discount(discount_perc)  # discount
            r = self.reward(bid_price=b, won=win, context=[1])  # Generate reward per auction fin.
        return np.mean(np.array(regret_arr))

    def learn_priors(self, df, prior_weight=1):
        """ This method will calculate priors per each bin in self"""
        last_T = 1
        last_F = 1
        min_bin = list(self.bid_model.keys())[0]
        for bin in reversed(list(self.bid_model.keys())):
            l = min(bin-min_bin, 0.9*bin)
            u = max(bin + min_bin, 1.1 * bin)
            ds = df[(df["bidPrice"]>=l) & (df["bidPrice"]<=u)]["hasWon"].value_counts(normalize=True).to_dict()
            T = ds.get(True, last_T)
            F = ds.get(False, last_F)
            self.bid_model[bin].prior_T = T*prior_weight
            self.bid_model[bin].prior_F = F*prior_weight
            last_T, last_F = T, F


# Main
from joblib import delayed, Parallel

if __name__ == '__main__':
    import pandas as pd
    from datetime import datetime

    cls = BetaBernoulli(1, 1)  # Defining an instance of this BetaBernoulli with 1 win, 1000 loses as our prior

    df = pd.read_csv("../data/data.csv")
    df = df.query("placementType == 'banner'")
    max_data_point = 1  # max number of bins
    n_bins=10

    format = "%m/%d/%Y, %H:%M:%S"
    print("Start:", datetime.now().strftime(format))
    biddingStrategy = BiddingStrategy(n_bins=n_bins, max_bid=max_data_point, priors=[1]*n_bins,
                                      classifier=cls, desired_win_rate=0.6)
    # regret = biddingStrategy.simulate_by_constant(4, 0.99)
    regret = biddingStrategy.simulate_by_real_dataset(df.head(2000), 0.99)
    print("Regret:", regret)

    ### General TODOs:
    # 1. Plotting each aspect of pricing here so we can see how the model is learning
    # 2. No context - making it work by heavily debugging
    # 3. With context? Adding it may help