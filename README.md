# My Submission to the Spatio-Temporal Beam-Level Traffic Forecasting Challenge 📶

[Competition Link](https://zindi.africa/competitions/spatio-temporal-beam-level-traffic-forecasting-challenge)

## Overview

Welcome to my repository for the Spatio-Temporal Beam-Level Traffic Forecasting Challenge on Zindi. I didn't win the competition, but I learned a lot working with complex datasets and building predictive models under a tight deadline.

## My Approach

I started by experimenting with XGBoost and ended up creating a complex model with over 100 features. However, after some testing, I realized that a simpler model actually performed better on my validation set.

So, for my final submission, I went with an ensemble of a linear model and quintiles of historical data. Since the competition involved predicting data far into the future, I thought a simpler model might generalize better over such a long period.

## Results

My model didn't make it to the top, achieving 0.256 MSE vs the winning 0.226, but the experience was really valuable! I'm excited to see how the winning teams approached the problem—many seemed to use gradient boosting ([here's a discussion about it](https://zindi.africa/competitions/spatio-temporal-beam-level-traffic-forecasting-challenge/discussions/22909)).

## Note

The code in this repo isn't the cleanest. I was working quickly to meet the deadline, so it doesn't strictly follow clean coding practices. 
