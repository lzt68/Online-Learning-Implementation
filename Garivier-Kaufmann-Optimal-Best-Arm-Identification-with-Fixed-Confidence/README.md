# README

This folder aims to implement the **Track-and-Stop** algorithms in *Garivier \& Kaufamm Optimal Best Arm Identification with Fixed Confidence* . Link of the paper:http://arxiv.org/abs/1602.04589. 

There are two versions of tracking strategy mentioned in the paper, which are C-Tracking and D-Tracking. Though the paper compared their proposed algorithms with others as numeric record, here I don't follow their ideas. Instead, I am going to test the changing trend of $\frac{\mathbb{E}_{\mu}[\tau_{\delta}]}{\log\frac{1}{\delta}}$, which is also the main theoretical result in the paper.

Here we assume the reward follows Bernoulli Distribution.