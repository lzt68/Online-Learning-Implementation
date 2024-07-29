# README

In this folder, we aim to reproduce the figure 1 and 2 in the paper "The True Sample Complexity of Identifying Good Arms", https://proceedings.mlr.press/v108/katz-samuels20a.html.

## Implementation Details

I firstly implemented the algorithm 1 mentioned in section 3. The paper delivers more details of the numeric experiments in the appendix G.

Following the description in the appendix G.3, there are 3 major differences between the true implementation and algorithm 1.

1. The size of the starting bracket is 64 instead of 2.
2. The samples of different brackets are shared, instead of being treated independently. To be specific, the pulling times of an arm $a$ is the summation of pulling times in all the brackets containing $a$. There isn't the concept of $\hat{\mu}_{i, r, T_{i,r}(t)}$, but $\hat{\mu}_{i, T_i(t)}$
3. The algorithm will stop generating new brackets once the new bracket size exceed the total arm number.





## File Structure





We aim to reproduce the appendix G of the paper https://proceedings.mlr.press/v108/katz-samuels20a.html, which utilizes the dataset https://github.com/nextml/caption-contest-data.



