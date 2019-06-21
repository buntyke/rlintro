# RL Intro
Material for knowledge sharing session introducing reinforcement learning.

The session has three parts:
* Introduction to RL: Slides available [here](slides/introtorl.pdf). This presentation introduces the field of reinforcement learning along with the basic components.
* MDP Formulation: Slides available [here](slides/mdpbasics.pdf). This introduces the concept of Markov Decision Process and the Q-learning algorithm.
* Q-learning Code: Code introduction for q-learning algorithm applied to the cart pole environment.

# Usage Instructions
* Setup a python virtual environment of version >= 3.6
* Install the requirements for running the code:
  ```
  $ pip3 install -r requirements.txt
  ```
* Run the q-learning code with the default arguments:
  ```
  $ python qlearn.py -n 1500 -p 150
  ```
  This runs q-learning algorithm for 1500 episodes and plots the performance for every 150 episodes

# Acknowledgements

The slides are adapted from the excellent [lecture series](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial4) by Prof. David Silver. The q-learning code is adapted from this tutorial series [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html). 