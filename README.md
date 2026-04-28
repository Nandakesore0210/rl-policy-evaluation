# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
To implement policy evaluation on the FrozenLake-v1 environment and compare two given policies by calculating their probability of reaching the goal, average return, and state-value function, in order to identify the better policy.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by policy
            
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        
        if delta < theta:
            break
    
    return V
```

## PROGRAM:
```
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))

# Create your own policy

pi_2 = lambda s: {
    0: RIGHT,
    1: RIGHT,
    2: DOWN,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: DOWN,
    7:LEFT,
    8: RIGHT,
    9: RIGHT,
    10:DOWN,
    11:DOWN,
    12:RIGHT,
    13:UP,
    14:RIGHT,
    15:LEFT #Stop
}[s]

print("Name: Nandakesore J")
print("Register Number: 212223240103")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

# Find the probability of success and the mean return of you your policy

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))
# Compare your policy with the first policy


# Evaluate first policy
success_frozenlake = probability_success(env, pi_frozenlake, goal_state=goal_state) * 100
avg_return_frozenlake = mean_return(env, pi_frozenlake)

# Evaluate your policy
success_pi2 = probability_success(env, pi_2, goal_state=goal_state) * 100
avg_return_pi2 = mean_return(env, pi_2)

print("----- Policy Comparison -----")
print("First Policy:")
print("  Success Rate: {:.2f}%".format(success_frozenlake))
print("  Mean Return : {:.4f}".format(avg_return_frozenlake))

print("\nYour Policy:")
print("  Success Rate: {:.2f}%".format(success_pi2))
print("  Mean Return : {:.4f}".format(avg_return_pi2))
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by policy
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)
# Code to evaluate the second policy
# Write your code here
V2 = policy_evaluation(pi_2, P,gamma=0.99)
print_state_value_function(V2, P, n_cols=4, prec=5)

V1>=V2

s1=np.sum(V1)

s2=np.sum(V2)

if(s1>=s2):
  print("The first policy is the better policy")
elif(s2>=s1):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
### pi_frozenlake:
<img width="753" height="533" alt="image" src="https://github.com/user-attachments/assets/09489657-4d88-4680-b337-9a5b3f8ed900" />

### Reaches goal & average undiscounted return:
<img width="842" height="106" alt="image" src="https://github.com/user-attachments/assets/f388917f-932c-4f14-8163-13990a0cd47e" />

### pi_2:
<img width="600" height="633" alt="image" src="https://github.com/user-attachments/assets/414ca420-9c05-401c-bc30-f27a990d9796" />

### Reaches goal & average undiscounted return:
<img width="761" height="138" alt="image" src="https://github.com/user-attachments/assets/e0ab7bed-bbf8-4389-8102-e583f68acc2f" />

### Policy Comparison:
<img width="267" height="164" alt="image" src="https://github.com/user-attachments/assets/de7cb7f7-48ee-4702-9648-45f8a2778285" />

### Policy Evaluation:
<img width="465" height="179" alt="image" src="https://github.com/user-attachments/assets/adf2ff1b-a2e5-45b3-a09c-860e112181cc" />
<img width="468" height="196" alt="image" src="https://github.com/user-attachments/assets/e7b82bbe-6eb6-44c6-9eda-599c8100869e" /><br>
<img width="618" height="369" alt="image" src="https://github.com/user-attachments/assets/da366f2d-c499-41a0-8d7e-2fe2b9523e00" />



## RESULT:
Therefore, policies are compared successfully using policy evaluation function in Frozen-Lake MDP.
