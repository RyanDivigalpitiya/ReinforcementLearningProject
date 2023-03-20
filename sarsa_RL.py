import numpy as np
import MDP

class sarsa_RL:
    def __init__(self,mdp,sampleReward):
        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]


    def sarsa(self,s0, initialQ, nEpisodes, nSteps, epsilon=0, temperature=0):
        avg_cum_rewards = np.zeros(nEpisodes)
        trials = 100

        for trial in range(trials):
            cumRewards = np.zeros(nEpisodes)
            n = np.zeros([self.mdp.nActions, self.mdp.nStates])
            Q = np.copy(initialQ)

            for episode in range(nEpisodes):
                state = s0
                action = self.choose_action(Q, state, self.mdp.nActions, epsilon, temperature)
                for step in range(nSteps):

                    reward, next_state = self.sampleRewardAndNextState(state, action)
                    next_action = self.choose_action(Q, next_state, self.mdp.nActions, epsilon, temperature)

                    n[action, state] += 1
                    alpha = 1.0 / n[action, state]

                    Q[action, state] = Q[action, state] + alpha * (reward + self.mdp.discount * Q[next_action, next_state] - Q[action, state])

                    state = next_state
                    action = next_action

                    cumRewards[episode] += pow(self.mdp.discount, step) * reward

            avg_cum_rewards += cumRewards

        avg_cum_rewards = avg_cum_rewards / trials
        policy = np.argmax(Q, axis=0)

        return [Q, policy, avg_cum_rewards]

    def choose_action(self,Q, state, nActions, epsilon, temperature):
        #epsilon-greedy exploration method is used for choice of action

        if epsilon > 0 and np.random.rand(1) < epsilon:
            numberOfActions = self.mdp.nActions
            action = np.random.randint(numberOfActions) #randomly explore any of these actions

        elif epsilon == 0 and temperature > 0:
            probability = np.exp(Q[:, state] / temperature)
            probability = probability / np.sum(probability)
            cumProbability = np.cumsum(probability)
            action = np.where(cumProbability >= np.random.rand(1))[0][0]

        else:
            action = np.argmax(Q[:, state])

        return action



###















###EndOfFile
