import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


class Arm:
    def __init__(self, mean: float, var: float, *, drift=False) -> None:
        self.var = var
        self.mean = np.random.normal(mean, self.var)
        self.drift = drift
        self.drift_report = [self.mean]

    def pull(self):
        if self.drift:
            self.mean += np.random.uniform(-self.var / 25, self.var / 25)
            self.drift_report.append(self.mean)
            
        return np.random.normal(self.mean, self.var)


class Bandit:
    def __init__(self, k: int, mean: float, var: float, *, drift=False) -> None:
        self.k = k
        self.drift = drift
        self.arms = [Arm(mean, var, drift=drift) for _ in range(k)]

    def pull(self, arm: int) -> float:
        return self.arms[arm].pull()

    def report(self) -> list[float]:
        return [arm.mean for arm in self.arms]


class Agent:
    def __init__(self, arms: int, n_games: int, *, OIV=False) -> None:
        self.arms = arms
        self.bandit = Bandit(arms, 5, 2.5)
        self.n_games = n_games
        self.alpha = 0.1
        self.q_star = np.random.normal(0, 5, self.arms) #Actual mean reward

        self.Q_greedy = np.zeros(self.arms)             #Calculated expected rewards for greedy algo.
        self.Q_epsilon_greedy = np.zeros(self.arms)     #Calculated expected rewards for epsilon greedy algo.
        self.Q_ucb = np.zeros(self.arms)                #Calculated expected rewards for UCB algo.
        self.Q_gradient = np.zeros(self.arms)           #Calculated expected rewards for Gradient algo.

        if OIV:
            self.Q_greedy += 10
            self.Q_epsilon_greedy += 10
            self.Q_ucb += 10
            self.Q_gradient += 10

        self.rewards = np.array([np.random.normal(self.q_star[i], 1, self.n_games) for i in range(self.arms)])

    def _get_reward(self, arm, num):
        return self.rewards[arm][num]
    
    def play_game(self):    
        t1 = Process(self._play_greedy())
        t2 = Process(self._play_epsilon_greedy())
        t3 = Process(self._play_ucb())
        t4 = Process(self._play_gradient_bandit())
        
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        
        print("Game Completed!!")

    def _select_arm_greedy(self):
        return np.argmax(self.Q_greedy, 0)

    def _play_greedy(self):
        actions = []
        rewards = []

        for i in range(self.n_games):
            a = self._select_arm_greedy()
            actions.append(a+1)
            reward = self._get_reward(a, i)
            rewards.append(reward)
            self.Q_greedy[a] = self.Q_greedy[a] + self.alpha * (reward - self.Q_greedy[a])

        self.greedy_results = (actions, np.array(rewards), np.sum(rewards)) 
    
    def _select_arm_epsilon_greedy(self):
        if np.random.binomial(1,0.9,1):
            return np.argmax(self.Q_epsilon_greedy, 0)
        return np.random.choice(self.arms, 1)[0]

    def _play_epsilon_greedy(self):        
        actions = []
        rewards = []

        for i in range(self.n_games):
            a = self._select_arm_epsilon_greedy()
            actions.append(a+1)
            reward = self._get_reward(a, i)
            rewards.append(reward)
            self.Q_epsilon_greedy[a] = self.Q_epsilon_greedy[a] + self.alpha * (reward - self.Q_epsilon_greedy[a])

        self.epsilon_greedy_results = (actions, np.array(rewards), np.sum(rewards))
    
    def _select_arm_ucb(self, t):
            N = [1] * self.arms
            l_max  = -1
            arm = -1

            for i in range(self.arms):
                curr_val = self.Q_ucb[i] + 2 * np.sqrt(np.log(t+1)/N[i])
                if l_max < curr_val:
                    arm = i
                    l_max = curr_val
            N[arm] = N[arm]+1

            return arm

    def _play_ucb(self):      
        actions = []
        rewards = []

        for i in range(self.n_games):
            a = self._select_arm_ucb(i)
            actions.append(a+1)
            reward = self._get_reward(a, i)
            rewards.append(reward)
            self.Q_ucb[a] = self.Q_ucb[a] + self.alpha * (reward - self.Q_ucb[a])
            
        self.ucb_results = (actions, np.array(rewards), np.sum(rewards))

    def _select_arm_gradient_bandit(self, P):
        return np.argmax(P, 0)


    def _play_gradient_bandit(self):
        P = [1 / self.arms] * self.arms

        actions = []
        rewards = []

        for i in range(self.n_games):
            a = self._select_arm_gradient_bandit(P)
            actions.append(a+1)
            reward = self._get_reward(a, i)
            rewards.append(reward)

            for j in range(self.arms):
                if i==a:
                    self.Q_gradient[a] = self.Q_gradient[a] + self.alpha * (reward - self.Q_gradient[a]) * (1-P[j])
                else:
                    self.Q_gradient[j] = self.Q_gradient[j] - self.alpha * (reward - self.Q_gradient[j]) * P[j]
            
            softmax = np.array(list(map(np.exp, self.Q_gradient)))
            P = softmax / np.sum(softmax)   

        self.gradient_results = (actions, np.array(rewards), np.sum(rewards))


def analyze(agent: Agent):
    arms = agent.arms
    games = agent.n_games

    # Analyzing the Q values
    plt.plot(range(arms), agent.q_star, label = 'Actual Q Function', alpha = 0.5)
    plt.plot(range(arms), agent.Q_greedy, label = 'Calculated Greedy Q values', alpha = 0.5)
    plt.plot(range(arms), agent.Q_epsilon_greedy, label = 'Calculated epsilon Greedy Q values', alpha = 0.5)
    plt.plot(range(arms), agent.Q_ucb, label = 'Calculated UCB Q values', alpha = 0.5)
    plt.plot(range(arms), agent.Q_gradient, label = 'Calculated Gardient Q values', alpha = 0.5)
    plt.xlabel('Arm')
    plt.ylabel('Values')
    plt.title('Q values')
    plt.legend(bbox_to_anchor=(2, 1.05), fancybox=True)
    plt.show()
    
    def rmse(a):
        return np.square(a[0]-a[1])
    print("Difference of Actual and calculated values for Greedy values:", 
          np.sqrt(np.mean(list(map(rmse, zip(agent.q_star, agent.Q_greedy))))), agent.greedy_results[-1])
    print("Difference of Actual and calculated values for epsilon Greedy values:", 
          np.sqrt(np.mean(list(map(rmse, zip(agent.q_star, agent.Q_epsilon_greedy))))),agent.epsilon_greedy_results[-1])
    print("Difference of Actual and calculated values for UCB values:", 
          np.sqrt(np.mean(list(map(rmse, zip(agent.q_star, agent.Q_ucb))))), agent.ucb_results[-1])
    print("Difference of Actual and calculated values for Gradient values:", 
          np.sqrt(np.mean(list(map(rmse, zip(agent.q_star, agent.Q_gradient))))), agent.gradient_results[-1])

    # Analyzing the Cummulative Rewards values
    plt.plot(range(games), np.cumsum(agent.greedy_results[1]), label = 'Calculated Greedy Returns', alpha = 0.5)
    plt.plot(range(games), np.cumsum(agent.epsilon_greedy_results[1]), label = 'Calculated epsilon Greedy Returns', alpha = 0.5)
    plt.plot(range(games), np.cumsum(agent.ucb_results[1]), label = 'Calculated UCB Returns', alpha = 0.5)
    plt.plot(range(games), np.cumsum(agent.gradient_results[1]), label = 'Calculated gradient Returns', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Cummulative Returns')
    plt.title('Rewards')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()

    # Analyzing the Rewards values
    plt.plot(range(games), agent.greedy_results[1], label = 'Calculated Greedy Returns', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Returns')
    plt.title('Rewards')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), agent.epsilon_greedy_results[1], label = 'Calculated epsilon Greedy Returns', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Returns')
    plt.title('Rewards')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), agent.ucb_results[1], label = 'Calculated UCB Returns', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Returns')
    plt.title('Rewards')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), agent.gradient_results[1], label = 'Calculated Gradient Returns', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Returns')
    plt.title('Rewards')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()

    best_action=np.argmax(agent.q_star)+1
    print("Best Action:", best_action)
    
    # Analyzing the Best Action Selection
    plt.plot(range(games), np.cumsum(np.where(np.array(agent.greedy_results[0]) == best_action,1,0)), label = 'Best Greedy Action', alpha = 0.5)
    plt.plot(range(games), np.cumsum(np.where(np.array(agent.epsilon_greedy_results[0]) == best_action,1,0)), label = 'Best epsilon Greedy Action', alpha = 0.5)
    plt.plot(range(games), np.cumsum(np.where(np.array(agent.ucb_results[0]) == best_action,1,0)), label = 'Best UCB Action', alpha = 0.5)
    plt.plot(range(games), np.cumsum(np.where(np.array(agent.gradient_results[0]) == best_action,1,0)), label = 'Best Gradient Action', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Number of times Best Action selected')
    plt.title('Actions')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    
    # Analyzing the Best Action Selection
    plt.plot(range(games), np.where(np.array(agent.greedy_results[0]) == best_action,1,0), label = 'Best Greedy Action', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Number of times Best Action selected')
    plt.title('Actions')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), np.where(np.array(agent.epsilon_greedy_results[0]) == best_action,1,0), label = 'Best epsilon Greedy Action', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Number of times Best Action selected')
    plt.title('Actions')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), np.where(np.array(agent.ucb_results[0]) == best_action,1,0), label = 'Best UCB Action', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Number of times Best Action selected')
    plt.title('Actions')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()
    plt.plot(range(games), np.where(np.array(agent.gradient_results[0]) == best_action,1,0), label = 'Best Gradient Action', alpha = 0.5)
    plt.xlabel('Games')
    plt.ylabel('Number of times Best Action selected')
    plt.title('Actions')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True)
    plt.show()








if __name__ == "__main__":
    # Without Optimistic initial values
    arms = 10
    games = 1000000
    agent = Agent(arms, n_games=1000000)
    agent.play_game()
    analyze(agent)

    # With Optimistic initial values
    arms = 10
    games = 1000000
    agent = Agent(arms, n_games=1000000, OIV=True)
    agent.play_game()
    analyze(agent)