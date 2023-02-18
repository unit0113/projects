#https://github.com/govindgnair23/RL_Exploration/blob/main/Solving%20Tic%20Tac%20Toe%20with%20RL%20Blog%20-%20vf.ipynb
import numpy as np
from itertools import product
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt


class TicTacToe():
    def __init__(self,player = 'X',reward_type ='goal_reward'):
        '''
        player: Role agent should play. If X, agent has the first turn else agent has second turn
        reward_type: 'goal_reward' or 'action_penalty'
        '''
        self.board = np.array(['__']*9).reshape(3,3)
        self.reward_type = reward_type
        self.winning_seqeunce = None #Keep track of winning move made by agent
        self.first_move = None #Keep track of first move made by agent
        self.game_over = False

        if player == 'X':
            self.me ='X'
            self.id = 1
            self.opponent = 'O'
        else:
            self.me = 'O'
            self.id = 2
            self.opponent = 'X'

        # Mapping of action representation in board to action representation in tuple 
        self.b_to_s = {'__':0,'X':1,'O':2} 
        # Mapping of action representation in tuple to action representation in board
        self.s_to_b = {0:'__',1:'X',2:'O'} 
        
        #Create mapping from 2D position in board to 1D position in tuple
        positions = self.available_positions()
        self.b2_to_s1 = {position:i for (i,position) in enumerate(positions)}
        
        #Create mapping from 1D position in tuple to 2D position in board 
        self.s1_to_b2 = {i:position for (i,position) in enumerate(positions)}
        
        #State the current player is in
        self.starting_state = self.board_to_state()
        
        #Initialize all possible states of the game
        l_o_l = [list(range(3)) for _ in range(9)]
        states = set(product(*l_o_l))
        
        #Player X states include states with odd number of blanks and both players have occupied equal number of slots
        #Player O players after Player X, so player O states include states with even number of blanks and where
        #player X has occupied one more slot than player O
        playerX_states = {state for state in states if (state.count(0)%2 == 1 and state.count(1)==state.count(2))} #
        playerO_states =  {state for state in states if (state.count(0)%2 == 0 and state.count(1)==(state.count(2)+1))}
        
        if player == 'X':
            self.my_states = playerX_states
        else:
            self.my_states = playerO_states
          
    def reset_board(self):
        "Function to reset game and reset board to starting state"
        self.board = np.array(['__']*9).reshape(3,3)
        self.starting_state = self.board_to_state()
        self.game_over = False
        self.winning_sequence = None
        self.first_move = None
    
    def show_board(self):    
        "Shows board as a pandas dataframe"
        return pd.DataFrame(self.board)
    
    def board_to_state(self):
        "Convert a board to a state in tuple format"
        return tuple([self.b_to_s[x] for x in np.ravel(self.board)])
    
    @staticmethod
    def possible_actions(state):
        "Return possible actions given a state"
        return [i for i,x  in enumerate(state) if x ==0]
    
    def is_game_over(self):
        "Function to check if game is over"
        if not np.any(self.board == '__') :
            self.game_over = True
            
        return self.game_over
    
    def available_positions(self):
        "Return available positions on the board"
        x,y = np.where(self.board =='__')
        return[(x,y) for x,y in zip(x,y)]
    
    def win(self,player):
        "Check if player won the game and record the winning sequence"
        if np.all(self.board[0,:] == player):
            self.winning_sequence = 'R1'
        elif np.all(self.board[1,:] == player): 
            self.winning_sequence = 'R2'
        elif np.all(self.board[2,:] == player):
            self.winning_sequence = 'R3'
        elif np.all(self.board[:,0] == player):
            self.winning_sequence = 'C1'
        elif np.all(self.board[:,1] == player):
            self.winning_sequence = 'C2'
        elif np.all(self.board[:,2] == player):
            self.winning_sequence = 'C3'
        elif np.all(self.board.diagonal()==player):
            self.winning_sequence = 'D1'
        elif  np.all(np.fliplr(self.board).diagonal()==player):
            self.winning_sequence = 'D2'
        else:
            return False
        
        return True
    
    def my_move(self,position):
        "Fills out the board in the given position with the action of the agent"
        
        assert position[0] >= 0 and position[0] <= 2 and position[1] >= 0 and position[1] <= 2 , "incorrect position"
        assert self.board[position] == "__" , "position already filled"
        assert np.any(self.board == '__') , "Board is complete"
        assert self.win(self.me) == False and self.win(self.opponent)== False , " Game has already been won"
        self.board[position] = self.me
        
        I_win = self.win(self.me)
        opponent_win = self.win(self.opponent)
        
        if self.reward_type == 'goal_reward':
            if I_win:
                self.game_over = True
                return 1
            
            elif opponent_win:
                self.game_over = True
                return -1
            
            else:
                return 0
            
        elif self.reward_type == 'action_penalty':
            if I_win:
                self.game_over = True
                return 0
            
            elif opponent_win:
                self.game_over = True
                return -10
            
            else:
                return -1
    
    def opponent_move(self,position):
        "Fills out the board in the given position with the action of the opponent"
        assert position[0] >= 0 and position[0] <= 2 and position[1] >= 0 and position[1] <= 2 , "incorrect position"
        assert self.board[position] == "__" , "position already filled"
        assert np.any(self.board == '__') , "Board is complete"
        assert self.win(self.me) == False and self.win(self.opponent)== False , " Game has already been won"
        self.board[position] = self.opponent
            
    def pick_best_action(self,Q,action_type,eps=None):
        '''Given a Q function return optimal action
        If action_type is 'greedy' return best action with ties broken randomly else return epsilon greedy action
        '''
        #Get possible actions
        current_state = self.board_to_state()
        actions =  self.possible_actions(current_state)
        
        best_action = []
        best_action_value = -np.Inf
        
        for action in actions:
            Q_s_a = Q[current_state][action]
            if Q_s_a == best_action_value:
                best_action.append(action)
            elif Q_s_a > best_action_value:
                best_action = [action]
                best_action_value = Q_s_a
        best_action = random.choice(best_action)

        if action_type == 'greedy':
            return self.s1_to_b2[best_action]
        else:
            assert eps != None , "Include epsilon parameter"
            n_actions =len(actions) #No of legal actions 
            p = np.full(n_actions,eps/n_actions)
            #Get index of best action
            best_action_i = actions.index(best_action)
            p[best_action_i]+= 1 - eps
            return self.s1_to_b2[np.random.choice(actions,p=p)]
        

def play_games(n_games,Q_X,Q_O,X_strategy = 'eps_greedy',O_strategy='eps_greedy',eps_X=0.05,eps_O=0.05,seed=1):
    """ Function to play tic tac toe specified no of times, and return summary of win statistics
        n_games: No of times to play the game
        Q_X: Q function for player X that gives X's policy
        Q_O: Q function for player O that gives O's policy
        X_strategy: eps_greedy or greedy
        O_strategy: eps_greedy or greedyj
        
    """
    np.random.seed(seed)
    #Dictionary for holding results of simulation
    win_stats = defaultdict(int)
    #List to  hold winning sequences of the winning player
    winning_sequences_X = []
    winning_sequences_O = []
    
    #List of final boards
    final_boards = []
   
    t_board_X = TicTacToe(player = 'X',reward_type ='action_penalty')
    t_board_O = TicTacToe(player = 'O',reward_type ='action_penalty')
    X_first_actions = [] #List to record first actions of player X
    O_first_actions = [] #List to record first actions of player O
    winning_X_first_actions = [] #List to record first actions that resulted in wins
    winning_O_first_actions = [] #List to record first actions that resulted in wins
    
    for _ in tqdm(range(n_games),position = 0 ,leave=True):
         #Boards for players X and O
        first_action_flag = True
        while True:
            #X plays first
            x_action = t_board_X.pick_best_action(Q_X,action_type=X_strategy,eps=eps_X)
            if first_action_flag == True:
                X_first_actions.append(x_action)
                
            t_board_X.my_move(x_action) #make move on X's board
            t_board_O.opponent_move(x_action) #make same move on O's board
            if t_board_X.is_game_over(): #need to end game here if X makes the winning move
                break
            #O plays second
            o_action = t_board_O.pick_best_action(Q_O,action_type=O_strategy,eps=eps_O)
            if first_action_flag == True:
                O_first_actions.append(o_action)
                first_action_flag = False
            t_board_O.my_move(o_action) #make move on O's board
            t_board_X.opponent_move(o_action) #make same move on X's board
            if t_board_O.is_game_over(): #need to end game here if O makes the winnng move
                break

        #Check who won game or if game was drawn
        if t_board_X.win('X'):
            win_stats['X_win'] += 1
            winning_sequences_X.append(t_board_X.winning_sequence)
            winning_X_first_actions.append(X_first_actions[-1])
            
        elif t_board_X.win('O'):
            win_stats['O_win'] += 1
            winning_sequences_O.append(t_board_O.winning_sequence)
            winning_O_first_actions.append(O_first_actions[-1])
        else:
            win_stats['Draw'] += 1
        final_boards.append(t_board_X.show_board())
        t_board_X.reset_board()
        t_board_O.reset_board()
    
    return win_stats,final_boards,winning_sequences_X,winning_sequences_O,X_first_actions, \
            O_first_actions,winning_X_first_actions,winning_O_first_actions


def get_win_statistics(Q_X,Q_O,sets = 5, games_in_set = 100,X_strategy = 'eps_greedy',O_strategy='eps_greedy', \
                       eps_X=1.0,eps_O=1.0):
    
    """
    Function to get winning statistics by pitting competing strategies. 
    Q_X: Q table representing the strategy of X
    Q_O: Q Table representing the strategy of O
    sets: No of sets to be played
    games_in_set: No of games in each set
    X_strategy: greedy or epsilon greedy
    O_strategy: greedy or epsilon greedy
    eps_X and eps_O: epsilon in case of epsilon greedy strategy, set to 1 for random strategy
    """
    win_stats_list = []
    winning_sequences_X_list = []
    winning_sequences_O_list = []
    X_first_actions_list = []
    O_first_actions_list = []
    winning_X_first_actions_list = []
    winning_O_first_actions_list = []

    for i in range(sets):
        win_stats, _ ,winning_sequences_X,winning_sequences_O,X_first_actions,O_first_actions, \
        winning_X_first_actions,winning_O_first_actions=  play_games(n_games=games_in_set,\
                            Q_X=Q_X,Q_O=Q_O,X_strategy = X_strategy,O_strategy=O_strategy,eps_X=eps_X,eps_O=eps_O,seed=i)
        win_stats_list.append(win_stats)
        winning_sequences_X_list.append(winning_sequences_X)
        winning_sequences_O_list.append(winning_sequences_O)
        X_first_actions_list.append(X_first_actions)
        O_first_actions_list.append(O_first_actions)
        winning_X_first_actions_list.append(winning_X_first_actions)
        winning_O_first_actions_list.append(winning_O_first_actions)
        
        #Unwrap these lists
    flatten =  lambda l:[item for sublist in l for item in sublist] 
    winning_sequences_X_list = flatten(winning_sequences_X_list)
    winning_sequences_O_list = flatten(winning_sequences_O_list)
    X_first_actions_list = flatten(X_first_actions_list)
    O_first_actions_list = flatten(O_first_actions_list)
    winning_X_first_actions_list = flatten(winning_X_first_actions_list)
    winning_O_first_actions_list = flatten(winning_O_first_actions_list)
    
    win_stats_df  = pd.DataFrame(win_stats_list)
#     stats = win_stats_df.describe()
#     lb = stats.loc['mean'] - 2 * stats.loc['std'] 
#     ub = stats.loc['mean'] + 2 * stats.loc['std']
#     results_df = pd.concat([lb,ub],axis=1)
#     results_df.columns= ['mu - 2 sd', 'mu + 2 sd']
    
    return (win_stats_df),(winning_sequences_X_list,winning_sequences_O_list),\
    (X_first_actions_list,winning_X_first_actions_list),(O_first_actions_list, winning_O_first_actions_list)


#Mapping from 2D position to 1D position
map_2d_1d = {(0, 0): 0, (0, 1): 1,(0, 2): 2, (1, 0): 3, (1, 1): 4,(1, 2): 5,(2, 0): 6,(2, 1): 7,(2, 2): 8}


def get_win_rate(first_actions_list,winning_first_actions_list):
    "Get win rate in appropriate format from experiment results"
    first_actions = dict(Counter(first_actions_list))
    winning_first_actions = dict(Counter(winning_first_actions_list))
    win_rate = np.array([winning_first_actions.get(key,0)/first_actions.get(key,1) \
                         for key in sorted(map_2d_1d.keys())]).reshape(3,3)
    
    return win_rate


def get_win_seqs(winning_sequences_list):
    "Get winning sequence stats in appropriate format from experiment results"
    temp_dict = dict(Counter(winning_sequences_list))
    win_seq_df = pd.DataFrame({'winning_sequence':list(temp_dict.keys()),'N':list(temp_dict.values())})
    
    return win_seq_df
    

def plot_results(win_statistics):
    "Function to visualize results of experiments"
    sns.set(font_scale=5)
    final_results,winning_sequences,first_actions_X,first_actions_O = win_statistics
    win_stats_df_long = pd.melt(final_results,var_name='Result',value_name='N')
    plt.subplots_adjust(hspace = 2)
    fig, axs = plt.subplots(nrows = 2, ncols=3,figsize=(100,40))
    bplot = sns.boxplot(x="Result",y="N",data=win_stats_df_long, \
                ax=axs[0,0]).set_title("Distribution of Wins,Losses and Ties")

    #Plot aggregate results
    final_results_agg = pd.DataFrame(final_results.apply(sum,0),columns=["N"])
    final_results_agg.reset_index(level=0,inplace=True)

    splot = sns.barplot(x="index",y="N",data=final_results_agg,ax=axs[0,1])
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, -12), 
                       textcoords = 'offset points')
    splot.set_title('Total No of Wins,Losses and Ties')

    win_rate_X = get_win_rate(*first_actions_X)
    win_rate_O = get_win_rate(*first_actions_O)

    _ = sns.heatmap(win_rate_X,annot=True,ax=axs[0,2]).set_title("Player X: % of wins for first move")
    _ = sns.heatmap(win_rate_O,annot=True,ax=axs[1,0]).set_title("Player O: % of wins for first move")

    #Gets stats of winning sequences for player X and player O
    win_seq_X = get_win_seqs(winning_sequences[0])
    win_seq_O = get_win_seqs(winning_sequences[1])

    _ = sns.barplot(x='winning_sequence',y='N',data=win_seq_X,
           order=['R1','R2','R3','C1','C2','C3','D1','D2'],ax=axs[1,1]).set_title("Player X: winning sequences")

    _ = sns.barplot(x='winning_sequence',y='N',data=win_seq_O,
           order=['R1','R2','R3','C1','C2','C3','D1','D2'],ax=axs[1,2]).set_title("Player O: winning sequences")

    plt.show()


def initialize_Q(S,seed = 1):
    "Given a state assign random values to each possible action"
    np.random.seed(seed)
    Q = {}
    for state in S:
        Q[state]= {}
        for i,x  in enumerate(state): # Loop through action
            if x == 0:
                Q[state][i] = np.random.rand()
    
    return Q


def moving_average(x, w):
    "Function to calculate moving average of rewards"
    return np.convolve(x, np.ones(w), 'valid') / w


def train(n_games=1000,alpha = 0.5, gamma = 0.9,train_X=True,train_O=False,is_random=True,**kwargs):
    """
    Function to train a player in a game of tic-tac-toe
    Arguments:
        n_games: Number of games on which to train
        alpha: Learning rate
        gamme: discount factor
        train_X: Flag indicating whether player X should be trained
        train_O: Flag inficating whether player O should be trained
        is_random: should actions of untrained agent be random or deterministic according to Q table
    
    """

    # If Q is not provided, randomize intially, if provided, it will be used to select actions greedily
    if "Q_X" in kwargs:
        action_type_X = "greedy"
        assert train_X == False ,"Train flag should be set to False if Q table is being provided"
        Q_X = kwargs["Q_X"]
    else:
        Q_X = initialize_Q(States_X)
        
    if "Q_O" in kwargs:
        action_type_O = "greedy"
        assert train_O == False ,"Train flag should be set to False if Q table is being provided"
        Q_O = kwargs["Q_O"]
    else:
        Q_O = initialize_Q(States_O)
    
    
    #Set epsilon value conditional on whether we are training X or O
    eps_ = lambda flag,i: 0.05*0.99**i if flag else 1.0
    
    
    #Lists to keep track of rewards earned by both players during training
    
    rewards_X = []
    rewards_O = []
    
    if train_X:
        X_action_type = 'eps_greedy'
    else:
        X_action_type = 'greedy'
        if is_random:
            X_action_type = 'eps_greedy'
    
    if train_O:
        O_action_type = 'eps_greedy'
    else:
        O_action_type = 'greedy'
        if is_random:
            O_action_type = 'eps_greedy'
            
    for i in tqdm(range(n_games),position=0,leave=True):
        
        eps = 0.05*0.99**i
        t_board_X.reset_board()
        t_board_O.reset_board()

        #X lands on empty board
        S_X = t_board_X.board_to_state()
        
        #X plays first
        eps = eps_(train_X,i)
        
            
        x_action = t_board_X.pick_best_action(Q_X,action_type = X_action_type,eps=eps)
        x_action1d = t_board_X.b2_to_s1[x_action]
        
        R_X = t_board_X.my_move(x_action) # make move on X's board
        t_board_O.opponent_move(x_action) # make same move on O's board

        while not (t_board_X.is_game_over() or t_board_O.is_game_over()):
            S_O = t_board_O.board_to_state()
            
            #O plays second
            eps = eps_(train_O,i)
            o_action = t_board_O.pick_best_action(Q_O,action_type=O_action_type,eps=eps)
            o_action1d = t_board_O.b2_to_s1[o_action]
            R_O = t_board_O.my_move(o_action) #make move on O's board
            t_board_X.opponent_move(o_action) #make same move on X's board
            if  t_board_O.is_game_over(): 
                #need to end game here if O makes the winnng move and add a reward 
                if train_O:
                    Q_O[S_O][o_action1d] += alpha*(R_O + 0 - Q_O[S_O][o_action1d]) # 0 given value of terminal state is 0
                
                if train_X:
                #Need to penalize X's previous action if game is over
                    Q_X[S_X][x_action1d] += alpha*(-R_O + 0 - Q_X[S_X][x_action1d]) 
                
                rewards_O.append(R_O)
                rewards_X.append(-R_O)
                break
            
            S_X_new = t_board_X.board_to_state() #Get new state
            #Calculate max_a Q_X(S',a)
            if train_X:
                x_action_ = t_board_X.pick_best_action(Q_X,action_type = 'greedy',eps=0.05) #best action from S_new
                x_action_1d = t_board_X.b2_to_s1[x_action_]
                Q_X[S_X][x_action1d]+= alpha*(R_X + gamma*Q_X[S_X_new][x_action_1d] - Q_X[S_X][x_action1d])
        
            S_X = S_X_new

            # X plays next
            eps = eps_(train_X,i)
            x_action = t_board_X.pick_best_action(Q_X,action_type = X_action_type,eps=eps)
            x_action1d = t_board_X.b2_to_s1[x_action]
            R_X = t_board_X.my_move(x_action) #make move on X's board
            t_board_O.opponent_move(x_action) #make same move on O's board

            if t_board_X.is_game_over(): 
                if train_O:
                    #need to end game here if X makes the winning move and make sure O's action is penalized
                    Q_O[S_O][o_action1d] += alpha*(-R_X + 0 - Q_O[S_O][o_action1d]) #0 given value of terminal state is 0
                
                if train_X:
                    #need to end game here if X makes the winning move and make sure reward is added to V
                    Q_X[S_X][x_action1d] += alpha*(R_X + 0 - Q_X[S_X][x_action1d]) #0 given value of terminal state is 0
                
                rewards_X.append(R_X)
                rewards_O.append(-R_X)
                break   


            S_O_new = t_board_O.board_to_state() #Get new state
            #Calculate max_a Q_O(S',a)
            if train_O:
                o_action_ = t_board_O.pick_best_action(Q_O,action_type = 'greedy',eps=0.05) #best action from S_new
                o_action_1d = t_board_O.b2_to_s1[o_action_]
                Q_O[S_O][o_action1d]+= alpha*(R_O + gamma*Q_O[S_O_new][o_action_1d] - Q_O[S_O][o_action1d])

            S_O = S_O_new
            
    if train_X:
        rewards = rewards_X
    elif train_O:
        rewards = rewards_O
        
    sns.set(font_scale=1)
    m_avg = moving_average(rewards,w=200)
    sns.lineplot(x=range(len(m_avg)),y=m_avg).set_title('Learning Curve')
    plt.show()
        
    return Q_X,Q_O,rewards_X,rewards_O


if __name__ == '__main__':
    t_board_X = TicTacToe(player = 'X',reward_type ='goal_reward')
    t_board_O = TicTacToe(player = 'O',reward_type ='goal_reward')
    States_X = t_board_X.my_states
    States_O = t_board_O.my_states
    Q_X = initialize_Q(States_X)
    Q_O = initialize_Q(States_O)

    #final_results,winning_sequences,first_actions_X,first_actions_O 
    win_statistics= get_win_statistics(Q_X, Q_O,sets = 10,\
    games_in_set = 100,X_strategy = 'eps_greedy',O_strategy='eps_greedy',eps_X=1.0,eps_O=1.0)# Setting eps = 1.0 ensures purely random policy
    plot_results(win_statistics)

    # Trained X vs random O
    np.random.seed(1)
    #,_,rewards_X,rewards_O = train(Q_X, Q_O,n_games=8000,alpha = 0.5, gamma = 0.9,train_X=True,train_O=False)
    Q_X,_,rewards_X,rewards_O = train(n_games=5000,alpha = 0.5, gamma = 0.9,train_X=True,train_O=False,is_random=True)
    Q_X_trained = Q_X

    
    win_statistics = get_win_statistics(Q_X_trained,Q_O,sets = 10, games_in_set = 100,X_strategy = 'greedy', O_strategy='eps_greedy',eps_X=1.0,eps_O=1.0)
    plot_results(win_statistics)

    # Random X vs trained O
    np.random.seed(1)
    #,_,rewards_X,rewards_O = train(Q_X, Q_O,n_games=8000,alpha = 0.5, gamma = 0.9,train_X=True,train_O=False)
    _,Q_O,rewards_X,rewards_O = train(n_games=20000,alpha = 0.5, gamma = 0.5,train_X=False,train_O=True,is_random=True)
    Q_O_trained = Q_O

    win_statistics = get_win_statistics(Q_X,Q_O_trained,sets = 10, games_in_set = 100,X_strategy = 'eps_greedy',O_strategy='greedy',eps_X=1.0, eps_O=1.0)
    plot_results(win_statistics)

    # Trained X vs trained O
    win_statistics = get_win_statistics(Q_X_trained,Q_O_trained,sets = 10, games_in_set = 100,X_strategy = 'greedy',O_strategy='greedy',eps_X=1.0, eps_O=1.0)
    plot_results(win_statistics)

    # Retrained X vs trained O
    np.random.seed(1)
    Q_X,Q_O,rewards_X,rewards_O = train(n_games=1000,alpha = 0.5, gamma = 0.9,train_X=True,train_O=False,is_random=False,Q_O = Q_O_trained)
    Q_X_retrained = Q_X

    win_statistics= get_win_statistics(Q_X_retrained,Q_O_trained,sets = 10, games_in_set = 100,X_strategy = 'greedy', O_strategy='greedy',eps_X=1.0,eps_O=1.0)
    plot_results(win_statistics)