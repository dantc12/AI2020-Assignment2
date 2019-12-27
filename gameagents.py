from agent import *
from searchagents import SearchNode


class GameAgent(Agent):
    def __init__(self, index, initial_state, other_initial_state, game_type, cutoff_val):
        Agent.__init__(self, index, initial_state)
        self.game_type = game_type
        self.other_curr_state = other_initial_state
        self.cutoff_val = cutoff_val
        self.curr_node = SearchNode(None, 0, cutoff_val, None, initial_state, other_initial_state, initial_state,
                                    other_initial_state)

    def action(self, percept):
        """
        :type percept: environment.Environment
        """
        ag_env = percept
        if not self.seq:
            first_player = ag_env.agents[0]
            second_player = ag_env.agents[1]
            first_player_state = first_player.curr_state
            second_player_state = second_player.curr_state
            first_player_node = SearchNode(None, 0, self.cutoff_val, None, first_player_state, second_player_state,
                                           first_player_state, second_player_state)
            print_info("---------------------AGENT_CALC_SEQ---------------------")
            seq, best_game_score = first_player_node.alpha_beta_search_with_seq(self.game_type)
            first_player.seq = seq[::2]
            second_player.seq = seq[1::2]
            print_info("---------------------A" + str(first_player.index) + "_CHOSE_SEQ_" + str(first_player.seq) +
                       "---------------------")
            print_info("---------------------A" + str(second_player.index) + "_CHOSE_SEQ_" + str(second_player.seq) +
                       "---------------------")
        action = self.seq.pop(0)
        print_info("---------------------A" + str(self.index) + "_DOING_" + str(action)+"---------------------")
        return action