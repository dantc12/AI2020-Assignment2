from copy import deepcopy

from graph import Vertex
import agent
from agent import AgentState
from helper_funcs import print_debug, dijkstra


class SearchNode:
    T = 0

    def __init__(self, parent, depth, cutoff_limit, action, curr_p_state, other_p_state, curr_p_initial_state,
                 other_p_initial_state):
        """
        :type parent: SearchNode|None
        :type depth: float
        :type curr_p_state: AgentState
        :type other_p_state: AgentState
        :type curr_p_initial_state: AgentState
        :type other_p_initial_state: AgentState
        """
        self.parent_node = parent
        self.action = action
        self.curr_p_initial_state = deepcopy(curr_p_initial_state)
        self.other_p_initial_state = deepcopy(other_p_initial_state)
        self.curr_p_state = deepcopy(curr_p_state)
        self.other_p_state = deepcopy(other_p_state)
        self.cutoff_limit = cutoff_limit
        self.depth = depth

        self.expansions = 0

    def terminal_test(self):
        return self.curr_p_state.is_terminated and self.other_p_state.is_terminated

    def cutoff_test(self):
        return self.cutoff_limit <= self.depth

    def alpha_beta_search_with_seq(self, game_type):
        full_seq, best_game_score = self.max_value_with_seq([], -float("inf"), -float("inf"), game_type)
        # print_debug("Chose: " + str(seq))
        return full_seq, best_game_score

    def max_value_with_seq(self, seq, alpha, beta, game_type):
        s_depth = str("\t" * int(self.depth*2))
        if self.terminal_test():
            # print_info("A1 UTILITY " + str(self.utility(game_type)) + ", seq: " + str(seq))
            # res = self.utility(game_type)
            game_score = (self.curr_p_state.state_score, self.other_p_state.state_score)
            my_util = utility_calc(game_type, game_score[0], game_score[1])
            print_debug(s_depth + "MAX_NODE_UTILITY: " + str(seq) + " = " + str(game_score) +
                        ", " + str(my_util))
            return seq, game_score
        if self.cutoff_test():
            h_score = heuristic(self, game_type)
            print_debug(s_depth + "MAX_NODE_HEURISTIC: " + str(seq) + " = " + str(h_score))
            return seq, h_score
        best_util = -float("inf")
        best_game_score = None
        best_seq = None
        successor_nodes = expand(self, None)

        # tmp_successor_nodes = successor_nodes[:]
        # prev_states = [self.curr_p_state]
        # tmp_n = self
        # while tmp_n.parent_node and tmp_n.parent_node.parent_node:
        #     prev_states.append(tmp_n.parent_node.parent_node.curr_p_state)
        #     tmp_n = tmp_n.parent_node.parent_node
        # for n in tmp_successor_nodes:
        #     for prev_s in prev_states:
        #         if repeat_state_check(prev_s, n.other_p_state):
        #             successor_nodes.remove(n)
        #             print_debug(s_depth + "MAX_NODE_REMOVED_REPEAT_" + str(n.action) + "_" + str(n.other_p_state))
        # if not successor_nodes:
        #     return seq + ["REMOVED_REPEATS"], v

        for n in successor_nodes:
            print_debug(s_depth + "MAX_NODE_DEPTH_" + str(self.depth) + "_EXPLORING: " + str(n.action))
            curr_chosen_seq, curr_chosen_game_score = n.min_value_with_seq(seq + [n.action], alpha, beta, game_type)
            curr_chosen_util = utility_calc(game_type, curr_chosen_game_score[0], curr_chosen_game_score[1])
            print_debug(s_depth + "MAX_NODE_DEPTH_" + str(self.depth) + "_EXPLORED: " + str(curr_chosen_seq) +
                        " = " + str(curr_chosen_game_score) + ", " + str(curr_chosen_util))
            if curr_chosen_util >= best_util:
                best_seq = curr_chosen_seq
                best_game_score = curr_chosen_game_score
                best_util = curr_chosen_util
                if game_type == 'Adversarial (zero sum game)':
                    other_p_util = utility_calc(game_type, best_game_score[1], best_game_score[0])
                    if other_p_util < beta:
                        print_debug(s_depth + "MAX_NODE_DEPTH_" + str(self.depth) + "_PRUNED: " + str(best_seq) +
                                    " = " + str(best_game_score) + ", " + str(other_p_util) + "<" + str(beta))
                        return best_seq + ["PRUNED"], best_game_score
            alpha = max(alpha, best_util)
        print_debug(s_depth + "MAX_NODE_DEPTH_" + str(self.depth) + "_CHOSE: " + str(best_seq) + " = " +
                    str(best_game_score) + ", " + str(best_util))
        return best_seq, best_game_score

    def min_value_with_seq(self, seq, alpha, beta, game_type):
        s_depth = str("\t" * int(self.depth * 2))
        if self.terminal_test():
            # res = self.utility(game_type)
            # print_info("A2 UTILITY " + str(self.utility(game_type)) + ", seq: " + str(seq))
            game_score = (self.other_p_state.state_score, self.curr_p_state.state_score)
            my_util = utility_calc(game_type, game_score[1], game_score[0])
            print_debug(s_depth + "MIN_NODE_UTILITY: " + str(seq) + " = " + str(game_score) +
                        ", " + str(my_util))
            return seq, game_score
        if self.cutoff_test():
            raise Exception("GOT INTO MIN NODE HEURSITC")
            # h_score = heuristic(self, game_type)
            # print_debug(s_depth + "MAX_NODE_HEURISTIC: " + str(seq) + " = " + str(h_score))
            # return seq, h_score
        best_util = -float("inf")
        best_game_score = None
        best_seq = None
        successor_nodes = expand(self, None)

        # tmp_successor_nodes = successor_nodes[:]
        # prev_states = [self.curr_p_state]
        # tmp_n = self
        # while tmp_n.parent_node and tmp_n.parent_node.parent_node:
        #     prev_states.append(tmp_n.parent_node.parent_node.curr_p_state)
        #     tmp_n = tmp_n.parent_node.parent_node
        # for n in tmp_successor_nodes:
        #     for prev_s in prev_states:
        #         if repeat_state_check(prev_s, n.other_p_state):
        #             successor_nodes.remove(n)
        #             print_debug(s_depth + "MIN_NODE_REMOVED_REPEAT_" + str(n.action) + "_" + str(n.other_p_state))
        # if not successor_nodes:
        #     return seq + ["REMOVED_REPEATS"], v

        for n in successor_nodes:
            print_debug(s_depth + "MIN_NODE_DEPTH_" + str(self.depth) + "_EXPLORING: " + str(n.action))
            curr_chosen_seq, curr_chosen_game_score = n.max_value_with_seq(seq + [n.action], alpha, beta, game_type)
            curr_chosen_util = utility_calc(game_type, curr_chosen_game_score[1], curr_chosen_game_score[0])
            print_debug(s_depth + "MIN_NODE_DEPTH_" + str(self.depth) + "_EXPLORED: " + str(curr_chosen_seq) +
                        " = " + str(curr_chosen_game_score) + ", " + str(curr_chosen_util))
            if curr_chosen_util >= best_util:
                best_seq = curr_chosen_seq
                best_game_score = curr_chosen_game_score
                best_util = curr_chosen_util
                if game_type == 'Adversarial (zero sum game)':
                    other_p_util = utility_calc(game_type, best_game_score[0], best_game_score[1])
                    if other_p_util < alpha:
                        print_debug(s_depth + "MIN_NODE_DEPTH_" + str(self.depth) + "_PRUNED: " + str(best_seq) +
                                    " = " + str(best_game_score) + ", " + str(other_p_util) + "<" + str(alpha))
                        return best_seq + ["PRUNED"], best_game_score
            beta = max(beta, best_util)
        print_debug(s_depth + "MIN_NODE_DEPTH_" + str(self.depth) + "_CHOSE: " + str(best_seq) + " = " +
                    str(best_game_score) + ", " + str(best_util))
        return best_seq, best_game_score

    def __eq__(self, other):
        """
        :type other: SearchNode
        """
        return self.parent_node == other.parent_node and \
               self.action == other.action and \
               self.curr_p_initial_state == other.curr_p_initial_state and \
               self.curr_p_state == other.curr_p_state and \
               self.depth == other.depth

    def __str__(self):
        return "Node: (" + str(self.curr_p_state.curr_location) + ", " + str(self.curr_p_state.p_carrying) + ", " + \
               str(self.curr_p_state.p_saved) + ", " + str(self.curr_p_state.time) + ", " + \
               str(self.curr_p_state.is_terminated) + ")"

# def result(curr_state, action):
#     """
#     :type action: str
#     :type curr_state: AgentState
#     """
#     res_state = deepcopy(curr_state)
#     opt_actions, opt_states = successor_fn(None, res_state)
#     for i in range(len(opt_actions)):
#         if opt_actions[i] == action:
#             return opt_states[i]
#     raise Exception("INVALID ACTION IN RESULT FUNCTION")


def successor_fn(problem, curr_state):
    """
    :type curr_state: agent.AgentState
    """
    res_state = deepcopy(curr_state)
    # Check for hurricane or  termination
    if res_state.state_hurricane_check() or res_state.is_terminated:
        return None, res_state
    if res_state.is_traversing:
        res_state.search_state_traverse_update()
        if res_state.state_hurricane_check():
            res_state.state_score -= (res_state.k_value + res_state.p_carrying)
            res_state.is_terminated = True
        else:
            res_state.state_pickup_loadoff_update()
        return None, res_state
    actions = []  # type: list[str]
    res_states = []  # type:list[AgentState]
    # Removing blocked edges
    edges = [edge for edge in curr_state.curr_location.connected_edges if not edge.is_blocked]
    for edge in edges:
        actions.append(str(edge))
        dest_state = deepcopy(curr_state)
        dest_state.search_state_traverse(edge)
        if dest_state.state_hurricane_check():
            dest_state.state_score -= (dest_state.k_value + dest_state.p_carrying)
            dest_state.is_terminated = True
        else:
            if not dest_state.is_traversing:
                dest_state.state_pickup_loadoff_update()
        dest_state.state_v_people_update()
        res_states.append(dest_state)

    # The option of a terminate action
    actions.append("TERMINATE")
    res_state.state_terminate()
    res_state.time_update()
    res_state.state_v_people_update()
    res_states.append(res_state)

    return [(actions[i], res_states[i]) for i in range(len(actions))]


def heuristic(node, game_type):
    """
    :type game_type: str
    :type node: SearchNode
    """

    curr_p_state = deepcopy(node.curr_p_state)
    other_p_state = deepcopy(node.other_p_state)

    curr_p_h_score = curr_p_state.state_score
    other_p_h_score = other_p_state.state_score

    # Checking if they're dead anyway
    if curr_p_state.is_terminated:
        return curr_p_h_score, heuristic_solo(other_p_state)
    elif other_p_state.is_terminated:
        return heuristic_solo(curr_p_state), other_p_h_score

    # Check if traversing and will die anyway
    while curr_p_state.is_traversing and not curr_p_state.is_terminated:
        curr_p_state = successor_fn(None, curr_p_state)[1]
    if curr_p_state.is_terminated:
        return curr_p_state.state_score, heuristic_solo(other_p_state)
    while other_p_state.is_traversing and not other_p_state.is_terminated:
        other_p_state = successor_fn(None, other_p_state)[1]
    if other_p_state.is_terminated:
        return heuristic_solo(curr_p_state), other_p_state.state_score

    curr_p_start_v = curr_p_state.curr_location  # type: Vertex
    other_p_start_v = other_p_state.curr_location  # type: Vertex
    graph = curr_p_start_v.graph
    g_vertices = graph.vertices
    curr_p_distances, curr_p_paths = dijkstra(graph, curr_p_start_v)
    other_p_distances, other_p_paths = dijkstra(graph, other_p_start_v)

    if not can_reach_shelter(g_vertices, curr_p_distances, curr_p_state.time):
        return (curr_p_state.state_score - curr_p_state.k_value - curr_p_state.p_carrying), \
                   heuristic_solo(other_p_state)
    if not can_reach_shelter(g_vertices, other_p_distances, other_p_state.time):
        return heuristic_solo(curr_p_state), (other_p_state.state_score - other_p_state.k_value -
                                              other_p_state.p_carrying)

    if curr_p_state.time == other_p_state.time:
        return heuristic_non_traverse(curr_p_state, other_p_state)
    else:
        curr_first = False
        if curr_p_state.time < other_p_state.time:
            faster_p_state = curr_p_state
            slower_p_state = other_p_state
            curr_first = True
        else:
            faster_p_state = other_p_state
            slower_p_state = curr_p_state
        slower_dest_v = slower_p_state.curr_location
        if game_type == 'Adversarial (zero sum game)':
            slower_dest_v_ppl_count = faster_p_state.v_people[slower_dest_v.index-1]
            if not slower_dest_v.is_shelter() and slower_dest_v_ppl_count > 0:
                curr_p_time2reach = time2reach(curr_p_distances, slower_dest_v, faster_p_state.time)
                t_distances, t_paths = dijkstra(graph, slower_dest_v)
                if -1 < curr_p_time2reach and faster_p_state.time + curr_p_time2reach < slower_p_state.time and \
                        can_reach_shelter(g_vertices, t_distances, faster_p_state.time + curr_p_time2reach):
                    faster_p_state.curr_location = slower_dest_v
                    faster_p_state.p_carrying += slower_dest_v_ppl_count
                    slower_p_state.p_carrying -= slower_dest_v_ppl_count
                    faster_p_state.time += curr_p_time2reach
                    faster_p_state.v_people[slower_dest_v.index-1] = 0
                    slower_p_state.v_people[slower_dest_v.index-1] = 0
                    # NOT THE SMARTEST HEURISTIC BUT PRETTY GOOD
                    if curr_first:
                        return heuristic_solo(faster_p_state), heuristic_solo(slower_p_state)
                    else:
                        return heuristic_solo(slower_p_state), heuristic_solo(faster_p_state)
            return heuristic_solo(curr_p_state), heuristic_solo(other_p_state)
        else:
            h1 = heuristic_solo(curr_p_state)
            h2 = heuristic_solo(other_p_state)
            if h1 + h2 > curr_p_state.ppl2save:
                return (float(h1)/float(h1+h2))*curr_p_state.ppl2save, (float(h2)/float(h1+h2))*curr_p_state.ppl2save
            else:
                return h1, h2


def heuristic_solo(state):
    """
    :type state: AgentState
    """
    # If will terminate in this node, amount of people we won't be able to save
    if state.is_terminated:
        return state.state_score

    start_state = deepcopy(state)
    while start_state.is_traversing and not start_state.is_terminated:
        start_state = successor_fn(None, start_state)[1]
    if start_state.is_terminated:
        return start_state.state_score

    start_v = start_state.curr_location

    # MY ADDITION: if arrived in "last minute" to location, need to see if we can even leave it.
    if start_v.deadline == start_state.time:
        if min([e.weight for e in start_v.connected_edges]) > 1 and not start_v.is_shelter():
            return start_state.state_score - start_state.k_value - start_state.p_carrying

    graph = start_state.curr_location.graph
    g_vertices = graph.vertices
    distances, paths = dijkstra(graph, start_v)

    if not can_reach_shelter(g_vertices, distances, start_state.time):
        return start_state.state_score - start_state.k_value - start_state.p_carrying

    # Our standard heuristic:  people we know we can reach and save
    res = 0
    for i in range(len(g_vertices)):
        if g_vertices[i].deadline >= start_state.time and distances[i] > (g_vertices[i].deadline - start_state.time):
            t_distances, t_paths = dijkstra(graph, g_vertices[i])
            if can_reach_shelter(g_vertices, t_distances, start_state.time + distances[i]):
                res += start_state.v_people[i]
    return start_state.state_score + start_state.p_carrying + res


def heuristic_non_traverse(curr_p_state, other_p_state):
    """
    :type other_p_state: AgentState
    :type curr_p_state: AgentState
    """
    # We're assuming they are not traversing and have the same time value!
    curr_p_start_v = curr_p_state.curr_location  # type:Vertex
    other_p_start_v = other_p_state.curr_location  # type:Vertex
    time = curr_p_state.time

    # MY ADDITION: if arrived in "last minute" to location, need to see if we can even leave it.
    if curr_p_start_v.deadline == time:
        if min([e.weight for e in curr_p_start_v.connected_edges]) > 1 and not curr_p_start_v.is_shelter():
            return (curr_p_state.state_score - curr_p_state.k_value - curr_p_state.p_carrying), \
                   heuristic_solo(other_p_state)
    if other_p_start_v.deadline == time:
        if min([e.weight for e in other_p_start_v.connected_edges]) > 1 and not other_p_start_v.is_shelter():
            return heuristic_solo(curr_p_state), (other_p_state.state_score - other_p_state.k_value -
                                                  other_p_state.p_carrying)

    graph = curr_p_start_v.graph
    g_vertices = graph.vertices  # type: list[Vertex]
    curr_p_distances, curr_p_paths = dijkstra(graph, curr_p_start_v)
    other_p_distances, other_p_paths = dijkstra(graph, other_p_start_v)

    if not can_reach_shelter(g_vertices, curr_p_distances, time):
        return (curr_p_state.state_score - curr_p_state.k_value - curr_p_state.p_carrying), \
                   heuristic_solo(other_p_state)
    if not can_reach_shelter(g_vertices, other_p_distances, time):
        return heuristic_solo(curr_p_state), (other_p_state.state_score - other_p_state.k_value -
                                              other_p_state.p_carrying)

    for i in range(len(g_vertices)):
        dest_v = g_vertices[i]
        dest_v_ppl_count = curr_p_state.v_people[i]
        if not dest_v.is_shelter() and dest_v_ppl_count > 0 and dest_v.deadline >= time:
            curr_p_time2reach = time2reach(curr_p_distances, dest_v, time)
            other_p_time2reach = time2reach(other_p_distances, dest_v, time)
            t_distances, t_paths = dijkstra(graph, dest_v)
            if curr_p_time2reach > -1 and can_reach_shelter(g_vertices, t_distances, time + curr_p_time2reach):
                curr_p_state.state_score += dest_v_ppl_count
            elif curr_p_time2reach == -1 and other_p_time2reach > -1 and \
                    can_reach_shelter(g_vertices, t_distances, time + other_p_time2reach):
                other_p_state.state_score += dest_v_ppl_count

    return curr_p_state.state_score + curr_p_state.p_carrying, other_p_state.state_score + other_p_state.p_carrying


def time2reach(distances, dest_v, time):
    """
    :type time: int
    :type distances: list[int]
    :type dest_v: Vertex
    """
    if dest_v.deadline - time >= distances[dest_v.index-1]:
        return distances[dest_v.index-1]
    else:
        return -1


def can_reach_shelter(vertices, distances, time):
    """
    :type time: int
    :type distances: list[int]
    :type vertices: list[Vertex]
    """
    for i in range(len(vertices)):
        if vertices[i].is_shelter() and time2reach(distances, vertices[i], time) >= 0:
            return True
    return False


def expand(node, problem):
    """
    :type node: SearchNode
    """
    successors = []  # type: list[SearchNode]
    successor_fn_output = successor_fn(problem, node.curr_p_state)
    if not successor_fn_output[0]:
        res_state = successor_fn_output[1]
        updated_other_p_state = deepcopy(node.other_p_state)
        updated_other_p_state.v_people = res_state.v_people
        return [SearchNode(node, node.depth + 0.5, node.cutoff_limit, None, updated_other_p_state,
                                          res_state, node.other_p_initial_state, node.curr_p_initial_state)]
    for (action, result_state) in successor_fn_output:
        updated_other_p_state = deepcopy(node.other_p_state)
        updated_other_p_state.v_people = result_state.v_people
        s = SearchNode(node, node.depth + 0.5, node.cutoff_limit, action, updated_other_p_state, result_state,
                       node.other_p_initial_state, node.curr_p_initial_state)
        successors.append(s)
    return successors

# def repeat_state_check(been_state, dest_state):
#     """
#
#     :type been_state: AgentState
#     :type dest_state: AgentState
#     """
#     return been_state.curr_location == dest_state.curr_location and \
#            been_state.p_carrying == dest_state.p_carrying and \
#            been_state.p_saved == dest_state.p_saved and \
#            been_state.is_terminated == dest_state.is_terminated
#
#     # return False


def utility_calc(game_type, score_curr_p, score_other_p):
    res = None
    if game_type == 'Adversarial (zero sum game)':
        res = score_curr_p - score_other_p
    elif game_type == 'Semi-cooperative':
        exp = pow(2, score_other_p)
        if score_other_p >= 0:
            res = score_curr_p + 0.5 + ((1 - 1 / float(exp)) / 2.0)
        else:
            res = score_curr_p + 0.5 + ((-1 + exp) / 2.0)
    elif game_type == 'Fully cooperative':
        res = score_curr_p + score_other_p
    return res
