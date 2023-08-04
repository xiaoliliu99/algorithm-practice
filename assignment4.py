def build_network(preferences, sysadmins_per_night, max_unwanted_shifts, min_shifts):
    """
    build network in adjacency matrix
    time complexity: O(n^2), n is the number of the sysadmin.
    :param preferences: array represent the sysadmins' preference
    :param sysadmins_per_night: number of sysadmins that should be on duty each night
    :param max_unwanted_shifts: max unwanted number
    :param min_shifts: min shift
    :return: graph, which represent the network
    """

    n_sysadmin = len(preferences[0])  # number of sysadmins
    # total node in the flow network = source + sink + demand node + number(sysadmin)
    # + number(sysadmin)*2: which represent the preference + 30days.
    total_node = 1 + 1 + 1 + n_sysadmin + n_sysadmin * 2 + 30

    # create adjacency matrix for network
    graph = [[0] * total_node for _ in range(total_node)]
    # add edges to the graph

    # add edge from source to demand node, graph[0] represent the source node,
    # graph[1] represent the demand node
    graph[0][1] = 30 * sysadmins_per_night - n_sysadmin * min_shifts  # 30*sysadmins_per_night is the total flow,

    # add edges from source to selector node,
    # graph[3]~graph[n_sysadmin+2] represent the selector nodes.
    # capacity is min_shifts
    for i in range(3, n_sysadmin + 3):
        graph[0][i] = min_shifts

    # add edge from demand node to selector nodes.
    for i in range(3, n_sysadmin + 3):
        graph[1][i] = 30 - min_shifts

    # add edge from selector to preference node.
    # graph[3+n_sysadmin]~graph[3+2*n_sysadmin-1] is not interested selection
    # graph[3+2*n_sysadmin]~graph[3+3*n_sysadmin-1] is the interested selection
    for i in range(3, n_sysadmin + 3):
        graph[i][i + n_sysadmin] = max_unwanted_shifts
        graph[i][i + n_sysadmin * 2] = 30

    # add edge from preference to days
    # graph[3+3*n_sysadmin] ~ graph[3+3*n_sysadmin+30-1] represent the days.
    for i in range(30):
        for j in range(n_sysadmin):
            if preferences[i][j] == 1:
                graph[j + 3 + 2 * n_sysadmin][i + 3 + 3 * n_sysadmin] = 1  # represent sysadmin j is interested in day i
            else:
                graph[j + 3 + n_sysadmin][i + 3 + 3 * n_sysadmin] = 1  # sysadmin j not interested in day i.

    # add edges from days to sink node, graph[2] represent the sink
    for i in range(3 + 3 * n_sysadmin, 3 + 3 * n_sysadmin + 30):
        graph[i][2] = sysadmins_per_night

    return graph


class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.ROW = len(graph)

    def BFS(self, s, t, parent):
        """
        bfs algorithm to check whether path exist.
        time complexity: O(n^2), n is the number of the sysadmin.
        :param s: source
        :param t: target
        :param parent: parent of the node(store the path)
        :return: true is exist the path from s to t, otherwise, return false
        """

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    # If find a connection to the sink node，
                    # return true
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        # if not reach, return false
        return False

    def FordFulkerson(self, source, sink):
        """
        Returns the maximum flow from s to t in the given graph
        time complexity: O(EF), E is the number of edges, max(E) = 1+n+2*n+30*n+30 = 3*n
        F is the max flow, which equals to 30*sysadmins_per_night
        hence total: O(n^2)
        :param source
        :param sink
        :return: max flow of the network
        """

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


def allocate(preferences, sysadmins_per_night, max_unwanted_shifts, min_shifts):
    """
    allocate work days depend on the preference for the sysadmin.
    time complexity: O(n^2), n is the number of the sysadmin.
    :param preferences: array represent the sysadmins' preference
    :param sysadmins_per_night: number of sysadmins that should be on duty each night
    :param max_unwanted_shifts: max unwanted number
    :param min_shifts: min shift
    :return: None is no solution, schedule array if solution exist.
    """
    n_sysadmin = len(preferences[0])
    graph = build_network(preferences, sysadmins_per_night, max_unwanted_shifts, min_shifts)
    g = Graph(graph)
    max_flow = g.FordFulkerson(0, 2)  # 0 is the source and 2 represent the sink node.

    if max_flow != 30 * sysadmins_per_night:
        return None

    # if result exist, build result array
    wanted = []  # is 1 if scheduled in this day
    unwanted = []  # is 1 if scheduled in this day

    for i in range(3 + n_sysadmin, 3 + 2 * n_sysadmin):
        unwanted.append(graph[i][3 + 3 * n_sysadmin:])

    for i in range(3 + 2 * n_sysadmin, 3 + 3 * n_sysadmin):
        wanted.append(graph[i][3 + 3 * n_sysadmin:])

    result = [[0] * n_sysadmin for _ in range(30)]

    for i in range(30):
        for j in range(n_sysadmin):
            result[i][j] = wanted[j][i] + unwanted[j][i]
    return result


####################################task2##################################
class Node:
    def __init__(self, value, no_chain):
        """
        construct the Node class
        :param value: value of the node
        :param no_chain: number of chain
        """
        self.value = value
        self.children = [-1] * 27  # children list for 26 character and $
        self.identity = [-1] * no_chain  # index based array to store the idx in ith chain
        self.no_pass = 0  # number of pass through this node


class EventsTrie:
    def __init__(self, timelines):
        """
        build the trie
        time complexity: O(N*M^2), N is the number of chain, M is the len of longest chain.
        space: O(N*M^2)
        :param timelines:
        """
        self.timelines = timelines
        self.no_chain = len(timelines)
        self.root = Node(0, self.no_chain)  # set the root node.
        self.no_occurence = [[-1] * 3 for _ in
                             range(self.no_chain + 1)]  # store [chain_id, start_idx, end_idx] for occurence
        self.current = self.root  # current node

        suffix_list = [[] for _ in range(self.no_chain)]
        # create suffix list
        for i in range(self.no_chain):
            for j in range(len(timelines[i])):
                suffix_list[i].append(timelines[i][j:])

        # insert suffix to trie
        for i in range(self.no_chain):  # for each chain, take time N
            for j in range(len(suffix_list[i])):  # for each suffix,  take time M
                self.current = self.root  # initial the root
                suffix = suffix_list[i][j]
                for k in range(len(suffix)):  # for each event(character), take time M
                    child_idx = ord(suffix[k]) - 97  # children idx, 0 for a, 1 for b ......

                    if self.current.children[child_idx] == -1:  # if current node do not have this child
                        new_node = Node(suffix[k], self.no_chain)
                        new_node.no_pass += 1  # add pass number
                        new_node.identity[i] = j + k  # update identity
                        if self.no_occurence[new_node.no_pass][2] - self.no_occurence[new_node.no_pass][
                            1] < k:
                            # if number of occurence < new occurence
                            self.no_occurence[new_node.no_pass][0] = i  # chain number
                            self.no_occurence[new_node.no_pass][1] = j  # start idx
                            self.no_occurence[new_node.no_pass][2] = j + k  # end idx

                        self.current.children[child_idx] = new_node  # add node to children list
                        self.current = new_node  # update current node
                    else:  # if this child already exist

                        self.current = self.current.children[child_idx]
                        if self.current.identity[i] == -1:  # if not visited for new chain
                            self.current.identity[i] = j + k
                            self.current.no_pass += 1

                        if self.no_occurence[self.current.no_pass][2] - self.no_occurence[self.current.no_pass][1] < k:
                            # if number of occurence < new occurence
                            self.no_occurence[self.current.no_pass][0] = i  # chain number
                            self.no_occurence[self.current.no_pass][1] = j  # start idx
                            self.no_occurence[self.current.no_pass][2] = j + k  # end idx

    def getLongestChain(self, noccurence):
        """
        get the longest chain
        time complexity: O(K), only need to slice the string
        :param noccurence:
        :return: return sub chain stastify the noccurence, otherwise, return None.
        """

        if noccurence > self.no_chain:
            return None
        chain_no = self.no_occurence[noccurence][0]
        start = self.no_occurence[noccurence][1]
        end = self.no_occurence[noccurence][2]
        return self.timelines[chain_no][start:end+1]

