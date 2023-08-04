"""
FIT2004
Assignment 2
"""
import math as m


def ideal_place(relevant):
    """
    desc: find a point that to all relevant point has the minimum distance.
    time complexity: O(n), n is the length of the relevant
    space complexity: O(n)
    :param relevant: a list of relevant points.
    :return: o_point, which is the optimal point for the kiosk
    """
    # for special case, relevant has one point or is empty.
    if len(relevant) == 0:
        return []
    if len(relevant) == 1:
        return relevant[0]
    # separately store the x and y values.
    x = []
    y = []
    for i in relevant:
        x.append(i[0])
        y.append(i[1])

    # calculate the mean of x and y
    x_sum = 0
    y_sum = 0
    for i in range(len(x)):
        x_sum += x[i]
        y_sum += y[i]

    x_mean = x_sum // len(x)
    y_mean = y_sum // len(y)

    # calculate the median of x and y
    x_idx = len(x) // 2
    y_idx = len(y) // 2

    x.sort()
    y.sort()

    x_median = x[x_idx]
    y_median = y[y_idx]
    # calculate the minimum distance from these two point.
    mean_dist = 0
    median_dist = 0
    for i in relevant:
        mean_dist += (abs(x_mean - i[0]) + abs(y_mean - i[1]))
        median_dist += (abs(x_median - i[0]) + abs(y_median - i[1]))
    # return the smaller one
    if mean_dist < median_dist:
        return [x_mean, y_mean]
    else:
        return [x_median, y_median]


# --------------------------------------------------task2-----------------------------------------------------
class Heap:
    """
    min heap class
    """

    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        """
        create minheap node
        :param v: vertex v
        :param dist: distance(from the dist[] array)
        :return: a list contain v and it dist
        time complexity: O(1)
        space : O(1)
        """
        minHeapNode = [v, dist]
        return minHeapNode

    def swapMinHeapNode(self, a, b):
        """
        swap two nodes in min heap
        :param a: node a
        :param b: node b
        :return: none
        time: O(1)
        space: O(1)
        """
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    def minHeapify(self, idx):
        """
        heapify at given idx
        :param idx:
        :return:
        time: O(log(n)), n is max idx of vertex(also the number of vertex),
        (worst case is send the tail element to the head.)
        """
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if (left < self.size and
                self.array[left][1]
                < self.array[smallest][1]):
            smallest = left

        if (right < self.size and
                self.array[right][1]
                < self.array[smallest][1]):
            smallest = right

        # The nodes to be swapped in min
        # heap if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest

            # Swap nodes
            self.swapMinHeapNode(smallest, idx)

            self.minHeapify(smallest)

    def extractMin(self):
        """
        pop the minimum element in the heap
        :return: root, which is minimum in the min heap
        time: O(log(n)), n is the element in the heap,(use the function minHeapify)
        """

        # Return NULL wif heap is empty
        if self.isEmpty():
            return

        # Store the root node
        root = self.array[0]

        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)

        return root

    def isEmpty(self):
        """
        check whether the heap is empty
        :return: True is empty, false is none empty
        time: O(1)
        """
        return True if self.size == 0 else False

    def decreaseKey(self, v, dist):
        """
        update the value of vertex v in the heap
        :param v: vertex
        :param dist: distance
        :return:
        time: O(logn), n is max idx of vertex(also the number of vertex)
        """
        # Get the index of v in  heap array

        i = self.pos[v]

        # Get the node and update its dist value
        self.array[i][1] = dist

        # Travel up while the complete tree is
        # not hepified. This is a O(Logn) loop
        while (i > 0 and self.array[i][1] <
               self.array[(i - 1) // 2][1]):
            # Swap this node with its parent
            self.pos[self.array[i][0]] = (i - 1) // 2
            self.pos[self.array[(i - 1) // 2][0]] = i
            self.swapMinHeapNode(i, (i - 1) // 2)

            # move to parent index
            i = (i - 1) // 2;

    # A utility function to check if a given
    # vertex 'v' is in min heap or not
    def isInMinHeap(self, v):
        """
        check whether c is in min heap
        :param v: vertex id
        :return: True is in, false if not in
        time: O(1)
        """
        if self.pos[v] < self.size:
            return True
        return False


class RoadGraph:
    """
    a class for generating the road graph.
    """

    def __init__(self, roads):
        """
        :param roads: set of tuples, contain vertex u, v, weight
        """
        self.roads = roads
        max_id = -1
        # get maximum ID
        for i in roads:
            if i[0] > max_id:
                max_id = i[0]
            if i[1] > max_id:
                max_id = i[1]

        # Create Graph and reverse Graph(use adjacency list)
        self.Graph = []
        self.r_Graph = []
        for i in range(max_id + 1):
            self.Graph.append([])
            self.r_Graph.append([])

        # store vertex and weight to the adjacency list
        for i in roads:
            self.Graph[i[0]].append((i[1], i[2]))
            self.r_Graph[i[1]].append((i[0], i[2]))

    def findPath(self, start, end, pred):
        """
        find the path from start point to end point, use the pred list.
        :param start: start point
        :param end: end point
        :param pred: pred list
        :return: path list
        time: O(V), V is the number of the vertices
        space: O(V)
        """
        path = [end]
        current_p = pred[end]  # find the path from end to start
        while current_p != m.inf:  # run in O(V)
            path.append(current_p)
            current_p = pred[current_p]

        path.reverse()  # run in O(V)
        return path  # return the correct order of path.

    def routing(self, start, end, chores_location):
        """
        find the shortest path from start to end and pass by one of the chores location point
        :param start: start point
        :param end: end point
        :param chores_location: set of chores point
        :return: result path
        time: O(ElogV), E is the number of the edges, V is the number of the vertices
        space: O(E+V)
        """

        g = dijkstra_traverse(self.Graph, start)  # use dijkstra algorithm find the shortest path from the start to other points
        r_g = dijkstra_traverse(self.r_Graph, end)  # use dijkstra find the shortest path from
        # end to other points(in the reverse graph)
        # which equals to the shortest path from other points to the end point.

        # get the separate dist and pred list.
        dist_g, pred_g = g[0], g[1]
        dist_rg, pred_rg = r_g[0], r_g[1]

        total_dist = [0] * len(dist_g)
        for i in range(len(dist_g)):
            total_dist[i] = dist_g[i] + dist_rg[i]  # get the total dist

        # find the minimum path through which chores location
        # run in O(V), V is the number of the vertices
        min_dist = m.inf
        chores = 0
        for i in chores_location:
            if total_dist[i] < min_dist:
                min_dist = total_dist[i]
                chores = i

        # find the path from the start to the chores location
        path_s_c = self.findPath(start, chores, pred_g)
        path_s_c.pop(-1)  # pop the duplicate chores
        # find the path from the chores location to the end
        path_c_e = self.findPath(end, chores, pred_rg)
        path_c_e.reverse()

        if (path_s_c == [] or path_c_e == [end]) and end != chores and start != chores:  # if any path not exist
            # exclude the start == chores or end == chores
            return "None"

        # combine the path and return
        result_path = path_s_c + path_c_e
        return result_path



def dijkstra_traverse(G, s):
    """
    algorithm that traverse all vertex in the graph use the dijkstra apprach.
    :param G: Graph from the GraphRoad object
    :param s: start point
    :return:dist and pred list, dist store the shortest distance from start point to the idx point,
    pred store the last point of the idx point
    time: O(ElogV), V is the number of vertices, E is number of the edges in the graph.
    space: O(V+E)
    """
    n = len(G)  # number of vertices
    dist = [m.inf] * n  # initial dist
    dist[s] = 0  # set start point to 0
    pred = [m.inf] * n

    p_q = Heap()  # priority q

    for i in range(n):  # initial heap
        p_q.array.append(p_q.newMinHeapNode(i, dist[i]))
        p_q.pos.append(i)

    p_q.pos[s] = s  # set start point position
    p_q.decreaseKey(s, dist[s])  # set the  # update start point position in heap
    p_q.size = n  # initialise the heap size

    while not p_q.isEmpty():  # if queue is not empty
        # pop the node that has the minimum distance
        new_node = p_q.extractMin()
        u = new_node[0]

        for i in G[u]:  # for each edge adjacency to u
            v = i[0]
            # relax edge

            if p_q.isInMinHeap(v) and dist[u] != m.inf and dist[u] + i[1] < dist[v]:  # if dist[u] + w(u,v) < dist[v]
                dist[v] = dist[u] + i[1]  # update dist[v]
                pred[v] = u  # update pred[v]
                p_q.decreaseKey(v, dist[v])  # update v in heap
    return dist, pred

