import math


def floydWarshall(graph):
    """
    floyd warshall algorithm to find the shortest path for all pairs of vertices
    :param graph: represent the  travel_days[x][y] from x to y.
    :return: the shortest path 2d array between all pairs of vertices.
    time complexity: O(n^3), n is the number of cities(vertex)
    """
    V = len(graph)
    dist = graph

    for k in range(V):

        # pick all vertices as source one by one
        for i in range(V):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist


def best_revenue(revenue, travel_days, start):
    """
    find the best revenue from start city within travel days.
    :param revenue: revenue[x][y] represent the revenue we could get in day x in city y
    :param travel_days: travel_days[x][y] represent the path from x to y.
    :param start:start city
    :return: the maximum revenue we can make in the total days.
    time complexity:O(n^3) for floyd warshall, and O(d*n^2) for construct the result array.
    total: O(n^3+d*n^2) = O(n^2*(d+n))
    """
    g_len = len(travel_days)  # graph length
    d_len = len(revenue)  # total days

    # convert -1 to inf and diagonal to 0 (for floyd warshall)
    for i in range(g_len):
        for j in range(g_len):
            if travel_days[i][j] == -1:
                travel_days[i][j] = math.inf
            if i == j:
                travel_days[i][j] = 0

    # use floyd warshall algorithm, find the shortest path.
    dist = floydWarshall(travel_days)

    result = []  # initial result array,-1 represent the not achieved
    for i in range(g_len):
        result.append([-1] * d_len)

    # initial the first choose(from start to other cities in day 0)
    for i in range(len(dist[start])):
        if i != start:  # from start to other cities
            day = dist[start][i] - 1
            result[i][day] = 0
        else:  # from start to start
            result[i][0] = revenue[0][start]

    # construct the result
    for d in range(d_len):  # for each day, start from 0
        for i in range(g_len):  # for each city
            if result[i][d] != -1:  # if it can reach city i in this day d
                for k in range(g_len):  # from city i go to city k
                    value = result[i][d]  # maximum revenue we can get in day d city i
                    travel_day = dist[i][k] + d + 1  # cost of make an action

                    if travel_day < d_len:  # if we can reach a city before the last day.
                        if value + revenue[travel_day][k] > result[k][travel_day]:
                            result[k][travel_day] = value + revenue[travel_day][k]

    # find the maximum value(in the last day)
    max_v = -1
    for i in result:
        if i[-1] > max_v:
            max_v = i[-1]
    return max_v



def mergeSort(array):
    """
    use merge sort algorithm to sort the given array
    :param array: array need to be sorted
    :return: sorted array
    """
    if len(array) > 1:

        # Finding the mid of the array
        mid = len(array) // 2

        # Dividing the array elements
        L = array[:mid]

        # into 2 halves
        R = array[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i][2] < R[j][2]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            array[k] = R[j]
            j += 1
            k += 1
    return array


def findCloset(target, arr):
    """
    use binary search algorithm find the closet number index in the array
    :param target:
    :param arr:
    :return: closet number
    time complexity: O(logn), n is length of the array.
    """
    n = len(arr)
    # Corner cases
    if target < arr[0][2]:
        return -1
    if target > arr[n - 1][2]:
        return n - 1

    # Doing binary search
    i = 0  # low
    j = n - 1  # high
    mid = 0
    t = -1
    while i <= j:
        mid = (i + j) // 2

        if target < arr[mid][2]:
            j = mid - 1
        elif target > arr[mid][2]:
            t = mid
            i = mid + 1
        else:
            return i
    return t


def hero(attack):
    """
    find the which group of multiverse we need to go to make defend the most attack.
    :param attack: attack list
    :return: return_array, list of multiverse
    time complexity: O(nlogn) =>O(logn+nlogn) merge sort + construct result
    space complexity: O(n)
    """
    n = len(attack)
    result = [0] * (n + 1)  # result array
    back_t = 0  # used for back tracking
    return_array = []  # return solution
    for i in range(n + 1):
        return_array.append([])

    print(return_array)
    # use merge algorithm sort the end time in ascending order
    mergeSort(attack)

    # construct the result array
    for i in range(1, n + 1):
        j = findCloset(attack[i - 1][1], attack)
        if j != -1:
            result[i] = max(result[i - 1], result[j + 1] + attack[i - 1][3])
            # store solution
            if result[i] == result[j + 1] + attack[i - 1][3]:
                return_array[i] = return_array[j + 1]
                return_array[i].append(attack[i - 1])  # if pick this mutiverse

        else:
            result[i] = attack[i - 1][3]
            return_array[i].append(attack[i - 1])

    return return_array[-1]


