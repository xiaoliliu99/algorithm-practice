"""
Name: Xiaoli Liu
ID: 30158087
Description: FIT2004 Assignment1
Task1: Partial Wordle Trainer
Task2: Finding a Local Maximum
"""
import math


# Task1

def normalize_word(word, target):
    """
    normalize the word and target word,
    convert word to 26 size array
    and compare with target word

    time complexity: O(M), M is the length of the word and the target.
    when convert word and target to array, need traverse the word and target.
    compare two 26 size array is constant, hence is O(1)
    :param word:
    :param target:
    :return: if word==target, return True, if not return False.
    """
    target_array = [0] * 26  # 26 spaces array, idx represent english letter.
    for i in target:  # convert target word to fix size array.(use ascii code)
        target_array[ord(i) - 97] += 1

    word_array = [0] * 26
    for j in word:  # convert word to fix size array.
        word_array[ord(j) - 97] += 1

    # compare two array, if word== target, return True
    if word_array == target_array:
        return True
    else:
        return False


def check_position(word, target, marker):
    """
    use marker check in that position, whether the word letter is equal to target letter.
    time complexity: O(M), M is the length of the word.
    need to traverse the word
    :param word:
    :param target:
    :param marker:
    :return: if word pass the marker condition, then return true, else false.
    """
    for i in range(len(word)):
        if marker[i] == 1 and word[i] != target[i]:
            return False
        if marker[i] == 0 and word[i] == target[i]:
            return False
    return True


def radix_pass(array, size, digit, base):
    """
    stable counting sort.
    time complexity: O(N), N is size of the array.(O(base*N) -> base is 26, is constant and small, hence is O(N))

    :param array: array we want to sort
    :param size: size of the array
    :param digit: digit th of the letter in the word we want to sort
    :param base: base of the element, at here is 26, cause the alphabet
    :return: sorted array by digit th letter.
    """
    output = [0] * size
    count = [0] * (base + 1)  # use one more position to store the position
    min_base = ord('a') - 1

    for item in array:  # set up count array
        letter = ord(item[digit]) - min_base
        count[letter] += 1

    for i in range(len(count) - 1):  # Accumulate counts
        count[i + 1] += count[i]

    for word in array:
        # Get index of current letter of item at index col in count array
        letter = ord(word[digit]) - min_base
        output[count[letter] - 1] = word
        count[letter] -= 1

    return output


def radix_sort(array):
    """
    radix sort
    time complexity: O(N*M), N is the size of the array, M is the length of the word.
    :param array:
    :return: sorted array
    """
    M = len(array[0])  # length of the word
    for m in range(M - 1, -1, -1):  # sort letter from left to right
        array = radix_pass(array, len(array), m, 26)

    return array


def trainer(wordlist, word, marker):
    """
    identify the possible word matches from the given word list
    time complexity: O(N*M), N is the size of the wordlist array, M is the length of the word
    function used: N * normalize_word() -> O(N*M); N * check_position() -> O(N*M); radix sort: O(N*M)
    hence total is O(N*M+N*M+N*M+N*M) -> O(N*M)
    :param wordlist:
    :param word:
    :param marker:
    :return: returns a list of strings containing the valid words, based on the input provided.
    """

    same_array = []  # store wordlist word which has the same characters with the guessed word.
    # normalize all words in the wordlist(sort word's characters),
    # pick the words have the same characters with the guessed target,

    for w in wordlist:
        equal = normalize_word(w, word)
        if equal:  # store it in result array.
            same_array.append(w)

    result_array = []  # store the correct position words.
    # check whether position is correct
    for r in same_array:
        result = check_position(r, word, marker)
        if result:  # if correct, append it in result array.
            result_array.append(r)

    # sort result_array.(use radix sort)
    radix_sort(result_array)

    return result_array


# -------------------------------Task 2 --------------------------------------

def check_local_maximum(array, row, col):
    """
    according to the index, check whether this number is the local maximum.
    time complexity: O(1), only compare with maximum four items.
    :param array:
    :param row:
    :param col:
    :return: if is, return index, if not, return 0.
    """
    size = len(array)

    if row < 0 or row > (size - 1) or col < 0 or col > (size - 1):  # check idx
        return 0

    idx = [[row, col - 1], [row, col + 1], [row - 1, col], [row + 1, col]]  # up, below, left and right element idx.
    for i in idx:  # check condition, prevent idx out of range.
        if i[0] < 0 or i[0] > (size - 1) or i[1] < 0 or i[1] > (size - 1):
            idx.remove(i)

    if not idx:  # if nothing in idx array, return 0
        return 0

    for i in idx:  # if check number less than any of neighbours, return 0.
        if array[row][col] < array[i[0]][i[1]]:
            return 0

    return [row, col]


def check_square(array, row, col):
    """
    check whether this 3*3 square exist local maximum. (sometimes may less than 3*3 items e.g. at the edge)
    time complexity: maximum check 9*4 items, is O(1).
    :param array:
    :param row: row idx of the center
    :param col: col idx of the center
    :return: if local exist, return its idx, if not ,return 0.
    """
    row_ary = [row, row + 1, row - 1]
    col_ary = [col, col + 1, col - 1]

    # check eight neighbours with central itself
    for r in row_ary:
        for c in col_ary:
            if check_local_maximum(array, r, c) != 0:
                return check_local_maximum(array, r, c)
    # if no local maximum, return 0.
    return 0


def local_maximum(M):
    """
    find the local maximum
    time complexity: check_local_maximum and check_square cost O(1),
    n is the size of the array.
    when n < 6, number of check point is constant, n=1,2,3, c=1; n=4,5,6 c=4;
    when n > 6, in worst case, number of check point is (9*4)*(ceil(sqrt(size)))**2, which equals to O(size) == O(n)
    :param M:  array we need to check
    :return: idx of the local maximum or none if no local maximum
    """
    size = len(M)

    # when number of check position less than n.
    if size == 1:  # n = 1, return [0,0]
        return [0, 0]
    elif size == 2 or size == 3:  # if n==2 or 3, only need to check position [1,1].
        if check_square(M, 1, 1) != 0:  # if find any local maximum, return idx.
            return check_square(M, 1, 1)
    elif size == 4 or size == 5 or size == 6:  # if size ==4,5,6, only need to check 4 position.
        check_p = [1, 4]
        for r in check_p:
            for c in check_p:
                if check_square(M, r, c) != 0:  # if find any local maximum, return idx.
                    return check_square(M, r, c)

    else:  # when number of check position greater than n

        no_check = math.ceil(math.sqrt(size))  # sqrt of number of  3*3 squares in the n*n grid when n >6.
        check_point = []

        if size % 3 == 0 or size % 3 == 2:  # if size is number like 6,9,12 or 8, 11, 14.
            for i in range(no_check):  # generate check point idx
                check_point.append(i * 3 + 1)  # start from [1][1], move to next 3*3 square.

            for r in check_point:  # check each check point
                for c in check_point:
                    if check_square(M, r, c) != 0:  # if find any local maximum, return idx
                        return check_square(M, r, c)

        else:  # size % 3 == 1,
            # in this case, we cant check (len(check_point))**2 of check points,
            # cause idx will out of range.
            # we will check the edge case separately.
            for i in range(no_check - 1):  # generate check point idx
                check_point.append(i * 3 + 1)  # start from [1][1], move to next 3*3 square.
            check_point.append(size - 1)  # add edge idx case

            for r in check_point:  # check each check point
                for c in check_point:
                    if check_square(M, r, c) != 0:  # if find any local maximum, return idx.
                        return check_square(M, r, c)
