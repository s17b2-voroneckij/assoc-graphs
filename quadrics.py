#import tensorflow as tf
import matplotlib
#%matplotlib inline
import numpy as np
from random import seed, randint, random
from copy import deepcopy
#from keras import backend as K


class Graph:
    def __init__(self, V, E, edges):
        self.V = V
        self.E = E
        for i in range(len(edges)):
            if type(edges[i]) == type(tuple()):
                edges[i] = list(edges[i])
        self.edges = edges
        self.bridges = set()  # содержит номера мостов графа, они же isthmus
        self.update_lists_of_edges()

    def update_lists_of_edges(self):
        # функция генерирует массив, где каждой вершине соответствует список рёбер, выходящих из неё
        self.g = [[] for i in range(self.V)]
        for i in range(len(self.edges)):
            edge = self.edges[i]
            self.g[edge[0]].append((edge[1], edge[2], i))  # вершина, вес, номер
            #self.g[edge[1]].append((edge[0], edge[2], i)) - теперь у нас ориентированный граф
        #self.init_bridges(0)

    def dfs(self, v, visited, forbidden):
        # функция реализует алгоритм поиска в ширину, не проходящему по рёбрам из forbidden
        if visited[v]:
            return
        visited[v] = True
        for edge in self.g[v]:
            if edge[2] not in forbidden:
                self.dfs(edge[0], visited, forbidden)
    
    def gamma(self, s, i):
        # возвращает значения gamma_{i, 0}, gamma_{i, 1} и так далее
        visited = [False] * self.V
        self.dfs(s, visited, {i})
        gamma_i = [0] * len(self.edges)
        for j in range(len(self.edges)):
            if i == j:
                continue
            gamma_i[j] = visited[self.edges[j][0]]
        return gamma_i
    
    def betti(self, s, forbidden):
        # считает первое число Бетти для графа без рёбер из forbidden.
        # Возвращает (число Бетти, число изолированных вершин, число компонент связности )
        comps = 0
        visited = [False] * self.V
        isolated = 0
        for j in range(self.V):
            incident = 0
            for edge in self.g[j]:
                if edge[2] not in forbidden:
                    incident += 1
            if incident == 0:
                isolated += 1
            if not visited[j]:
                self.dfs(j, visited, forbidden)
                comps += 1  # вообще говоря, условие велит нам удалять изолированные вершины, но от этого число Бетти не изменяется
        return self.E - len(forbidden) - self.V + comps, isolated, comps - isolated

    def init_bridges(self, s):
        self.bridges = set()
        betti_G = self.betti(s, {})
        for i in range(self.E):
            if self.betti(s, {i})[1:] != betti_G[1:]:
                self.bridges.add(i)
    
    def delta(self, s, i, j):
        return i not in self.bridges and j not in self.bridges

    def our_quadratic(self, s):  # self.edges - список рёбер, self.V - количество вершин, s - источник. Рёбра хранятся в формате (вершина, вершина, вес)
        self.E = len(self.edges)
        our = [[0] * self.E for i in range(self.E)]
        self.init_bridges(s)
        #  нулевая сумма
        for i in range(self.E):
            gamma_i = self.gamma(s, i)
            for j in range(self.E):
                if i == j:
                    continue
                our[i][j] -= 0.5 * gamma_i[j] * (2 ** self.betti(s, {i})[0])
    
        # первая сумма
        betti_RG = self.betti(s, set())
        temp = 2 ** (betti_RG[0] - 1)
        for num in self.bridges:
            edge = self.edges[num]
            if len(self.g[edge[0]]) > 1 and len(self.g[edge[1]]) > 1:
                our[num][num] -= temp
        # вторая сумма
        for i in range(self.E):
            for j in range(i + 1, self.E):
                betti_G = self.betti(s, {i, j})
                if betti_G[2] == 1:
                    m = betti_G[1]
                    our[i][j] += (4 - m) * 2 ** (betti_G[0])
        # третья сумма
        for i in range(self.E):
            for j in range(i + 1, self.E):
                betti_G = self.betti(s, {i, j})
                if betti_G[2] == 2:
                    our[i][j] += 2 ** (betti_G[0] + self.delta(s, i, j))
    
        # четвёртая сумма
        for i in range(self.E):
            for j in range(i + 1, self.E):
                e1 = self.edges[i]
                e2 = self.edges[j]
                if (e1[0] == e2[0] and len(self.g[e1[0]]) == 2) or (e1[0] == e2[1] and len(self.g[e1[0]]) == 2) or (e1[1] == e2[0] and len(self.g[e1[1]]) == 2) or (e1[1] == e2[1] and len(self.g[e1[1]]) == 2):
                    our[i][j] -= 2 ** self.betti(s, {i, j})[0]

        for i in range(self.E):
            for j in range(i + 1, self.E):
                s = our[i][j] + our[j][i]
                our[i][j] = s / 2
                our[j][i] = s / 2
        return np.array(our)
    
    def incidence_matrix(self, min_weight=0):
        self.E = len(self.edges)
        res = [[0] * self.E for i in range(self.V)]
        for i in range(len(self.edges)):
            if self.edges[i][2] < min_weight:
                continue
            res[self.edges[i][0]][i] = 1
            res[self.edges[i][1]][i] = 1
        return res
    
    def adjacency_matrix(self, min_weight=0):
        res = [[0] * self.V for i in range(self.V)]
        for edge in self.edges:
            if edge[2] < min_weight:
                continue
            res[edge[0]][edge[1]] += 1
            res[edge[1]][edge[0]] += 1
        return res

    def comp(self, i, pos):
        return (i >> pos) & 1

    def cut_matrix(self):
        self.E = len(self.edges)
        res = [[0] * self.E for i in range(2 ** self.V)]
        for i in range(2 ** self.V):
            for j in range(self.E):
                edge = self.edges[j]
                if self.comp(i, edge[0]) != self.comp(i, edge[1]):
                    res[i][j] = 1
        return res

    def laplace_matrix(self, min_weight=0):
        self.E = len(self.edges)
        res = np.zeros((self.V, self.V), dtype=np.float32)
        self.update_lists_of_edges()
        for edge in self.edges:
            if edge[2] < min_weight:
                continue
            res[edge[0]][edge[0]] += 1
            res[edge[1]][edge[1]] += 1
            res[edge[0]][edge[1]] -= 1
            res[edge[1]][edge[0]] -= 1
        return res
    
    def laplace_matrix_weighted(self):
        self.E = len(self.edges)
        res = np.zeros((self.V, self.V), dtype=np.float32)
        self.update_lists_of_edges()
        for edge in self.edges:
            res[edge[0]][edge[0]] += edge[2]
            res[edge[1]][edge[1]] += edge[2]
            res[edge[0]][edge[1]] -= edge[2]
            res[edge[1]][edge[0]] -= edge[2]
        return res  
    
    def add_graph(self, other, v1, v2):
        for _edge in other.edges:
            edge = _edge[:]
            if edge[0] == v2:
                edge[0] = v1
            else:
                edge[0] += self.V
                if edge[0] > v2 + self.V:
                    edge[0] -= 1
            if edge[1] == v2:
                edge[1] = v1
            else:
                edge[1] += self.V
                if edge[1] > v2 + self.V:
                    edge[1] -= 1
            self.edges.append(edge)
        self.V += other.V - 1
        self.E = len(self.edges)
        self.update_lists_of_edges()    


def eigenvalues(a):
    a = np.array(a)
    return list(np.linalg.eig(a)[0])


def full_graph(V):
    return Graph(V, V * (V - 1) // 2, [[i, j, 0] for i in range(V) for j in range(i + 1, V)])
    
    
def rand_graph(V, E, randomizer=lambda: random()):
    edges = []
    for i in range(E):
        edges.append([randint(0, V - 1), randint(0, V - 1), randomizer()])
    return Graph(V, E, edges)


def cycle(V):
    return Graph(V, V, [[i, i + 1, 0] for i in range(V - 1)] + [[V - 1, 0, 0]])

    
def symm_sum(a):
    a = np.array(a)
    return (a + a.T) / 2
    
    
def symm_mul_TA(a):
    a = np.array(a)
    return a.T.dot(a)
    
    
def symm_mul_AT(a):
    a = np.array(a)
    return a.dot(a.T)
    
    
def oldest_eigenvalue(our):
    ei = sorted(eigenvalues(our))
    return max(-ei[0], ei[-1])


def print_matrix(m):
    for elem in m:
        print('\t'.join(map(str, elem)))


def get_values(m):
    values = eigenvalues(m)
    for i in range(len(values)):
        values[i] = round(values[i], 13)
        if abs(abs(values[i]) - abs(float(values[i]))) < 1e-4:
            values[i] = float(values[i])
    return sorted(values)


def print_values(m):
    print(get_values(m))


def linear_graph(V):
    return Graph(V, V - 1, [[i, i + 1, 0] for i in range(V - 1)])


def star(V):
    return Graph(V, V - 1, [[0, i, 0] for i in range(1, V)])


def matr_to_tex(a):
    print('\\begin{pmatrix}')
    for line in a:
        for elem in line:
            print(elem, '&', end=' ')
        print('\\\\')
    print('\\end{pmatrix}')


class Poly:
    def __init__(self, a=0, b=0):
        self.x = b
        self.a = a

    def __add__(self, other):
        self.x += other.x
        self.a += other.a

    def __sub__(self, other):
        self.x -= other.x
        self.a -= other.a

    def __str__(self):
        res = ''
        if self.x != 0:
            res += str(self.x) + 'x'
            if self.a >= 0:
                res += '+'
        res += str(self.a)
        return res


def magic_process(a):
    b = deepcopy(a)
    b += np.diag([0.5] * len(a))
    k = len(a) - 1
    for i in range(1, k):
        b[i] += b[0]
    b[k] -= b[0]
    for i in range(1, k):
        b[0] -= b[i]
    b = b.T
    for i in range(1, k):
        b[0] -= b[i] * 3
    b[k] += b[0]
    for i in range(1, k):
        b[k] -= 4 * b[i]
    return b



def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

def eig_fast(matr):
    s = reset_tf_session()
    input_matr = tf.placeholder(tf.float16, shape=(None, None))
    eig = tf.linalg.eigvalsh(input_matr)
    e = s.run([eig], {input_matr : matr})
    return e[0]

def graph_from_file(file, limit=0):
    word_id = dict()
    edges = []
    s = file.readline()
    i = 0
    while s:
        w1, w2, weight = s.split(',')
        weight = float(weight)
        if w1 not in word_id:
            word_id[w1] = len(word_id)
        if w2 not in word_id:
            word_id[w2] = len(word_id)
        id1 = word_id[w1]
        id2 = word_id[w2]
        edges.append([id1, id2, weight])
        s = file.readline()
        i += 1
        if i == limit:
            break
    return Graph(len(word_id), len(edges), edges), word_id

def wasserstein_distance1(a, b):
    Len = min(len(a), len(b))
    return ss.wasserstein_distance(a[:Len], b[:Len])

def wasserstein_distance2(a, b):
    if len(a) > len(b):
        a, b = b, a
    return ss.wasserstein_distance(a + [0] * (len(b) - len(a)), b)

def save(vector, string):
    f = open(string, 'w')
    print(vector, file=f)
    f.close()

if __name__ == "__main__":
    lg1 = star(6)
    lg2 = star(5)
    lg1.add_graph(lg2, 1, 1)
    lg1.our_quadratic(1)
