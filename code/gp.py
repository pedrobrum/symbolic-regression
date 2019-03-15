#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Esse arquivo possui as classes e funções utilizadas para implementar a um GP
    que resolve o problema de regressão simbólica.
"""

from random import sample, random, randint, choice
from math import sqrt, sin, cos, log, pi, e, log2
from copy import deepcopy, copy
import numpy as np


class Node(object):

    def __init__(self, value):
        if value == "rand":
            self.value = str(choice([0.1, 0.2, 0.3, 0.4, 0.5]))
        else:
            self.value = value

        self.left = None
        self.right = None
        self.parent = None


class Tree(object):

    def __init__(self, primitives, set_dict, method, depth):
        self.root = None
        self.primitives = primitives
        self.set_dict = set_dict
        self.depth = depth
        self.size = 2 ** (self.depth + 1) - 1
        self.last_level = 2 ** self.depth - 1
        if method == 'full':
            self.root = self._full(self.root, self.size, self.last_level, 0)
        elif method == 'grow':
            self.root = self._grow(self.root, self.size, self.last_level, 0)


    def _full(self, node, s, m, n, parent = None):

        if(m == 0):
            node = Node(choice(self.set_dict["terminals"]))
            node.parent = parent
        elif(n < m):
            node = Node(choice(self.set_dict["functions"]))
            node.parent = parent
            node.left = self._full(node.left, s, m, 2*n + 1, node)
            node.right = self._full(node.right, s, m, 2*n + 2, node)
        elif(n < s):
            node = Node(choice(self.set_dict["terminals"]))
            node.parent = parent

        return node


    def _grow(self, node, s, m, n, parent = None):

        if n == 0:
            if self.depth >= 1:
                prim = choice(self.set_dict["primitives"])
                node = Node(prim)
                node.parent = parent
                node.left = self._grow(node.left, s, m, 2*n + 1, node)
                node.right = self._grow(node.right, s, m, 2*n + 2, node)
            elif self.depth == 0:
                prim = choice(self.set_dict["terminals"])
                node = Node(prim)
        elif (n < m):
            if parent.value not in self.set_dict["functions"]:
                node = None
            else:
                prim = choice(self.set_dict["primitives"])
                node = Node(prim)
                node.parent = parent
                node.left = self._grow(node.left, s, m, 2*n + 1, node)
                node.right = self._grow(node.right, s, m, 2*n + 2, node)
        elif (n < s):
            if parent.value not in self.set_dict["functions"]:
                node = None
            else:
                node = Node(choice(self.set_dict["terminals"]))
                node.parent = parent

        return node


    def build_program(self, node):
        eq = ""
        if node != None:
            eq = node.value
            left = self.build_program(node.left)
            right = self.build_program(node.right)
            eq = "(" + left + eq + right + ")"

        return eq


    def nodes(self, node, i, n):

        if i > n:
            return []
        elif node == None:
            return []

        return [node] + self.nodes(node.left, 2*i + 1, n) + self.nodes(node.right, 2*i + 2, n)


    def random_node(self, n = 2):

        all_nodes = self.nodes(self.root, 0, 2*n + 2)
        i = randint(0, len(all_nodes) - 1)

        return all_nodes[i]


def subtree_crossover(population, n, data, elitist):

    first, first_score = tournament(population, n, data)
    second, second_score = tournament(population, n, data)

    first_parent = deepcopy(first)
    second_parent = deepcopy(second)

    cross_pt1 = first_parent.random_node()
    cross_pt2 = second_parent.random_node()

    new_individual = _crossover(first_parent, cross_pt1, cross_pt2)
    new_score = fitness(new_individual, data)
    mean_score = (first_score + second_score)/2.0

    x = 0
    if new_score < (first_score + second_score)/2.0:
        x = 1

    if (elitist == 0) or (new_score < first_score and new_score < second_score):
        return new_individual, x
    elif first_score < second_score:
        return first, x
    else:
        return second, x


def subtree_mutation(population, n, data, max_depth, elitist):

    individual, score = tournament(population, n, data)
    p = individual.primitives
    s = individual.set_dict

    init_options = ['full', 'grow']
    first_parent = deepcopy(individual)
    second_parent = Tree(p, s, choice(init_options), randint(1, max_depth))
    new_individual = _crossover(first_parent, first_parent.random_node(), second_parent.root)
    new_score = fitness(new_individual, data)

    if (elitist == 0) or new_score < score:
        return new_individual
    else:
        return individual


def reproduction(population, n, data):

    individual, score = tournament(population, n, data)
    return deepcopy(individual)


def fitness(tree, dataset):

    prog = tree.build_program(tree.root)
    variables = tree.set_dict["variables"]
    m = len(variables)
    yi = []
    y_eval = []

    for item in dataset:
        for i in range(m):
            vars()[variables[i]] = item[i]
        try:
            result = eval(prog)
        except:
            result = 0.0
        yi.append(item[-1])
        y_eval.append(result)

    yi = np.array(yi)
    y_eval = np.array(y_eval)

    nrmse = 0.0
    nrmse = sqrt(np.sum((yi - y_eval) ** 2))
    nrmse = nrmse/sqrt(np.sum((yi - np.mean(yi)) ** 2))

    return nrmse


def tournament(population, n, data):

    pop_sample = sample(population, n)
    best = None
    best_score = None
    for item in pop_sample:
        score = fitness(item, data)
        if (best_score == None) or (score < best_score):
            best = item
            best_score = score

    if best == None:
        return tournament(population, n, data)

    return best, best_score


def _crossover(tree1, cross_pt1, cross_pt2):

    parent = cross_pt1.parent
    aux = cross_pt2
    if parent == None:
        tree1.root = aux
        aux.parent = None
    elif parent.left == cross_pt1:
        parent.left = aux
        aux.parent = parent
    elif parent.right == cross_pt1:
        parent.right = aux
        aux.parent = parent

    return tree1


def evaluation(population, data):

    best = None
    best_score = None
    scores = []
    individuals = {}

    for individual in population:
        score = fitness(individual, data)
        scores.append(score)
        if score in individuals:
            individuals[score] += 1
        else:
            individuals[score] = 1
        if (best_score == None) or (score < best_score):
            best = individual
            best_score = score

    worst_score = np.max(scores)
    mean_score = np.mean(scores)

    repeated = 0
    for score in scores:
        if individuals[score] != 1:
            repeated += 1

    return best, best_score, worst_score, mean_score, repeated








