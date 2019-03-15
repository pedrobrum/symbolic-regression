#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import requests
import zipfile
import io
import re
import timeit
import time
import sys
import os
import argparse
from math import sqrt, sin, cos, log, pi, e


from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gp
from random import sample, random, randint, choice, uniform
from copy import deepcopy


""" 
    Seleciona aleatoriamente uma opçao de um vetor dada as probabilidades.
"""
def random_pick(choices, probabilities):
    x = uniform(0, 1)
    cumulative_probability = 0.0

    for item, item_probability in zip(choices, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            choice = item
            break

    return choice


""" 
    Retorna o conjunto de variáveis, funções, terminais, e primitivos.
"""
def nodes(prim_dict, variables):

    functions = []
    terminals = []

    for key in prim_dict:
        if prim_dict[key] == 0:
            terminals.append(key)
        else:
            functions.append(key)

    primitives = functions + terminals
    return {"primitives":primitives, "functions":functions,
            "terminals":terminals, "variables":variables}



""" 
    Gera a população inicial utilizando o metodo ramped half-and-half.
"""
def initializes(popsize, op_set, s, max_depth):

    population = []

    x = int(popsize/(2*max_depth))
    for depth in range(1, max_depth + 1):
        for i in range(1, x + 1):
            _full = gp.Tree(op_set, s, "full", depth)
            _grow = gp.Tree(op_set, s, "grow", depth)
            population.append(_full)
            population.append(_grow)

    y = len(population)
    if y < popsize:
        for i in range(1, popsize - y + 1):
            population.append(gp.Tree(op_set, s, choice(["full", "grow"]),
                              randint(1, max_depth)))

    return population


""" 
    Essa função evolui a população de indivíduos até determinado número de
    gerações e retorna a melhor solução avaliada com o conjunto de treino, bem
    como a melhor fitness, a pior fitness e a fitness média de todas gerações.
"""
def evolve(pop, train, cross_rate, mut_rate, tourn_size, max_depth, generations,
           elitist, repetition):


    target_fitness = 0.0

    best_solution = None
    fitness_solution = None

    print("Training")

    result = []

    best, best_score, worst_score, mean_score, repeated = gp.evaluation(pop, train)
    best_solution = best
    fitness_solution = best_score

    k = 0
    while(True):

        if k >= generations:
            break

        rep_rate = 1.0 - (cross_rate + mut_rate)
        operations = ["cross", "mut", "rep"]
        probabilities = [cross_rate, mut_rate, rep_rate]

        better_solutions = 0
        worse_solutions = 0

        next_gen = []
        for i in range(len(pop)):
            op = random_pick(operations, probabilities)
            if op == "cross":
                child, better = gp.subtree_crossover(pop, tourn_size, train,
                                                     elitist)
            elif op == "mut":
                child = gp.subtree_mutation(pop, tourn_size, train,
                                                    max_depth, elitist)
            elif op == "rep":
                child = gp.reproduction(pop, tourn_size, train)
            next_gen.append(child)
            
            if elitist == 1:
                if op == "cross" and better == 1:
                    better_solutions += 1
                elif op == "cross" and better == 0:
                    worse_solutions += 1

        pop = next_gen

        print('Generation ', k)
        best, best_score, worst_score, mean_score, repeated = gp.evaluation(pop, train)

        if (best_score < fitness_solution):
            best_solution = best
            fitness_solution = best_score

        winner = deepcopy(best)
        winner = winner.build_program(winner.root)
        print("The winning program is:", winner)
        print("Best fitness:", best_score)
        print("Worst fitness:", worst_score)
        print("Fitness mean:", mean_score)
        print("Repeated individuals:", repeated)

        if elitist == 1:
            print("Individuals generated with crossover that are better than the parents:", better_solutions)
            print("Individuals generated with crossover that are worst than the parents:", worse_solutions)

        result.append((repetition, k + 1, best_score, worst_score, mean_score,
                       repeated, better_solutions, worse_solutions, winner))

        print("------------------------------------")

        pop = next_gen
        k += 1

    return best_solution, result

""" 
Lê o conjunto de treino e o conjunto de teste
"""
def read_data(filename):

    data = []
    file = open(filename, "r")
    for line in file:
        line_string = line.rstrip('\n')
        line_list = line_string.split(',')
        for i in range(len(line_list)):
            line_list[i] = float(line_list[i])
        line_tuple = tuple(line_list)
        data.append(line_tuple)
    
    file.close()
    
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--train", required=True,
                        help="train file directory.")
    parser.add_argument("-t", "--test", required=True,
                        help="test file directory.")
    parser.add_argument("-p", "--pop_size", type=int, default=50,
                        help="population size.")
    parser.add_argument("-g", "--generations", type=int, default=5,
                        help="number of generations.")    
    parser.add_argument("-d", "--max_depth", type=int, default=4,
                        help="max depth of a tree.")
    parser.add_argument("-c", "--cross_rate", type=float, default=0.90,
                        help="crossover rate.")
    parser.add_argument("-m", "--mut_rate", type=float, default=0.05,
                        help="crossover rate.") 
    parser.add_argument("-s", "--tourn_size", type=int, default=2,
                        help="tournament size.")
    parser.add_argument("-l", "--op_elitist", type=int, default=1,
                        help="elitist operators.")
    parser.add_argument("-o", "--output_file", required=True,
                        help="output csv to be written.")
    parser.add_argument("--repetitions", type=int, default=30,
                        help="number of executions.")
    parser.add_argument("-b", "--dataset", type=str, default='concrete',
                        help="dataset to evaluate the solution.")

    return parser.parse_args()


def main():

    args = parse_args()

    train_file = args.train
    test_file = args.test

    # Leitura dos conjunto de treino e teste
    train = read_data(train_file)
    test = read_data(test_file)

    # Especifica o conjunto de funções e terminais
    op_set = {"+":1, "-":1, "*":1, "/":1, "rand":0}

    dataset = args.dataset

    if dataset == "synth1" or dataset == "synth2":
        v = ['x', 'y']
    else:
        v = ['x', 'y', 'z', 'a', 'b', 'c', 'p', 'q']

    for item in v:
        op_set[item] = 0

    s = nodes(op_set, v)

    # inicializa os parâmetros
    popsize = args.pop_size
    generations = args.generations
    max_depth = args.max_depth - 1
    cross_rate = args.cross_rate
    mut_rate = args.mut_rate
    tourn_size = args.tourn_size
    elitist = args.op_elitist
    output_file = args.output_file
    times = args.repetitions

    train_results = []
    test_results = []
    for i in range(1, times + 1):
        print("%d Run" % i)
        pop = initializes(popsize, op_set, s, max_depth)
        best_solution, result = evolve(pop, train, cross_rate, mut_rate,
                                       tourn_size, max_depth, generations, elitist,
                                       i)
        train_results = train_results + result
        winner = deepcopy(best_solution)
        winner = winner.build_program(winner.root)
        fitness = gp.fitness(best_solution, test)
        print("Test data")
        print("The winning program is:", winner)
        print("Fitness:", fitness)
        test_results.append((i, fitness, winner))
        print()

    col_names = ("repetition", "generation", "best_score", "worst_score",
                 "mean_score", "repeated", "better", "worse", "winner")
    frame = pd.DataFrame.from_records(train_results, columns=col_names)
    frame.to_csv(output_file + "_train.csv", index=False, sep='\t', encoding='utf-8')

    col_names = ("repetition", "fitness", "winner")
    frame = pd.DataFrame.from_records(test_results, columns=col_names)
    frame.to_csv(output_file + "_test.csv", index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
