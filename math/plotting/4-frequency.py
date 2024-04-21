#!/usr/bin/env python3
"""code to plot a histogram of student scores for a project"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ x-axis has bins every 10 units"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
