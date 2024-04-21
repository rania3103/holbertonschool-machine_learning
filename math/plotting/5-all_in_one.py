#!/usr/bin/env python3
"""code to plot all 5 previous graphs in one figure"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """The plots make a 3 x 2 grid"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    # plots
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('All in One', fontsize="x-small")
    plot1, plot2, plot3, plot4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    # plot1
    plot1.plot(y0, '-r')
    # plot2
    plot2.scatter(x1, y1, c="magenta")
    plot2.set_title("Men's Height vs Weight", fontsize="x-small")
    plot2.set_xlabel('Height (in)', fontsize="x-small")
    plot2.set_ylabel('Weight (lbs)', fontsize="x-small")
    # plot3
    plot3.plot(x2, y2)
    plot3.set_yscale('log')
    plot3.set_xlim(0, 28650)
    plot3.set_title("Exponential Decay of C-14", fontsize="x-small")
    plot3.set_xlabel('Time (years)', fontsize="x-small")
    plot3.set_ylabel('Fraction Remaining', fontsize="x-small")
    # plot4
    plot4.plot(x3, y31, '--r', label='C-14')
    plot4.plot(x3, y32, '-g', label='Ra-226')
    plot4.set_ylim(0, 1)
    plot4.set_xlim(0, 20000)
    plot4.set_title(
        "Exponential Decay of Radioactive Elements",
        fontsize="x-small")
    plot4.set_xlabel('Time (years)', fontsize="x-small")
    plot4.set_ylabel('Fraction Remaining', fontsize="x-small")
    plot4.legend(loc="upper right")
    # plot5
    plot5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    plot5.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plot5.set_ylim(0, 30)
    plot5.set_xlim(0, 100)
    plot5.set_title('Project A', fontsize="x-small")
    plot5.set_xlabel('Grades', fontsize="x-small")
    plot5.set_ylabel('Number of Students', fontsize="x-small")
    axs[2, 0].axis("off")
    axs[2, 1].axis("off")
    plot5.set_xticks(np.arange(0, 101, 10))
    fig.tight_layout()
    plt.show()
all_in_one()