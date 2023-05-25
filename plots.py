import matplotlib.pyplot as plt
import numpy as np


class ConfusionMatrix:
    def __init__(self, title: str, description: str = "", xlabel: str = "Program (Prediction)",
                 ylabel: str = "Validation (True)",
                 pos_label: str = "Positive", neg_label: str = "Negative", save_location=None,
                 width=10, height=5):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.title = title
        self.description = description
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.save_location = save_location
        self.width = width
        self.height = height

    def genPlot(self, show=False):
        fig, ax = plt.subplots()
        fig.set_figheight(self.height)
        fig.set_figwidth(self.width)

        xdata = [self.tp, self.fn]
        ydata = [self.fp, self.tn]
        data = [xdata, ydata]
        ax.imshow(data, interpolation='nearest')
        plt.xticks(np.arange(0, 2), [self.pos_label, self.neg_label])
        plt.yticks(np.arange(0, 2), [self.pos_label, self.neg_label])
        # plt.title(self.description)
        plt.title(self.title)

        # Description
        props = dict(boxstyle='round', alpha=0.5)
        plt.text(1.1, 0.5, self.description,
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax.transAxes,
                 color='green', fontsize=7, bbox=props)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        # Add value texts
        for i in range(len(xdata)):
            for j in range(len(ydata)):
                plt.text(j, i, f"{data[i][j]}", ha="center", va="center", color="w")

        if self.save_location is not None:
            plt.savefig(self.save_location)
        if show:
            plt.show()
