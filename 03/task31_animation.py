#!/usr/bin/python
# encoding: utf8


import numpy as np
from tkinter import Tk, Canvas, Frame, BOTH, Label, LabelFrame
from kmeans import KMeans


class Animation(Frame):
    """"""

    def __init__(self, parent, w_size, kmeans):
        Frame.__init__(self, parent)
        self.parent = parent
        self.pack(fill=BOTH, expand=1)
        self.title = self.parent.title()
        self.w_size = w_size

        # canvas
        self.canvas = Canvas(self, bg="#555")
        self.parent.title("Not convergence")
        self.canvas.pack(side="top", fill="both", expand="true")

        self.kmeans = kmeans
        wait = 1000 if kmeans.method == "lloyd" else 10
        self.draw(wait)


    def draw(self, delay):
        """"""
        colors = ["#ff0", "#0ff", "#0f0", "#fff"]
        # colors = ["#f00", "#00f", "#020", "#fff"]

        self.title = self.parent.title()

        width, height = self.w_size, self.w_size
        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")

        min, max = -4, 4
        range_data = max - min
        step = float(self.w_size)/range_data

        # draw centroids
        for k, c in enumerate(self.kmeans.centroids):
            x, y = (max - c[0])*step , (max - c[1])*step
            y = self.w_size - y

            self.canvas.create_rectangle(
                    x, y, x+15, y+15,
                    fill=colors[k]
                    )

        # draw clusters
        for k in self.kmeans.clusters:
            for i in self.kmeans.clusters[k]:

                row = self.kmeans.dataset[i]
                x, y = (max - row[0])*step , (max - row[1])*step
                y = self.w_size - y

                self.canvas.create_oval(
                        x, y, x+3, y+3,
                        outline=colors[k], fill=colors[k]
                        )

        self.kmeans.next()

        if not self.kmeans.convergence:
            self.after(delay, lambda: self.draw(delay))
        else:
            text = self.kmeans.purity_function()
            self.parent.title("Convergence reached - Purity value %s" % text)
            self.canvas.update()
            path = "out/{}.eps".format(kmeans.method)
            self.canvas.postscript(file=path, colormode='color')
            self.after(delay)


dataset = np.genfromtxt(
            "data/data-clustering-1.csv", dtype=float,
            delimiter=',', 
            #usecols = (0, 1, 2, 3)
            )


# initialize centroids
dataset = dataset.T
np.random.shuffle(dataset)

#dataset = np.array([[5, 5], [10, 5], [5, 10], [10, 10]])
#dataset = np.array([[-3, -3], [-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [3, 3]])

#Animations
kmeans = KMeans(dataset, method="lloyd", centroids=False, k=3, distance="euclidean")
root = Tk()
Animation(root, 600, kmeans)
root.mainloop()

kmeans = KMeans(dataset, method="macqueen", k=3, distance="euclidean")
root = Tk()
Animation(root, 600, kmeans)
root.mainloop()

kmeans = KMeans(dataset, method="hartigan", k=3, distance="euclidean")
root = Tk()
Animation(root, 600, kmeans)
root.mainloop()
