import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import time
from collections import namedtuple

#Reference made to this website: 
#	https://salzis.wordpress.com/2014/06/28/kd-tree-and-nearest-neighbor-nn-search-2d-case/
#	as well as Wikipedia

Node = namedtuple('Node','split left right')
k = 2
line_width = [4., 3.5, 3., 2.5, 2., 1.5, 1., .5, 0.3]
depth_dict = {}

def main():
	testData = np.loadtxt('data/data2-test.dat',dtype=np.object,comments='#',\
		delimiter=None)

	trainData = np.loadtxt('data/data2-train.dat',dtype=np.object,comments='#',\
		delimiter=None)
	kNN(testData,trainData,k)	# uses SciPy to evaluate the run-time with nearest-neighbours
	
	test_data_points = point_list(testData)
	tree = kd_tree_build(test_data_points,0)
	
	draw_plot(tree, test_data_points)
	# process_tree(tree)
	# print tree


def kNN(testData, trainData, k):
	x_train = np.asarray(column(trainData,0))
	x_test = np.asarray(column(testData,0))

	y_train = np.asarray(column(trainData,1))
	y_test = np.asarray(column(testData,1))

	labels_train = np.asarray(column(trainData,2))
	labels_test = np.asarray(column(testData,2))

	z_result = []

	#create kD Tree using SciPy
	tree = sp.KDTree(list(zip(x_train.ravel(), y_train.ravel(), labels_train.ravel())))

	corr_rate = 0.
	total_sum = 0.
	start = time.time()
	for i in range(x_test.shape[0]):
		x_i = x_test[i]
		y_i = y_test[i]
		z_i = labels_test[i]
		target = np.array([x_i,y_i,z_i])
		distance, index = tree.query(target)
		result = labels_train[index]
		z_result.append(result)
		if labels_test[i] == result :
			corr_rate += 1.0
		total_sum += 1.0
	
	end = time.time()
	print z_result
	elapsed = end-start
	print "--------------------------RESULTS---------------------------"
	print "Time elapsed: %.3f seconds"%elapsed
	print "Correct classification rate:  %.2f%%" % (corr_rate / total_sum * 100.0)
	print "------------------------------------------------------------"
	plot_data(x_test,y_test,z_result,"Classification results", "out/kdtree/classification.png")
	plot_data(x_test,y_test,labels_test,"Test data", "out/kdtree/test.png")





def draw_plot(tree, test_data_points):
# n = 50        # number of points
	min_x_val = int(min(np.minimum(column(test_data_points,0), column(test_data_points,0))) )  # minimal coordinate value
	max_x_val = int(max(np.maximum(column(test_data_points,0), column(test_data_points,0))) )  # maximal coordinate value

	min_y_val = int(min(np.minimum(column(test_data_points,1), column(test_data_points,1))) )  # minimal coordinate value
	max_y_val = int(max(np.maximum(column(test_data_points,1), column(test_data_points,1))) )  # maximal coordinate value

 	delta = 2

	plt.figure("K-d Tree")
	plt.grid(b=True, which='major', color='0.75', linestyle='--')
	plt.axis( [min_x_val-delta, max_x_val+delta, min_y_val-delta, max_y_val+delta] )
 
	# draw the tree
	plot_tree(tree, min_x_val-delta, max_x_val+delta, min_y_val-delta, max_y_val+delta, None, None)
 
	plt.title('K-D Tree')
        plt.savefig("out/kdtree/kdtree.png", bbox_inches="tight", pad_inches=0)
	plt.show()
	plt.close()	

def kd_tree_build(data_points,depth,mode='alternate',split='median'):
	if data_points.shape[0] == 0 : 
		return
	x_i = None
	_data_points_left = None
	_data_points_left = None
	if mode == 'alternate':
		d = depth%k #alternate between x and y dimension for splitting
	#find median
	elif mode == 'variance':

		if np.var(data_points[:,0]) >= np.var(data_points[:,1]):
			d = 0
		else:
			d = 1

		depth_dict[depth] = d

	if split=='median':
		sorted_points = data_points[np.argsort(data_points[:,d])]
	
		x_i = sorted_points[len(sorted_points)/2]
		_data_points_right = sorted_points[len(sorted_points)/2+1:\
		len(sorted_points)]
		_data_points_left = sorted_points[0:len(sorted_points)/2]
	elif split=='centre':
		x_i = data_points[len(data_points)/2]
		_data_points_right = data_points[len(data_points)/2+1:\
		len(data_points)]
		_data_points_left = data_points[0:len(data_points)/2]
	
	return Node(x_i, kd_tree_build(_data_points_left,depth+1),\
		kd_tree_build(_data_points_right,depth+1))
	

def plot_tree(tree, min_x, max_x, min_y,max_y,prev_node, branch,depth=0,mode='alternate'):
	cur_node = tree.split
	left_branch = tree.left 
	right_branch = tree.right

	if depth > len(line_width)-1:
		ln_width = line_width[len(line_width)-1]
	else:
		ln_width = line_width[depth]

	if mode == 'alternate':
		axis = depth % k
	elif mode == 'variance':
		axis = depth_dict[depth]

	if axis == 0 :
		if branch is not None and prev_node is not None:
			if branch:
				max_y = prev_node[1]
			else:
				min_y = prev_node[1]
		# if cur_node is not None:
		plt.plot([cur_node[0], cur_node[0]], [min_y,max_y], \
			linestyle='-', color='blue', linewidth=ln_width)
	elif axis == 1 :
		if branch is not None and prev_node is not None:
			if branch: 
				max_x = prev_node[0]
			else:
				min_x = prev_node[0]
		# if cur_node is not None:
		plt.plot([min_x,max_x],[cur_node[1],cur_node[1]],linestyle='-', color='red', linewidth=ln_width)
	
	if cur_node is not None:
		plt.plot(cur_node[0],cur_node[1],'ko')

	if left_branch is not None:
		plot_tree(left_branch, min_x, max_x, min_y, max_y, cur_node, True, depth+1)
	if right_branch is not None:
		plot_tree(right_branch, min_x, max_x, min_y, max_y, cur_node, False, depth+1)
	
	

def column(matrix, i):
	return [float(row[i]) for row in matrix]

def point_list(matrix):
	return np.asarray([(float(row[0]),float(row[1])) for row in matrix])

def plot_data(x,y,z,title, path):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(title)
	ax.scatter(x,y,c=z,s=100)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
	plt.show();

if __name__ == "__main__":
	main()
	
