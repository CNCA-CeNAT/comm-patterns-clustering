#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Creates a graph representation from a set of logs.
# Date: 01-20-2015

# Costa Rica National High Technology Center
# Advanced Computing Laboratory
# Diego Jimenez
# Based on the script by Menses
# Creates a graph representation from an mpiP reports and extracts their:
# heatmap, basic stats and a degree distribution
# Date: 11-16-2017

import sys
import re
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import numpy as np
from graph_tool.all import *


FAST=1

def showGraphInfo(graphInput,fileName):
    """ Show information on a given Graph """
    fileName.write("--------------- Graph Information----------------\n")
    #for v in graphInput.vertices():
    #    fileName.write(str(v)+"\n")
    for e in graphInput.edges():
        fileName.write(str(e)+"\n")
    weights = graphInput.edge_properties["weights"]
    #for i in weights:
    #	fileName.write (str(i)+"\n") 
    fileName.write("----------------Graph info end -------------------\n")

def showVertexInfo(graphInput, fileName):
    """ Show vertices' information on: in and out degree"""
    fileName.write("----------------Vertex Information ---------------\n")
    for v in graphInput.vertices():
    	fileName.write("vertex:" + str(int(v)) + " in-degree: " + str(v.in_degree())+ " out-degree: "+ str(v.out_degree())+"\n")
    fileName.write("----------------Vertex info end ------------------\n")

def getDegreeDist(graphInput, outputDir):
	""" BUGGED: Creates a png visualization of the vertex degree distribution"""
	size = len(graphInput.get_vertices())
	in_degrees = graphInput.get_in_degrees(graphInput.get_vertices())	
	out_degrees = graphInput.get_out_degrees(graphInput.get_vertices())
	total = np.append(in_degrees, out_degrees)
	bins = range(1, size+1)
	histogram = vertex_hist(graphInput, "total", bins=bins)	
	print (histogram[0])	
	#plt.bar(bins,histogram[0])
	#output_file = outputDir + "/distribution2.png"
	#plt.autoscale(enable=True, axis='both')
	#plt.savefig(output_file, bbox_inches='tight')

def getHist(graphInput, outputDir):
	""" Creates a png visualization of the vertex degree distribution """
	hist = []
	for v in graphInput.get_vertices():
		in_neighbors = graphInput.get_in_neighbors(v)
		out_neighbors = graphInput.get_out_neighbors(v)
		total_neighbors = np.append(in_neighbors, out_neighbors)
		tots = sorted(set(total_neighbors))
		degree = len(tots)
		hist.append(degree)
	print (hist)
	#print (len(hist))
	bins = range(1, len(hist)+2)	
	histogram = np.histogram(hist, bins=bins)
	print(histogram[0])
	plt.hist(hist, bins=bins)
	output_file = outputDir + "/distribution.png"
	plt.savefig(output_file, bbox_inches='tight')
    
def obtainGraphStats(outputDir):
	""" Obtain various graph stats in a file"""
	print ("\r Analyzing graph \n")
	graphInput = load_graph(outputDir+"/graph.xml.gz")
	outFileName = outputDir + "/graphStats.txt"
	outputStats = open(outFileName,'w')
	showGraphInfo(graphInput, outputStats)
	showVertexInfo(graphInput,outputStats)
	#getDegreeDist(graphInput, outputDir)
	getHist(graphInput, outputDir)
	outputStats.close()

def addGraph(graph,source,target,volume,weights):
	""" Adding volume to edge source -> target (it converts volume to MB) """

	if volume != 0.0:	
		edge = graph.edge(source,target)
		if edge is None:
			edge = graph.add_edge(source,target)
		weights[edge] = weights[edge] + volume/1000000.0

def connected(edge_x,edge_y):
	x = edge_x.target()
	y = edge_y.target()
	for v in x.out_neighbours():
		if v == y:
			return True
	return False

def drawGraph(outSizeX, outSizeY, outputDir):
	""" Creates a png visualization of the communications graph """
	graphViz = load_graph(outputDir+"/graph.xml.gz")
	print ("\n Algoritmo de layout")
	outFileName = outputDir + "/outputGraph.png"
	pos = sfdp_layout(graphViz)	
	graph_draw(graphViz,pos, vertex_text=graphViz.vertex_index, vertex_font_size=6,
           output_size=(outSizeX, outSizeY), output=outFileName)

def drawHeatmap(matrix, numberRanks, outputDir):
	labels = list(range(int(numberRanks)))
	fig, ax = plt.subplots(figsize=(15,9))
	im = ax.pcolor(matrix, cmap='viridis', edgecolor='k', lw=0.5)
	temp = fig.colorbar(im, shrink=0.6,aspect=10)
	temp.set_label("MB transmitted",horizontalalignment='right', labelpad=8)
	# Shift ticks to be at 0.5, 1.5, etc
	for axis in [ax.xaxis, ax.yaxis]:
	    axis.set(ticks=np.arange(0.5, len(labels)), ticklabels=labels)
	
	plt.tick_params(axis='both', labelsize=6)
	plt.xticks(rotation=90)
	output_file = outputDir + "/heatmap.png"
	plt.autoscale(enable=True, axis='both')
	plt.savefig(output_file, bbox_inches='tight')
	
def createGraph(matrix, size, outputDir):
	""" Creates a graph from the collection of communication logs """	
	# graph data structure
	print ("\r Creating graph \n")
	graph = Graph()
	graph.add_vertex(size)
	weights = graph.new_edge_property("float")
	graph.edge_properties["weights"] = weights
	output_file = outputDir + '/graph.xml.gz'

	rows = matrix.shape[0]
	cols = matrix.shape[1]

	for i in range(rows):
		for j in range(cols):			
			sourceRank = i
			destRank = j
			volumeEdge = matrix[i,j]
			addGraph(graph,sourceRank,destRank,volumeEdge,weights)        		

	# storing graph into a file
	graph.save(output_file, fmt='graphml')

	print ("\rDone \n")
	if(FAST):
		return

""" Creates a communication matrix from the collection mpiP log """
def readMatrix(file, rank_amount):
    size = rank_amount
    local_matrix = np.zeros((size,size))    
    with open(file) as input_data:
        # Skips text before the beginning of the block of interest:
        for line in input_data:
            if line.strip() == '------- Start Communication Matrix -------------':  # Or whatever test is needed
                break
        # Reads text until the end of the block of interest:
        for line in input_data:  # This keeps reading the file
            if line.strip() == '------- End Communication Matrix -------------':
                break
            else:
                stats_line = line.split()
                # Line is extracted (or block_of_lines.append(line), etc.)
                if (stats_line != []) and (stats_line[0].isalpha()==False):
                    stats_list = [int(i) for i in stats_line]
                    local_matrix[stats_list[0]][stats_list[1]] = stats_list[2]
    return local_matrix

def scaleMatrix(comm_matrix):
	""" Subtracts the min value from all the values in the matrix """
	""" This is done to avoid all to all communication effects """
	min_value = comm_matrix.min()
	print (min_value)
	scaled_matrix = np.subtract(comm_matrix, min_value)	
	return scaled_matrix
	
if len(sys.argv) > 3:
	fileName = sys.argv[1]
	size = int(sys.argv[2])
	outputDir = sys.argv[3]
	comm_matrix = readMatrix(fileName,size)
	res_matrix = scaleMatrix(comm_matrix)
	createGraph(res_matrix,size,outputDir)
	obtainGraphStats(outputDir)
	drawGraph(2500, 2500, outputDir)
	if size==64:
		drawHeatmap(res_matrix,size,outputDir)
	

	#show_config()
else:
	print ("ERROR, usage: %s <filena> <size> <output directory>\n" \
		"<directory>: directory containing log files\n" \
		"<size>: number of ranks in MPI execution\n" \
		"<output directory>: directory where file graph.xml.gz will be created" \
		 % sys.argv[0])

