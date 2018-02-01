

from __future__ import division
from igraph import Graph
from igraph import plot
from igraph import layout
import igraph as igraph 
import networkx as nx
import sys
import shutil
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
tls.set_credentials_file(username='casaman', api_key='XlEZ7WzT12zDXa0cHXhf')
startLineToParse = '------- Start Communication Matrix -------------'
endLineToParse = '------- End Communication Matrix -------------'

graph = None
vertexAdded = {}
edges = {}


def convertGraph(graphInput):
    netXGraph = nx.Graph()
    names = list(map(int, graphInput.vs['name']))	
    netXGraph.add_nodes_from(names)
    netXGraph.add_edges_from(graphInput.get_edgelist())	
	#print(list(netXGraph.edges()))
    return netXGraph

def plotGraph(graph, outputDir):
	""" Plot the graph structure on an image"""
	fileName = outputDir + "/graph.png"
	layout = graph.layout_lgl()
	igraph.plot(graph,fileName, layout=layout)

def orderTuple(a,b):
	return (a, b) if a < b else (b,a)

def removeOutliers(distribution):
	index = []
	thresholdMin = np.percentile(distribution, 5)
	thresholdMax = np.percentile(distribution, 95)
	for i in range(len(distribution)):
		if distribution[i] < thresholdMin:
			index.append(i)
		elif distribution[i] > thresholdMax:
			index.append(i)
		else:
			continue
	distribution = np.delete(distribution, index)
	return distribution

def calculateCPL(graphInput):
	nodes = nx.nodes(graphInput)
	averages = []
	for node in nodes:
		path_lengths = nx.single_source_shortest_path_length(graphInput, node)
		average = sum(path_lengths.values())/len(path_lengths)
		averages.append(average)
	median = np.median(averages)
	return median


def calculateScaleFreeness(graphInput):
	graphNodes = list(graphInput.nodes)
	graphDegrees = list(graphInput.degree(graphNodes))
	degrees = []
	for x in graphDegrees:
		degrees.append(x[1])
	#print (degrees)
	degrees = np.sort(degrees)
	#print (degrees)
	adjacencyMat = nx.adjacency_matrix(graphInput)
	scaleResult = 0
	for i in range(1,len(graphDegrees)):
		for j in range(i+1,len(graphDegrees)):
			#print (str(degrees[i])+"  "+str(degrees[j])+" "+str(adjacencyMat[i,j]))
			scaleResult += (degrees[i]*degrees[j]*adjacencyMat[i,j])
	return scaleResult
		

def getConnStats(graphInput, fileName):
	""" Function to extract different connectivity stats using networkX algorithms """
	graphNodes = list(graphInput.nodes)
	graphDegrees = list(graphInput.degree(graphNodes))
	degrees = []
	for x in graphDegrees:
		degrees.append(x[1])
	averageDegree = np.mean(degrees)
	stdDeviation = np.std(degrees)
	degreeCorrelation = nx.degree_assortativity_coefficient(graphInput)
	scaleFreeness = calculateScaleFreeness(graphInput)
	fileName.write("--------------- Connectivity Statistics----------------\n")   
	fileName.write("Average Degree:  " + str(averageDegree) + "\n") 
	fileName.write("Standard Deviation:  " + str(stdDeviation) + "\n")
	fileName.write("Degree Correlation:  " + str(degreeCorrelation) + "\n")
	fileName.write("Scale Freeness:  " + str(scaleFreeness) + "\n")
	fileName.write("----------------Connectivity Statistics end -------------------\n")


def getDistStats(graphInput, fileName):
	""" Function to extract different distance stats using networkX algorithms """
	avPathLength = nx.average_shortest_path_length(graphInput)
	graphDiameter = nx.diameter(graphInput)
	eccentricityList = nx.eccentricity(graphInput)
	averageEccentricity = np.mean(eccentricityList.values())
	characteristicPathLength = calculateCPL(graphInput)
	fileName.write("--------------- Distance Statistics----------------\n")   
	fileName.write("Average Path Length:  " + str(avPathLength) + "\n") 
	fileName.write("Characteristic Path Length:  " + str(characteristicPathLength) + "\n")
	fileName.write("Graph Diameter:  " + str(graphDiameter) + "\n")
	fileName.write("Average Eccentricity:  " + str(averageEccentricity) + "\n")
	fileName.write("----------------Distance Statistics end -------------------\n")

def getClustStats(graphInput, fileName):
	""" Function to extract different clustering stats using networkX algorithms """
	triangles = nx.triangles(graphInput)
	averageTriangles = np.mean(triangles.values())
	stdTriangles = np.std(triangles.values())
	transitivity = nx.transitivity(graphInput)
	aveClustCoeff = nx.average_clustering(graphInput)
	fileName.write("--------------- Clustering Statistics----------------\n")   
	fileName.write("Average triangles:  " + str(averageTriangles) + "\n") 
	fileName.write("Standard Deviation triangles:  " + str(stdTriangles) + "\n") 
	fileName.write("Graph Transitivity:  " + str(transitivity) + "\n")
	fileName.write("Average Clustering Coefficient:  " + str(aveClustCoeff) + "\n")
	fileName.write("----------------Clustering Statistics end -------------------\n")

def getCentralStats(graphInput,fileName):
	""" Function to extract different centrality stats using networkX algorithms """
	aveDegreeCentrality = np.mean(nx.degree_centrality(graphInput).values())
	aveClosenessCentrality = np.mean(nx.closeness_centrality(graphInput).values())
	aveBetweennessCentrality = np.mean(nx.betweenness_centrality(graphInput).values())
	fileName.write("--------------- Centrality Statistics----------------\n")   
	fileName.write("Average Degree Cenetrality:  " + str(aveDegreeCentrality) + "\n") 
	fileName.write("Average Closeness Centrality:  " + str(aveClosenessCentrality) + "\n")
	fileName.write("Average Betweenness Centrality:  " + str(aveBetweennessCentrality) + "\n")
	fileName.write("----------------Centrality Statistics end -------------------\n")


def showGraphInfo(graphInput,fileName):
    """ Show information on a given Graph """
    fileName.write("--------------- Graph Information----------------\n")   
    for e in graphInput.es:
        fileName.write(str(e.source)+" , "+str(e.target)+" = "+str(e['weight'])+"\n")     
    fileName.write("----------------Graph info end -------------------\n")



def getHist(graphInput, outputDir):
        """ Creates a png visualization of the vertex degree distribution """
        hist = graphInput.degree()
        print ("-------------Original count------------------")
        
        size = len(hist)
        print (hist)
        print (len(hist))
        print ("-------------Outliers count------------------")
        hist = list(removeOutliers(hist))
        print (hist)
        print (len(hist))
        print ("-------------Histogram count------------------")
        bins = np.arange(1, np.max(hist)+2)	
        weightsNumpy = np.ones_like((hist))/float(len(hist))
        histogram, bin_edges = np.histogram(hist, bins=bins)
        pdf = histogram/size
        print(pdf)
	#print (bin_edges)
	#print (len(pdf))
	#print (len(bins))
        print ("-------------Saving PDF------------------")
        xaxis = range(1,len(pdf)+1)
        plt.bar(xaxis, pdf)	
        output_file = outputDir + "/PDF.png"
        plt.savefig(output_file, bbox_inches='tight')
        print ("-------------Preparing CDF------------------")
        cdf = np.cumsum(pdf)
        print (cdf)
        plt.bar(xaxis, cdf)
        output_file = outputDir + "/CDF.png"
        plt.savefig(output_file, bbox_inches='tight')


	
def obtainGraphStats(graphIn, outputDir):
	""" Obtain various graph stats in a file"""
	print ("\r Analyzing graph \n")
	outFileName = outputDir + "/graphStats.txt"
	outputStats = open(outFileName,'w')
	showGraphInfo(graphIn, outputStats)
	getHist(graphIn, outputDir)
	netXGraph = convertGraph(graphIn)
	getConnStats(netXGraph, outputStats)
	getDistStats(netXGraph, outputStats)
	getClustStats(netXGraph, outputStats)
	getCentralStats(netXGraph, outputStats)
	calculateScaleFreeness(netXGraph)
	outputStats.close()


def loadInGraph(line):	
	global graph
	global vertexAdded
	global edges
	
	if not line:
		return
		
	if 'Statistics of rank' in line:
		return
	if 'Source	Target	Total Amount	Total Count' == line:
		return
	
	values = line.split('\t')
	
	assert len(values) == 4, 'The line does not have the number of fields required'
	
	if values[0] == values[1]:
		return 
		
	if values[0] not in vertexAdded:
		graph.add_vertex(values[0])
		vertexAdded[values[0]] = 1
	if values[1] not in vertexAdded:
		graph.add_vertex(values[1])
		vertexAdded[values[1]] = 1
	
	weight = float(values[2])
	
	if weight == 0.0:
		return
	
	originalEdge = (int(values[0]), int(values[1]))
			
	edge = orderTuple(int(values[0]), int(values[1]))
	
	if edge not in edges:
		edges[edge] = {'weight': 0.0}
	
	edges[edge]['weight'] += weight
	edges[edge][originalEdge] = math.log(weight)
	
	 



from random import randint

def _plot(g, membership=None):
    if membership is not None:
        print(membership)
        gcopy = g.copy()
        edges = []
        edges_colors = []
        for edge in g.es():
            if membership[edge.tuple[0]] != membership[edge.tuple[1]]:
                edges.append(edge)
                #edges_colors.append("gray")
            else:
                edges_colors.append("black")
 
        gcopy.delete_edges(edges)
        g.delete_edges(edges)
        layout = gcopy.layout("kk")
        g.es["color"] = edges_colors
    else:
        layout = g.layout("kk")
        g.es["color"] = "gray"
    visual_style = {}
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_shape"] = "circle"
    visual_style["edge_color"] = g.es["color"]
    # visual_style["bbox"] = (4000, 2500)
    visual_style["vertex_size"] = 10
    visual_style["layout"] = layout
    visual_style["bbox"] = (1024, 768)
    visual_style["margin"] = 10
    #visual_style["edge_label"] = g.es["weight"]
    #for vertex in g.vs():
        #vertex["label"] = vertex.index
    if membership is not None:
        colors = []
        for i in range(0, max(membership)+1):
            colors.append('%06X' % randint(0, 0xFFFFFF))
        for vertex in g.vs():
            vertex["color"] = str('#') + colors[membership[vertex.index]]
        visual_style["vertex_color"] = g.vs["color"]
    plot(g,**visual_style)


def createHeatMapFromGraphCommunities(title, matrix):
	
	fig = plt.figure()
	
	ax = fig.add_subplot(111)

	ax.set_title(title)

	plotly_fig = tls.mpl_to_plotly( fig )

	trace = dict(z=matrix, type="heatmap", zmin=matrix.min(), zmax=matrix.max(),colorscale='Jet', showscale=False)

	plotly_fig['data'] = [trace]

	plotly_fig['layout']['xaxis'].update({'autorange':True})
	
	plotly_fig['layout']['yaxis'].update({'autorange':True})

	plot(plotly_fig, filename=title + ".html")
	
def createHeatMapFromGraph(title, matrix):
				
	fig = plt.figure()
	
	ax = fig.add_subplot(111)

	ax.set_title(title)

	plotly_fig = tls.mpl_to_plotly( fig )

	trace = dict(z=matrix, type="heatmap", zmin=matrix.min(), zmax=matrix.max(),colorscale='Greys')

	plotly_fig['data'] = [trace]

	plotly_fig['layout']['xaxis'].update({'autorange':True})
	
	plotly_fig['layout']['yaxis'].update({'autorange':True})

	plot(plotly_fig, filename=title + ".html")
	
def defaultFunction(line):
	return line
	
	
options = {startLineToParse: loadInGraph, endLineToParse: defaultFunction}


def createMatrixFromCluster(cluster):	
	global graph
	
	nVertex = len(graph.vs)
	
	matrixCommunity = np.zeros((nVertex, nVertex))
	
	for v in graph.vs:
		v = int(v['name'])
		matrixCommunity[v][v] = cluster.membership[v] + 1
		
	for e, values in edges.items():
		(source,  target) = e
		if(cluster.membership[source] == cluster.membership[target]):
			matrixCommunity[source][target] = cluster.membership[source] + 1
			matrixCommunity[target][source] = cluster.membership[target] + 1
	return matrixCommunity

def createGraphCommunitiesSize(communitiesSize, algorithm, app, nodes):
	title = 'Communities size detected by ' + algorithm + ' algorithm (' + nodes +' nodes)'
	trace1 = Bar(
		x=[i for i in range(len(communitiesSize))],
		y=communitiesSize,
		name='Communities size')
	data = Data([trace1])
	
	layout = Layout(
		autosize=True,
		legend=Legend(
			font=Font(
				size=12
		)
	),
	showlegend=True,
	title=title,
	xaxis=XAxis(
		autorange=True,
		type='category'
	),
    yaxis=YAxis(
			autorange=True,
			type='linear'
		)
	)
	fig = Figure(data=data, layout=layout)
	plot(fig, filename=title + ".html")

def parseFileAndLoadGraph(input_file):
	actualFunction = defaultFunction
	with open(input_file, "r") as inFile:
		for line in inFile:
			line = line.replace('\n', '')
			if line in options:
				actualFunction = options[line]
				continue
			actualFunction(line)


def run_algorithm(actualApplication, algorithm, nodes, root_dir):
	
	print("Executing algorithm %s\n" % algorithm)
	clusterU = None
	if algorithm == 'fast greedy':
		clusterU = graph.community_fastgreedy(weights='weight').as_clustering()
	else:
		if algorithm == 'leading eigen vector':
			clusterU = graph.community_leading_eigenvector(weights='weight')
		else:
			clusterU = graph.community_multilevel(weights='weight')

	print(clusterU.q)

	print("=============== Testing igraph and graph==================")
	
	outputDir = os.path.join(root_dir, '%s_graph_stats' % algorithm.replace(" ", "_"))
	os.mkdir(outputDir)
	obtainGraphStats(graph, outputDir)
	plotGraph(graph, outputDir)

	print("=============== Ending test igraph and graph==================")
	
	message = 'Communities in application %s using %s algorithm (%s nodes)' % (actualApplication, algorithm, nodes)	
	createHeatMapFromGraphCommunities(message, createMatrixFromCluster(clusterU))
	communitiesSize = []
	
	for i in range(clusterU.__len__()):
		#print(clusterU.subgraph(i))	
		communitiesSize.append(clusterU.subgraph(i).vs.__len__())
	
	createGraphCommunitiesSize(communitiesSize, algorithm, actualApplication, nodes)
	

def run_generation(inputFile, algorithm, nodes, outputDir):
	

	print("Application name %s, number of nodes %s" % (algorithm, nodes))
	
	global graph
	global edges
	
	graph = Graph()
	
	parseFileAndLoadGraph(input_file=inputFile)
	
	c = 0
	
	nVertex = len(graph.vs)
	
	
	matrixToHeatMap = np.zeros((nVertex, nVertex))
	for e, values in edges.items():
		(source,  target) = e
		graph.add_edge(source, target, weight=values['weight'])
		if (source, target) in values:
				matrixToHeatMap[source][target] = values[(source,target)]
		if (target, source) in values:
				matrixToHeatMap[target][source] = values[(target,source)]
		c += 1

	#for e in graph.es:
	#	print(e.source, e.target, '=', e['weight'])

	actualApplication = algorithm
	if os.path.exists(outputDir):
		shutil.rmtree(outputDir)
	os.mkdir(outputDir)
	createHeatMapFromGraph('Communication matrix ' + actualApplication + ' (' + nodes + ' nodes)', matrixToHeatMap.T)
	run_algorithm(actualApplication, 'multi level', nodes, outputDir)	
	run_algorithm(actualApplication, 'leading eigen vector', nodes, outputDir)
	
	
run_generation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
