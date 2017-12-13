

from igraph import Graph
from igraph import plot
from igraph import layout
import sys
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

def orderTuple(a,b):
	return (a, b) if a < b else (b,a)
	
	

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
    plot(g, **visual_style)


def createHeatMapFromGraphCommunities(title, matrix):
	
	fig = plt.figure()
	
	ax = fig.add_subplot(111)

	ax.set_title(title)

	plotly_fig = tls.mpl_to_plotly( fig )

	trace = dict(z=matrix, type="heatmap", zmin=matrix.min(), zmax=matrix.max(),colorscale='Jet', showscale=False)

	plotly_fig['data'] = [trace]

	plotly_fig['layout']['xaxis'].update({'autorange':True})
	
	plotly_fig['layout']['yaxis'].update({'autorange':True})

	plot(plotly_fig, filename=title)
	
def createHeatMapFromGraph(title, matrix):
				
	fig = plt.figure()
	
	ax = fig.add_subplot(111)

	ax.set_title(title)

	plotly_fig = tls.mpl_to_plotly( fig )

	trace = dict(z=matrix, type="heatmap", zmin=matrix.min(), zmax=matrix.max(),colorscale='Greys')

	plotly_fig['data'] = [trace]

	plotly_fig['layout']['xaxis'].update({'autorange':True})
	
	plotly_fig['layout']['yaxis'].update({'autorange':True})

	plot(plotly_fig, filename=title)
	
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
	plot(fig, filename=title)

def parseFileAndLoadGraph(input_file):
	actualFunction = defaultFunction
	with open(input_file, "r") as inFile:
		for line in inFile:
			line = line.replace('\n', '')
			if line in options:
				actualFunction = options[line]
				continue
			actualFunction(line)

def run_generation(inputFile, algorithm, nodes):
	
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

	for e in graph.es:
		print(e.source, e.target, '=', e['weight'])

	actualApplication = algorithm
	
	createHeatMapFromGraph('Communication matrix ' + actualApplication + ' (' + nodes + ' nodes)', matrixToHeatMap.T)
	
	print("First method\n")

	clusterU = graph.community_fastgreedy(weights='weight').as_clustering()
	
	print(clusterU.q)
			
	createHeatMapFromGraphCommunities('Communities in application ' +
									  actualApplication + ' using fast greedy algorithm (' + nodes + ' nodes)',
									  createMatrixFromCluster(clusterU))
	communitiesSize = []
	
	for i in range(clusterU.__len__()):
		#print(clusterU.subgraph(i))	
		communitiesSize.append(clusterU.subgraph(i).vs.__len__())
	
	createGraphCommunitiesSize(communitiesSize, 'fast greedy', actualApplication, nodes)
	
run_generation(sys.argv[1], sys.argv[2], sys.argv[3])
