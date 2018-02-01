import os,fnmatch
from subprocess import call
import shutil

dir_proxy_apps = '../ProxyApps'
directory_name = 'graphs_stats'

proxy_apps = os.listdir(dir_proxy_apps)

script_name = 'readAndProcessGraph.py'

for proxy_app in proxy_apps:
	proxy_app_path = os.path.join(dir_proxy_apps, proxy_app)
	ranks_list = os.listdir(proxy_app_path)
	for rank in ranks_list:
		if 'ranks' in rank:
			n_ranks = rank.split('_')[0]
			rank_path  = os.path.join(proxy_app_path, rank)
			files = fnmatch.filter(os.listdir(rank_path), '*.mpiP')
			assert len(files) == 1, "The folder %s should contain only one file with extension .mpiP" % rank_path
			file_path = os.path.join(rank_path, files[0])
			returnCode=call(["python3", script_name, file_path, proxy_app, n_ranks, directory_name])
			if returnCode == 0:
				print("Execution of the script over %s with %s ranks was successfull" % (proxy_app, n_ranks))
			else:
				print("Execution of the script over %s with %s ranks failed" % (proxy_app, n_ranks))
			dir_stats = os.path.join(rank_path,directory_name)
			if os.path.exists(dir_stats):
				shutil.rmtree(dir_stats)
			shutil.copytree(directory_name, dir_stats)
			list_html_files = fnmatch.filter(os.listdir('.'), '*.html')
			print("Copying files: " + str(list_html_files) + (" to dir %s" % rank_path))
			returnCode = call(["mv"] + list_html_files + [rank_path])
	
	

