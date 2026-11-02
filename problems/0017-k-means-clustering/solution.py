def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	final_centroids = []
	def dist(point, centroid):
		if len(point)==2:
			x,y = point
			x2,y2 = centroid
			return ((x2-x)**2+(y2-y)**2)**0.5
		else:
			x,y,z=point
			x2,y2,z2 = centroid
			return ((x2-x)**2+(y2-y)**2+(z2-z)**2)**0.5

	for iteration in range(max_iterations):
		clusters = [[] for _ in range(k)]
		for point in points:
			min_dist=2*10**10
			assign_to = 1
			for i in range(len(initial_centroids)):
				if dist(point,initial_centroids[i])<min_dist:
					assign_to = i
					min_dist = dist(point, initial_centroids[i])
			clusters[assign_to].append(point)
		for item in clusters:
			sum_x=0
			sum_y=0
			sum_z=0
			if len(item[0])==2:
				for x,y in item:
					sum_x+=x
					sum_y+=y
				centx,centy = sum_x/len(item), sum_y/len(item)
				final_centroids.append((centx,centy))
			else: 
				for x,y,z in item:
					sum_x+=x
					sum_y+=y
					sum_z+=z 
				centx, centy, centz = sum_x/len(item), sum_y/len(item), sum_z/len(item)
				final_centroids.append((centx,centy,centz))

		initial_centroids=final_centroids
		final_centroids=[]



	return initial_centroids