import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    def dist(x0,y0):
        x1,y1 = x0
        x2,y2 = y0
        return np.sqrt((x1-x2)**2+(y1-y2)**2)
    dists = []
    for point in points:
        dists.append(dist(point,query_point))
    dists = np.array(dists)
    args = np.argsort(dists, kind='stable')
    args = args[:k]
    return [points[i] for i in args]