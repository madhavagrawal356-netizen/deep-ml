import numpy as np

def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.

    Args:
        image: nested list / array of shape (H, W, 3) with pixel values.
        k: number of clusters.
        max_iters: maximum number of K-Means iterations.

    Returns:
        Segmented image as a nested list of shape (H, W, 3).
    """
    # Convert input to numpy array and save the original dimensions
    image_np = np.array(image, dtype=np.float64)
    H, W, C = image_np.shape
    
    # Flatten the image into an array of shape (H*W, 3) in row-major order
    pixels = image_np.reshape(-1, C)
    
    # Initialize centroids as the first k pixels
    centroids = pixels[:k].copy()
    
    for _ in range(max_iters):
        # Compute squared Euclidean distances from each pixel to each centroid
        # Shape: (H*W, k)
        distances = np.sum((pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        
        # Assign pixels to the closest centroid index. 
        # np.argmin naturally picks the lowest index in case of a tie.
        labels = np.argmin(distances, axis=1)
        
        # Track the new centroids to check for convergence
        new_centroids = np.empty_like(centroids)
        
        for cluster_idx in range(k):
            # Select pixels belonging to the current cluster
            cluster_pixels = pixels[labels == cluster_idx]
            
            if len(cluster_pixels) > 0:
                new_centroids[cluster_idx] = np.mean(cluster_pixels, axis=0)
            else:
                # If a cluster has no pixels, keep the previous centroid unchanged
                new_centroids[cluster_idx] = centroids[cluster_idx]
        
        # Check for early stopping using exact equality
        if np.array_equal(centroids, new_centroids):
            break
            
        centroids = new_centroids

    # Map each pixel to its corresponding final cluster centroid
    segmented_pixels = centroids[labels]
    
    # Reshape back to the original 3D spatial dimensions
    segmented_image = segmented_pixels.reshape(H, W, C)
    
    # Round values to 4 decimal places and convert back to a nested Python list
    return np.round(segmented_image, 4).tolist()


