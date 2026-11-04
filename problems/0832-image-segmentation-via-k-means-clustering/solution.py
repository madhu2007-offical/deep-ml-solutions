import numpy as np

def segment_image(image, k, max_iters=100):
    img = np.asarray(image, dtype=np.float64)
    H, W, C = img.shape
    pixels = img.reshape(-1, C)                  # row-major flatten

    centroids = pixels[:k].copy()                # first k pixels
    labels = np.zeros(len(pixels), dtype=int)

    for _ in range(max_iters):
        # squared Euclidean distance to every centroid: (N, k)
        d = ((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d, axis=1)            # ties -> lowest index

        new_centroids = centroids.copy()         # empty clusters keep old centroid
        for j in range(k):
            members = pixels[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)

        if np.array_equal(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    segmented = centroids[labels].reshape(H, W, C)
    return np.round(segmented, 4).tolist()