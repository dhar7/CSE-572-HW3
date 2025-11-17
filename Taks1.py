import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
def load_data():
    # Load features (10000 samples, 784 features)
    X = pd.read_csv('data.csv', header=None).values.astype(float)
    # Load labels (10000 samples)
    y = pd.read_csv('label.csv', header=None).values.ravel()
    return X, y

# ---------------------------------------------------------
# 2. Distance Metrics
# ---------------------------------------------------------
def euclidean_dist(X, centroids):
    """
    Computes Euclidean distance between X (N x D) and centroids (K x D).
    Returns (N x K) distance matrix.
    """
    # Using squared expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    X2 = np.sum(X**2, axis=1)[:, np.newaxis]
    C2 = np.sum(centroids**2, axis=1)
    XC = np.dot(X, centroids.T)
    # Clip negative values due to precision errors
    dists = np.sqrt(np.maximum(X2 + C2 - 2*XC, 0))
    return dists

def cosine_dist(X, centroids):
    """
    Computes 1 - Cosine Similarity.
    Returns (N x K) distance matrix.
    """
    # Norms
    X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis]
    C_norm = np.linalg.norm(centroids, axis=1)
    
    # Avoid division by zero
    X_norm[X_norm == 0] = 1e-10
    C_norm[C_norm == 0] = 1e-10
    
    # Similarity
    sim = np.dot(X, centroids.T) / (X_norm * C_norm)
    return 1 - sim

def jaccard_dist(X, centroids):
    """
    Computes 1 - Generalized Jaccard Similarity.
    J(A,B) = sum(min(A,B)) / sum(max(A,B))
    Returns (N x K) distance matrix.
    """
    N = X.shape[0]
    K = centroids.shape[0]
    dists = np.zeros((N, K))
    
    # Loop over centroids to save memory (NumPy broadcasting can be heavy for 10k x 10 x 784)
    for k in range(K):
        c = centroids[k]
        # Compute min and max per element
        mins = np.minimum(X, c)
        maxs = np.maximum(X, c)
        
        sum_mins = np.sum(mins, axis=1)
        sum_maxs = np.sum(maxs, axis=1)
        
        # Avoid div by zero
        sum_maxs[sum_maxs == 0] = 1e-10
        
        dists[:, k] = 1 - (sum_mins / sum_maxs)
        
    return dists

# ---------------------------------------------------------
# 3. K-Means Implementation
# ---------------------------------------------------------
class MyKMeans:
    def __init__(self, k=10, metric='euclidean', max_iter=500, tol=1e-4):
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.final_sse = 0
        self.iterations = 0
        
    def fit(self, X):
        N, D = X.shape
        
        # Initialize centroids: Pick k random points from X
        np.random.seed(42) # For reproducibility
        indices = np.random.choice(N, self.k, replace=False)
        self.centroids = X[indices].copy()
        
        prev_sse = float('inf')
        
        for i in range(self.max_iter):
            self.iterations = i + 1
            
            # --- Step 1: Compute Distances ---
            if self.metric == 'euclidean':
                dists = euclidean_dist(X, self.centroids)
            elif self.metric == 'cosine':
                dists = cosine_dist(X, self.centroids)
            elif self.metric == 'jaccard':
                dists = jaccard_dist(X, self.centroids)
            
            # --- Step 2: Assign Clusters ---
            self.labels = np.argmin(dists, axis=1)
            
            # --- Step 3: Compute SSE (Objective Function) ---
            # SSE is defined here as the sum of squared errors (Euclidean) 
            # or Sum of Distances (Cosine/Jaccard) for the metric used.
            min_dists = dists[np.arange(N), self.labels]
            
            if self.metric == 'euclidean':
                current_sse = np.sum(min_dists**2)
            else:
                current_sse = np.sum(min_dists)
            
            # --- Step 4: Check Convergence ---
            # Stop if SSE increases (as per question requirement)
            if current_sse > prev_sse:
                break
            
            # Stop if minimal change in SSE (standard check)
            if abs(prev_sse - current_sse) < self.tol:
                break
                
            prev_sse = current_sse
            
            # --- Step 5: Update Centroids ---
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    # Standard K-means update: Mean of points
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids[k] = self.centroids[k]
            
            # Stop if centroids don't move
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
                
            self.centroids = new_centroids
            
        self.final_sse = prev_sse

# ---------------------------------------------------------
# 4. Main Execution & Analysis
# ---------------------------------------------------------
def main():
    X, y = load_data()
    k = 10  # Number of labels in y
    
    metrics = ['euclidean', 'cosine', 'jaccard']
    
    print(f"{'Metric':<12} | {'SSE (Objective)':<20} | {'Accuracy':<10} | {'Iterations':<10}")
    print("-" * 60)
    
    results = {}
    
    for m in metrics:
        # Initialize and Fit
        kmeans = MyKMeans(k=k, metric=m, max_iter=500)
        kmeans.fit(X)
        
        # Compute Accuracy
        # 1. Assign label to each cluster by majority vote
        cluster_labels = {}
        for i in range(k):
            indices = np.where(kmeans.labels == i)[0]
            if len(indices) > 0:
                true_labels = y[indices]
                counts = np.bincount(true_labels)
                majority = np.argmax(counts)
                cluster_labels[i] = majority
            else:
                cluster_labels[i] = -1
        
        # 2. Predict labels for all points
        pred_labels = np.array([cluster_labels[l] for l in kmeans.labels])
        acc = accuracy_score(y, pred_labels)
        
        results[m] = {
            'sse': kmeans.final_sse,
            'acc': acc,
            'iters': kmeans.iterations
        }
        
        print(f"{m:<12} | {kmeans.final_sse:<20.4f} | {acc:<10.4f} | {kmeans.iterations:<10}")

if __name__ == "__main__":
    main()