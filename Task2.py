import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds

# Helper function for Q(b)
def get_rmse_mae(y_true, y_pred):
    """Calculates RMSE and MAE, ignoring NaNs in predictions."""
    mask = ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return np.nan, np.nan  # No valid predictions
    
    # Apply mask to both true and pred
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    return rmse, mae

class PMF_SVD:
    """Probabilistic Matrix Factorization using SVD."""
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.global_mean = 0
        self.user_means = None
        self.U = None
        self.Vt = None
        self.user_map = None
        self.item_map = None

    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        
        # Create user-item matrix
        train_matrix = train_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        )
        
        # Store user means for bias
        self.user_means = train_matrix.mean(axis=1)
        
        # Mean-center the matrix and fill NaNs with 0 for SVD
        matrix_filled = train_matrix.sub(self.user_means, axis=0).fillna(0)
        
        # Perform SVD
        # k must be < min(shape)
        k = min(self.n_components, min(matrix_filled.shape) - 1)
        U, sigma, Vt = svds(matrix_filled.values, k=k)
        
        # Store factors (U and Vt are the user and item latent factors)
        # Note: sigma is just the singular values, not a diagonal matrix
        self.U = U
        self.Vt = Vt # Vt is already V.T
        
        # Store index mappings
        self.user_map = {uid: i for i, uid in enumerate(train_matrix.index)}
        self.item_map = {mid: i for i, mid in enumerate(train_matrix.columns)}

    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            u, i = row['userId'], row['movieId']
            
            # Check for cold start (user or item not in training)
            if u not in self.user_map or i not in self.item_map:
                predictions.append(self.global_mean)
                continue
            
            # Get matrix indices
            u_idx = self.user_map[u]
            i_idx = self.item_map[i]
            
            # Predict: user_mean + dot(user_factor, item_factor)
            pred = self.user_means.loc[u] + np.dot(self.U[u_idx, :], self.Vt[:, i_idx])
            
            # Clip predictions to valid rating range [1, 5]
            pred = np.clip(pred, 1, 5)
            predictions.append(pred)
            
        return np.array(predictions)

class BasicCollaborativeFiltering:
    """
    User-Based or Item-Based Collaborative Filtering.
    
    This is a basic implementation and will be slow,
    especially for Item-Based CF on this dataset.
    """
    def __init__(self, mode='user', k=40, sim_metric='cosine'):
        if mode not in ['user', 'item']:
            raise ValueError("mode must be 'user' or 'item'")
        if sim_metric not in ['cosine', 'msd', 'pearson']:
            raise ValueError("sim_metric must be 'cosine', 'msd', or 'pearson'")
            
        self.mode = mode
        self.k = k
        self.sim_metric = sim_metric
        self.sim_matrix = None
        self.train_matrix = None
        self.mean_ratings = None
        self.global_mean = 0

    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        
        if self.mode == 'user':
            # User-Based: rows=users, cols=items
            self.train_matrix = train_df.pivot(index='userId', columns='movieId', values='rating')
        else:
            # Item-Based: rows=items, cols=users
            self.train_matrix = train_df.pivot(index='movieId', columns='userId', values='rating')

        # Store mean ratings for prediction bias
        self.mean_ratings = self.train_matrix.mean(axis=1)
        
        # Fill NaNs with 0 for similarity calculation
        matrix_filled = self.train_matrix.fillna(0)

        # Calculate similarity matrix
        print(f"  Calculating {self.sim_metric} similarity for {self.mode}-based... (This may be slow)")
        
        if self.sim_metric == 'cosine':
            self.sim_matrix = cosine_similarity(matrix_filled)
        elif self.sim_metric == 'msd':
            # MSD is inversely related to Euclidean distance
            dists = euclidean_distances(matrix_filled)
            # 1 / (1 + dist) is a common way to convert dist to sim
            with np.errstate(divide='ignore'): # Ignore divide-by-zero
                self.sim_matrix = 1 / (1 + dists)
            np.fill_diagonal(self.sim_matrix, 0) # Set self-similarity to 0
        elif self.sim_metric == 'pearson':
            # Pearson is corrcoef on the raw (non-filled) data
            # np.corrcoef handles NaNs by default
            self.sim_matrix = np.corrcoef(matrix_filled)
            np.nan_to_num(self.sim_matrix, copy=False) # Convert NaNs (from 0 variance) to 0
            
        # Convert to DataFrame for easy lookup by userId/movieId
        self.sim_matrix = pd.DataFrame(
            self.sim_matrix,
            index=self.train_matrix.index,
            columns=self.train_matrix.index
        )

    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            user, item = row['userId'], row['movieId']
            
            # Select target and candidate based on mode
            if self.mode == 'user':
                target, candidate = user, item
            else:
                target, candidate = item, user # target=item, candidate=user

            # --- Cold Start Check ---
            # 1. Target not in training (new user or new item)
            # 2. Candidate not in training (new item or new user)
            if target not in self.sim_matrix.index or candidate not in self.train_matrix.columns:
                predictions.append(self.global_mean)
                continue
            
            # Get similarities of the target (user U or item I)
            target_sims = self.sim_matrix.loc[target]
            
            # Get ratings for the candidate (item I or user U)
            candidate_ratings = self.train_matrix[candidate]
            
            # --- Find Neighbors ---
            # 1. Find neighbors who have rated the candidate
            valid_neighbors = candidate_ratings.dropna().index
            
            # 2. Get similarities for only these valid neighbors
            valid_sims = target_sims.loc[valid_neighbors]
            
            # 3. Exclude self-similarity
            if target in valid_sims:
                valid_sims = valid_sims.drop(target)
            
            # --- Top-K ---
            if valid_sims.empty:
                # No neighbors, predict target's mean rating
                predictions.append(self.mean_ratings.loc[target])
                continue
            
            # Sort by similarity and get top K
            top_k_sims = valid_sims.nlargest(self.k)
            
            # Get the ratings from these neighbors
            top_k_ratings = candidate_ratings.loc[top_k_sims.index]
            
            # --- Calculate Prediction (Weighted Average) ---
            # pred = sum(sim * rating) / sum(abs(sim))
            sum_sim_x_rating = np.dot(top_k_sims.values, top_k_ratings.values)
            sum_abs_sim = np.sum(np.abs(top_k_sims.values))
            
            if sum_abs_sim == 0:
                pred = self.mean_ratings.loc[target]
            else:
                pred = sum_sim_x_rating / sum_abs_sim

            # Clip to [1, 5]
            pred = np.clip(pred, 1, 5)
            predictions.append(pred)
            
        return np.array(predictions)

def run_task_2():
    print("--- Task 2: Recommender System Analysis (No Surprise) ---")
    
    # --- Q2 (a): Load Data ---
    print("Loading Ratings Data...")
    try:
        df = pd.read_csv('ratings_small.csv')
        df = df[['userId', 'movieId', 'rating']]
    except FileNotFoundError:
        print("Error: 'ratings_small.csv' not found.")
        return

    # --- Q2 (c & d): Compare Algorithms (5-Fold CV) ---
    print("\n--- Q2 (c, d): 5-Fold CV for PMF, User-CF, Item-CF ---")
    print("WARNING: This will take several minutes...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'PMF (SVD)': PMF_SVD(n_components=20),
        'User-Based CF (k=40, msd)': BasicCollaborativeFiltering(mode='user', k=40, sim_metric='msd'),
        'Item-Based CF (k=40, msd)': BasicCollaborativeFiltering(mode='item', k=40, sim_metric='msd')
    }
    
    results_c_d = {name: {'RMSE': [], 'MAE': []} for name in models}
    
    fold = 1
    for train_idx, test_idx in kf.split(df):
        print(f"\nStarting Fold {fold}/5...")
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        for name, model in models.items():
            start_time = time.time()
            print(f"  Fitting {name}...")
            model.fit(train_data)
            
            print(f"  Predicting with {name}...")
            preds = model.predict(test_data)
            
            rmse, mae = get_rmse_mae(test_data['rating'].values, preds)
            results_c_d[name]['RMSE'].append(rmse)
            results_c_d[name]['MAE'].append(mae)
            
            elapsed = time.time() - start_time
            print(f"  {name} Fold {fold} Done. RMSE: {rmse:.4f}, Time: {elapsed:.2f}s")
        fold += 1

    print("\n--- Q2(c) & Q2(d) Results ---")
    for name, metrics in results_c_d.items():
        print(f"  {name}:\n    Average RMSE = {np.mean(metrics['RMSE']):.4f}\n    Average MAE  = {np.mean(metrics['MAE']):.4f}")

    # --- Q2 (e, f, g): Analysis on a single split for speed ---
    # Running CV for similarity/K analysis would take hours
    print("\n--- Q2 (e, f, g): Analysis on a single 3-fold split ---")
    # We use n_splits=3 and take the first fold
    kf_analysis = KFold(n_splits=3, shuffle=True, random_state=42)
    train_idx, test_idx = next(kf_analysis.split(df))
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    y_true = test_data['rating'].values

    # --- Q2 (e): SIMILARITY METRICS IMPACT ---
    print("\n--- Q2 (e): Impact of Similarity Metrics (Cosine, MSD, Pearson) ---")
    sim_metrics = ['cosine', 'msd', 'pearson']
    cf_types = ['User-Based', 'Item-Based']
    impact_results_e = {cf_type: [] for cf_type in cf_types}

    for sim in sim_metrics:
        for cf_type in cf_types:
            mode = 'user' if cf_type == 'User-Based' else 'item'
            print(f"  Testing {cf_type} with {sim}...")
            model = BasicCollaborativeFiltering(mode=mode, k=40, sim_metric=sim)
            model.fit(train_data)
            preds = model.predict(test_data)
            rmse, _ = get_rmse_mae(y_true, preds)
            impact_results_e[cf_type].append(rmse)
            print(f"    RMSE: {rmse:.4f}")

    # Plotting Q2(e)
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(len(sim_metrics))
    width = 0.35
    plt.bar(x_axis - width/2, impact_results_e['User-Based'], width, label='User-Based', color='royalblue')
    plt.bar(x_axis + width/2, impact_results_e['Item-Based'], width, label='Item-Based', color='darkorange')
    plt.xticks(x_axis, [s.upper() for s in sim_metrics])
    plt.ylabel('RMSE (on one split)')
    plt.xlabel('Similarity Metric')
    plt.title('Impact of Similarity Metrics on RMSE')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('Q2_e_similarity_impact_scratch.png')
    print("Plot saved to 'Q2_e_similarity_impact_scratch.png'")

    # --- Q2 (f & g): NUMBER OF NEIGHBORS (K) IMPACT ---
    print("\n--- Q2 (f, g): Impact of Neighbor Count (K) ---")
    ks = [5, 10, 20, 30, 40, 50, 60]
    impact_results_f = {cf_type: [] for cf_type in cf_types}
    
    for k in ks:
        for cf_type in cf_types:
            mode = 'user' if cf_type == 'User-Based' else 'item'
            print(f"  Testing {cf_type} with K={k} (msd)...")
            model = BasicCollaborativeFiltering(mode=mode, k=k, sim_metric='msd')
            model.fit(train_data)
            preds = model.predict(test_data)
            rmse, _ = get_rmse_mae(y_true, preds)
            impact_results_f[cf_type].append(rmse)
            
    print("\n--- Q2(f) Results (RMSE vs. K) ---")
    k_results_df = pd.DataFrame({
        'K': ks,
        'User-Based': impact_results_f['User-Based'],
        'Item-Based': impact_results_f['Item-Based']
    })
    print(k_results_df.to_string(index=False, float_format="%.4f"))

    # Plotting Q2(f)
    plt.figure(figsize=(10, 6))
    plt.plot(ks, impact_results_f['User-Based'], marker='o', linestyle='-', label='User-Based', color='royalblue')
    plt.plot(ks, impact_results_f['Item-Based'], marker='s', linestyle='-', label='Item-Based', color='darkorange')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('RMSE (on one split)')
    plt.title('Impact of K Neighbors on RMSE (MSD Sim)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('Q2_f_neighbors_impact_scratch.png')
    print("Plot saved to 'Q2_f_neighbors_impact_scratch.png'")

    # --- Q2 (g): Identify Best K ---
    best_k_user_idx = np.argmin(impact_results_f['User-Based'])
    best_k_user = ks[best_k_user_idx]
    best_rmse_user = impact_results_f['User-Based'][best_k_user_idx]
    
    best_k_item_idx = np.argmin(impact_results_f['Item-Based'])
    best_k_item = ks[best_k_item_idx]
    best_rmse_item = impact_results_f['Item-Based'][best_k_item_idx]

    print("\n--- Q2(g) Results (Best K) ---")
    print(f"  Best K for User-Based: {best_k_user} (with RMSE: {best_rmse_user:.4f})")
    print(f"  Best K for Item-Based: {best_k_item} (with RMSE: {best_rmse_item:.4f})")
    print(f"  Are they the same? {'Yes' if best_k_user == best_k_item else 'No'}")

if __name__ == "__main__":
    run_task_2()