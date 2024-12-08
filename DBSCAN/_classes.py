import numpy as np
from _distance import _calculate_distance

class DBSCAN:
    def __init__(self, eps=0.25, min_samples=15, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X):
        """Applies DBSCAN clustering algorithm to the dataset."""
        # Change to array
        X = np.array(X).copy()

        # Find neighbors
        neighbors = self._search_neighbors(X)
        core_ = self._search_core_points(neighbors)

        # Iteration visit
        assignment = {}  # indexing
        next_cluster_id = 0  # cluster number
        visited = set()

        # Visited core points
        for i in core_:
            if i not in visited:  # If not visited
                visited.add(i)
                assignment[i] = next_cluster_id  # Assign cluster

                # Update new cluster
                self._expand_cluster(i, neighbors, core_, assignment, visited)
                next_cluster_id += 1  # Finish cluster

        # Finishing
        self.core_sample_indices_ = core_  # Core points
        self.labels_ = self._assignment_to_labels(assignment, X)  # Assign labels

    def _search_neighbors(self, X):
        """Finds the neighbors of each point in the dataset based on the epsilon distance."""
        n_samples, n_features = X.shape
        neighbors = []

        for i in range(n_samples):
            adj_i = self._region_query(X[i], X)  # Find neighbors within eps distance
            ind_i = self._get_true_indices(adj_i)  # Get indices of neighbors
            neighbors.append(adj_i)

        return neighbors

    def _region_query(self, p, X):
        """Finds the points within the epsilon distance of a given point"""
        n_samples, n_features = X.shape
        adj = np.zeros(n_samples, dtype=int)

        # Start iterating over all points
        for i in range(n_samples):
            dist_i = _calculate_distance(point_1=p, point_2=X[i], metric=self.metric)

            # Mark points within epsilon distance
            if dist_i <= self.eps:
                adj[i] = 1

        return adj

    def _get_true_indices(self, sample):
        """Extracts the indices of points that are within epsilon distance."""
        indices = set(np.where(sample == 1)[0])
        return indices

    def _search_core_points(self, neighbors):
        """Identifies core points based on the number of neighbors."""
        # Initialize core points
        core_ind = set()
        
        # Iterate through each point and its list of neighbors
        for i, neigh_i in enumerate(neighbors):
            
            # A point is a core point if it has at least 'min_samples' neighbors
            if len(neigh_i) >= self.min_samples:
                
                core_ind.add(i)  # Add the index to the set

        return core_ind

    def _expand_cluster(self, p, neighbors, core_, assignment, visited):
        """Expands the cluster from a given core point by adding its neighbors."""
        reachable = set(neighbors[p])

        while reachable:
            q = reachable.pop()

            if q not in visited:
                # Mark as visited
                visited.add(q)

                # If q is also a core point, add its neighbors
                if q in core_:
                    reachable |= neighbors[q]

            # Assign q to the same cluster as p
            if q not in assignment:
                assignment[q] = assignment[p]

    def _assignment_to_labels(self, assignment, X):
        """Converts the assignment dictionary into a label array."""
        # Get the number of samples (n_samples)
        n_samples, _ = X.shape
        
        # Initialize labels array with -1 (unassigned) for all samples
        labels = -1 * np.ones(n_samples, dtype=int)
        
        # Assign the cluster ID from the 'assignment'
        for i, cluster_id in assignment.items():
            labels[i] = cluster_id  
        
        return labels
