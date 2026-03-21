import numpy as np
from scipy.linalg import lu

class TCIFitter:
    def __init__(self, func, domain, rank=10):
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.rank = rank
        # Keep pivot_paths as integer indices
        self.pivot_paths = np.zeros((rank, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            self.pivot_paths[:, d] = np.random.randint(0, len(domain[d]), size=rank)

    def _path_to_coords(self, path_indices):
        """Convert discrete index paths to physical/logical coordinate arrays."""
        # Note: returned type depends on the content type of self.domain
        return np.array([self.domain[d][path_indices[d]] for d in range(self.n_dims)])

    def _get_maxvol(self, matrix, tolerance=1.05):
        """
        MaxVol (Maximum Volume) row selection.
        
        Fix note (2026-02-09):
        Uses QR decomposition with column pivoting instead of LU for better
        stability and controlled output length.
        """
        m, n = matrix.shape
        if n == 0: 
            return np.array([], dtype=int)
        
        k = min(self.rank, m, n)  # select at most k rows
        
        try:
            # Pivoted QR on the transposed matrix
            # matrix.T: (n, m), Q: (n, k), R: (k, m), P: (m,)
            from scipy.linalg import qr
            Q, R, P = qr(matrix.T, mode='economic', pivoting=True)
            I = np.array(P[:k], dtype=int)
        except Exception:
            # Fallback: select by row norms
            row_norms = np.linalg.norm(matrix, axis=1)
            I = np.argsort(row_norms)[-k:][::-1]
        
        # Iterative MaxVol refinement (greedy improvement)
        for _ in range(20):
            try:
                sub_matrix = matrix[I, :]
                if sub_matrix.shape[0] == 0:
                    break
                # Z[i, j] = (volume gain after replacement)
                Z = matrix @ np.linalg.pinv(sub_matrix)
                max_idx = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)
                if max_idx[1] >= len(I): 
                    break
                if np.abs(Z[max_idx]) > tolerance: 
                    I[max_idx[1]] = max_idx[0]
                else: 
                    break
            except:
                break
        
        return I.astype(int)

    def _build_sweep_matrix_vectorized(self, d, left, right):
        """
        Vectorized construction of the sweep matrix.
        Removed dtype=float coercion to support integer pass-through to QTTEncoder.
        """
        n_curr = len(self.domain[d])
        r_prev, r_next = left.shape[0], right.shape[0]
        
        # 1. Build full index tensor (Batch, n_dims)
        coords_idx = np.zeros((r_prev, n_curr, r_next, self.n_dims), dtype=int)
        
        # Fill left paths
        for i in range(d): 
            coords_idx[:,:,:,i] = left[:, i][:, None, None]
        # Fill current dimension (iterate over all possible values)
        coords_idx[:,:,:,d] = np.arange(n_curr, dtype=int)[None, :, None]
        # Fill right paths
        for i in range(self.n_dims - d - 1):
            coords_idx[:,:,:,d+1+i] = right[:, i][None, None, :]
        
        flat_idx = coords_idx.reshape(-1, self.n_dims)
        
        # 2. Map indices back to domain values (key fix)
        # Create container with type following the first domain element, not forcing float
        sample_val = self.domain[0][0]
        flat_coords = np.zeros(flat_idx.shape, dtype=type(sample_val))
        
        for i in range(self.n_dims):
            flat_coords[:, i] = self.domain[i][flat_idx[:, i]]
            
        # 3. Evaluate the physical function
        # Reshape to (Left * Current, Right) to match TCI matrix structure
        return self.func(flat_coords).reshape(r_prev * n_curr, r_next)

    def build_cores(self, anchors=None, n_sweeps=3, verbose=False):
        """
        Bidirectional TCI sweep algorithm (DMRG-like).
        
        Fix note (2026-02-09):
        Unidirectional sweeps cannot guarantee global consistency in deep TT
        structures. Bidirectional sweeps let each site "see" optimization
        results from both ends.
        
        Args:
            anchors: strategic anchor point array
            n_sweeps: number of full sweeps (forward + backward = 1 sweep)
            verbose: whether to print convergence info
        """
        # 1. Smart initialization
        if anchors is not None:
            anchor_coords = np.array([self.domain[d][anchors[:,d]] for d in range(self.n_dims)]).T
            vals = self.func(anchor_coords)
            best = np.argmax(np.abs(vals))
            if np.abs(vals[best]) > 1e-15:
                for r in range(self.rank): 
                    self.pivot_paths[r, :] = anchors[best]
                if verbose:
                    print(f"[Init] Best anchor value: {vals[best]:.6e}")

        # 2. Bidirectional sweeps
        for sweep in range(n_sweeps):
            pivot_change = 0
            
            # 2a. Forward sweep (d: 0 -> n_dims-1)
            for d in range(self.n_dims):
                old_pivots = self.pivot_paths[:, d].copy()
                self._optimize_site(d)
                pivot_change += np.sum(self.pivot_paths[:, d] != old_pivots)
            
            # 2b. Backward sweep (d: n_dims-2 -> 0)
            for d in range(self.n_dims - 2, -1, -1):
                old_pivots = self.pivot_paths[:, d].copy()
                self._optimize_site(d)
                pivot_change += np.sum(self.pivot_paths[:, d] != old_pivots)
            
            if verbose:
                print(f"[Sweep {sweep+1}/{n_sweeps}] Pivot changes: {pivot_change}")
            
            # Early stopping: pivots no longer change
            if pivot_change == 0:
                if verbose:
                    print(f"[Early Stop] Converged at sweep {sweep+1}")
                break
    
    def _optimize_site(self, d):
        """Optimize pivot selection for a single site."""
        l = np.zeros((1, 0), dtype=int) if d == 0 else self.pivot_paths[:, :d]
        r = np.zeros((1, 0), dtype=int) if d == self.n_dims-1 else self.pivot_paths[:, d+1:]
        
        sweep_matrix = self._build_sweep_matrix_vectorized(d, l, r)
        new_I = self._get_maxvol(sweep_matrix)
        
        old = self.pivot_paths.copy()
        for r_idx, f_idx in enumerate(new_I):
            if r_idx >= self.rank: break
            prev_r_idx = f_idx // len(self.domain[d])
            curr_val_idx = f_idx % len(self.domain[d])
            
            if d > 0: 
                self.pivot_paths[r_idx, :d] = old[prev_r_idx, :d]
            self.pivot_paths[r_idx, d] = curr_val_idx
        
        # Broadcast to fill remaining ranks
        if 0 < len(new_I) < self.rank:
            for r_idx in range(len(new_I), self.rank):
                self.pivot_paths[r_idx, :] = self.pivot_paths[r_idx % len(new_I), :]