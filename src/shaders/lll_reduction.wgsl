// shaders/lll_reduction.wgsl
@group(0) @binding(0) var<storage, read_write> basis: array<f32>;
@group(0) @binding(1) var<storage, read_write> gs_coeffs: array<f32>;
@group(0) @binding(2) var<storage, read_write> norms: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<f32>; // delta, eta, precision
@group(0) @binding(4) var<uniform> dimensions: vec2<u32>; // rows, cols

var<workgroup> reduction_data: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn lll_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let rows = dimensions.x;
    let cols = dimensions.y;
    let delta = params.x;
    let eta = params.y;
    
    let global_idx = global_id.x;
    let local_idx = local_id.x;
    
    if (global_idx >= rows * cols) {
        return;
    }
    
    let row = global_idx / cols;
    let col = global_idx % cols;
    
    // Initialize Gram-Schmidt coefficients (identity)
    if (row < rows && col < rows) {
        gs_coeffs[row * rows + col] = select(0.0, 1.0, row == col);
    }
    
    workgroupBarrier();
    
    // Size reduction phase
    for (var k = 1u; k < rows; k = k + 1u) {
        for (var j = 0u; j < k; j = j + 1u) {
            // Compute mu_kj in parallel
            var mu_kj: f32 = 0.0;
            if (local_idx < cols) {
                let basis_k_idx = k * cols + local_idx;
                let basis_j_idx = j * cols + local_idx;
                mu_kj = basis[basis_k_idx] * basis[basis_j_idx];
            }
            
            // Parallel reduction for mu_kj
            reduction_data[local_idx] = mu_kj;
            workgroupBarrier();
            
            var offset = 32u;
            while (offset > 0u) {
                if (local_idx < offset) {
                    reduction_data[local_idx] = reduction_data[local_idx] + reduction_data[local_idx + offset];
                }
                workgroupBarrier();
                offset = offset / 2u;
            }
            
            let total_mu = reduction_data[0];
            let rounded_mu = round(total_mu);
            
            // Apply size reduction
            if (rounded_mu != 0.0 && local_idx < cols) {
                let basis_k_idx = k * cols + local_idx;
                let basis_j_idx = j * cols + local_idx;
                basis[basis_k_idx] = basis[basis_k_idx] - rounded_mu * basis[basis_j_idx];
            }
            
            workgroupBarrier();
        }
        
        // Check Lovász condition
        if (k >= 1u) {
            var norm_k: f32 = 0.0;
            var norm_km1: f32 = 0.0;
            
            if (local_idx < cols) {
                let idx_k = k * cols + local_idx;
                let idx_km1 = (k - 1u) * cols + local_idx;
                norm_k = basis[idx_k] * basis[idx_k];
                norm_km1 = basis[idx_km1] * basis[idx_km1];
            }
            
            // Parallel reduction for norms
            reduction_data[local_idx] = norm_k;
            workgroupBarrier();
            
            var offset = 32u;
            while (offset > 0u) {
                if (local_idx < offset) {
                    reduction_data[local_idx] = reduction_data[local_idx] + reduction_data[local_idx + offset];
                }
                workgroupBarrier();
                offset = offset / 2u;
            }
            
            let total_norm_k = reduction_data[0];
            
            reduction_data[local_idx] = norm_km1;
            workgroupBarrier();
            
            offset = 32u;
            while (offset > 0u) {
                if (local_idx < offset) {
                    reduction_data[local_idx] = reduction_data[local_idx] + reduction_data[local_idx + offset];
                }
                workgroupBarrier();
                offset = offset / 2u;
            }
            
            let total_norm_km1 = reduction_data[0];
            
            // Compute mu for Lovász condition
            var mu_k_km1: f32 = 0.0;
            if (local_idx < cols) {
                let idx_k = k * cols + local_idx;
                let idx_km1 = (k - 1u) * cols + local_idx;
                mu_k_km1 = basis[idx_k] * basis[idx_km1];
            }
            
            reduction_data[local_idx] = mu_k_km1;
            workgroupBarrier();
            
            offset = 32u;
            while (offset > 0u) {
                if (local_idx < offset) {
                    reduction_data[local_idx] = reduction_data[local_idx] + reduction_data[local_idx + offset];
                }
                workgroupBarrier();
                offset = offset / 2u;
            }
            
            let total_mu = reduction_data[0];
            let mu_squared = total_mu * total_mu;
            
            // Swap condition
            if (total_norm_k < (delta - mu_squared) * total_norm_km1) {
                // Swap rows k-1 and k
                if (local_idx < cols) {
                    let idx_k = k * cols + local_idx;
                    let idx_km1 = (k - 1u) * cols + local_idx;
                    var temp = basis[idx_km1];
                    basis[idx_km1] = basis[idx_k];
                    basis[idx_k] = temp;
                }
            }
            
            workgroupBarrier();
        }
    }
}