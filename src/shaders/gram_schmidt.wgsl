// shaders/gram_schmidt_parallel.wgsl
@group(0) @binding(0) var<storage, read> input_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> reduction_buffer: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec2<u32>;

var<workgroup> workgroup_dot: array<f32, 64>;
var<workgroup> workgroup_norm: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn gram_schmidt_parallel(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let rows = dimensions.x;
    let cols = dimensions.y;
    let global_idx = global_id.x;
    let local_idx = local_id.x;
    
    if (global_idx >= rows * cols) {
        return;
    }
    
    let row = global_idx / cols;
    let col = global_idx % cols;
    
    // Initialize output with input
    output_matrix[global_idx] = input_matrix[global_idx];
    
    workgroupBarrier();
    
    // Process Gram-Schmidt sequentially by row, but parallel within each operation
    for (var ortho_row = 0u; ortho_row < rows; ortho_row = ortho_row + 1u) {
        // Only orthogonalize rows after the current orthogonal row
        if (row <= ortho_row) {
            continue;
        }
        
        // Parallel dot product computation
        var dot_contribution: f32 = 0.0;
        var norm_contribution: f32 = 0.0;
        
        if (col < cols) {
            let current_idx = row * cols + col;
            let ortho_idx = ortho_row * cols + col;
            dot_contribution = output_matrix[current_idx] * output_matrix[ortho_idx];
            norm_contribution = output_matrix[ortho_idx] * output_matrix[ortho_idx];
        }
        
        // Store contributions in workgroup memory
        workgroup_dot[local_idx] = dot_contribution;
        workgroup_norm[local_idx] = norm_contribution;
        
        workgroupBarrier();
        
        // Parallel reduction for dot product
        var offset = 32u;
        while (offset > 0u) {
            if (local_idx < offset) {
                workgroup_dot[local_idx] = workgroup_dot[local_idx] + workgroup_dot[local_idx + offset];
                workgroup_norm[local_idx] = workgroup_norm[local_idx] + workgroup_norm[local_idx + offset];
            }
            workgroupBarrier();
            offset = offset / 2u;
        }
        
        // First thread computes final dot product and applies orthogonalization
        if (local_idx == 0u) {
            let total_dot = workgroup_dot[0];
            let total_norm = workgroup_norm[0];
            
            if (total_norm > 1e-10) {
                let coefficient = total_dot / total_norm;
                
                // Store coefficient for all threads to use
                reduction_buffer[workgroup_id.x] = coefficient;
            } else {
                reduction_buffer[workgroup_id.x] = 0.0;
            }
        }
        
        workgroupBarrier();
        
        let coefficient = reduction_buffer[workgroup_id.x];
        
        // All threads apply orthogonalization to their columns
        if (col < cols && coefficient != 0.0) {
            let current_idx = row * cols + col;
            let ortho_idx = ortho_row * cols + col;
            output_matrix[current_idx] = output_matrix[current_idx] - coefficient * output_matrix[ortho_idx];
        }
        
        workgroupBarrier();
    }
}