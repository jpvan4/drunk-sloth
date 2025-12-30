// shaders/matrix_mult.wgsl
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // rows_a, cols_a, cols_b

var<workgroup> tile_a: array<f32, 64>;
var<workgroup> tile_b: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn matrix_mult(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let rows_a = dimensions.x;
    let cols_a = dimensions.y;
    let cols_b = dimensions.z;
    
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= rows_a || col >= cols_b) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Tiled matrix multiplication for better cache performance
    let tile_size = 8u;
    for (var t = 0u; t < (cols_a + tile_size - 1u) / tile_size; t = t + 1u) {
        let tile_row = local_id.x;
        let tile_col = local_id.y;
        
        // Load tile from matrix A
        let a_row = row;
        let a_col = t * tile_size + tile_col;
        if (a_col < cols_a) {
            tile_a[tile_row * tile_size + tile_col] = matrix_a[a_row * cols_a + a_col];
        } else {
            tile_a[tile_row * tile_size + tile_col] = 0.0;
        }
        
        // Load tile from matrix B  
        let b_row = t * tile_size + tile_row;
        let b_col = col;
        if (b_row < cols_a) {
            tile_b[tile_row * tile_size + tile_col] = matrix_b[b_row * cols_b + b_col];
        } else {
            tile_b[tile_row * tile_size + tile_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial sum for this tile
        for (var k = 0u; k < tile_size; k = k + 1u) {
            sum = sum + tile_a[tile_row * tile_size + k] * tile_b[k * tile_size + tile_col];
        }
        
        workgroupBarrier();
    }
    
    matrix_c[row * cols_b + col] = sum;
}