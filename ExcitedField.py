import cupy as cp
import numpy as np
import torch
def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func  

chunked_dot_product_reduced_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void chunked_dot_product_reduced_kernel(
        const complex<float>* interaction_matrix,
        const complex<float>* p_fft,
        complex<float>* result,
        int nx, int ny, int nz,
        int chunk_start, int chunk_size, int original_ny
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int total_size = chunk_size * 2 * ny * 2 * nz * 3;
        if (i >= total_size) return;
        
        int full_idx = i / 3;
        int component = i % 3;
    
        int local_ix = full_idx / (2 * ny * 2 * nz);
        int iy = (full_idx / (2 * nz)) % (2 * ny);  // Changed to match original
        int iz = full_idx % (2 * nz);
        
        // Global x position
        int ix = chunk_start + local_ix;
        
        // Determine which region we're in
        bool in_x_mirror = (ix > nx);   // Changed to >
        bool in_y_mirror = (iy > ny);   // Changed to >
        bool in_z_mirror = (iz > nz);   // Changed to >
        
        // Calculate mirrored indices
        int mx = in_x_mirror ? (2*nx - ix) : ix;     // Removed -1
        int my = in_y_mirror ? (2*ny - iy) : iy;     // Removed -1
        int mz = in_z_mirror ? (2*nz - iz) : iz;     // Removed -1
        
        // Clamp to reduced matrix size
        mx = min(mx, nx);
        my = min(my, ny);
        mz = min(mz, nz);
        
        int reduced_idx = mx * (ny+1) * (nz+1) + my * (nz+1) + mz;
        
        // Calculate signs for symmetry
        float xy_sign = ((in_x_mirror != in_y_mirror) ? -1.0f : 1.0f);
        float xz_sign = ((in_x_mirror != in_z_mirror) ? -1.0f : 1.0f);
        float yz_sign = ((in_y_mirror != in_z_mirror) ? -1.0f : 1.0f);
        
        complex<float> sum = 0;
        
        // Global index for p_fft
        int global_idx = local_ix * 2 * ny * 2 * nz + iy * 2 * nz + iz;
        
        if (component == 0) {
            sum += interaction_matrix[reduced_idx * 6 + 0] * p_fft[global_idx * 3];
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[global_idx * 3 + 1];
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[global_idx * 3 + 2];
        } 
        else if (component == 1) {
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[global_idx * 3];
            sum += interaction_matrix[reduced_idx * 6 + 3] * p_fft[global_idx * 3 + 1];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[global_idx * 3 + 2];
        } 
        else {
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[global_idx * 3];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[global_idx * 3 + 1];
            sum += interaction_matrix[reduced_idx * 6 + 5] * p_fft[global_idx * 3 + 2];
        }
        
        result[i] = sum;
    }

    ''', 'chunked_dot_product_reduced_kernel')

def setup_chunked_values(nx, ny, nz):
    total_length = 2*nx
    target_chunk_size = 10
    min_chunk_size = 5
    
    best_num_chunks = 1
    best_chunk_size = total_length
    smallest_difference = abs(target_chunk_size - best_chunk_size)

    for num_chunks in range(1, total_length // min_chunk_size + 1):
        if total_length % num_chunks == 0:
            chunk_size = total_length // num_chunks
            if chunk_size >= min_chunk_size:
                difference = abs(target_chunk_size - chunk_size)
                if difference < smallest_difference:
                    best_num_chunks = num_chunks
                    best_chunk_size = chunk_size
                    smallest_difference = difference

    num_chunks = best_num_chunks
    chunk_size = best_chunk_size
    print(f'num chunks: {num_chunks}, chunk size: {chunk_size}')

    # Rest of the function remains the same
    chunks = []
    chunk_params = []
    threads_per_block = 256
    
    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk = cp.arange(chunk_start, chunk_start + chunk_size)
        chunks.append(chunk)
        
        blocks = ((chunk_size * 2 * ny * 2 * nz * 3 + threads_per_block - 1) 
                 // threads_per_block)
        
        chunk_params.append({
            'chunk': chunk,
            'chunk_start': chunk_start,
            'chunk_size': chunk_size,
            'blocks': blocks,
            'threads_per_block': threads_per_block
        })
    
    return chunk_params
@free_gpu_memory 
def E_field_solver(polarization, interaction_matrix, grid_size):
    """
    Calculate the excited E field from a given polarization using G*p
    
    Parameters:
    polarization: The input polarization (p) [nx,ny,nz,3]
    interaction_matrix: The interaction matrix (G)
    grid_size: Tuple of (nx, ny, nz) for the grid dimensions
    """
    nx, ny, nz = grid_size
    expected_size = nx * ny * nz * 3

    # Shape validation
    if isinstance(polarization, cp.ndarray):
        if polarization.size != expected_size:
            raise ValueError(f"Polarization size {polarization.size} doesn't match expected size {expected_size}")
        if len(polarization.shape) == 1:
            polarization = polarization.reshape(nx, ny, nz, 3)
    else:
        polarization = cp.asarray(polarization, dtype=cp.complex64)
        if polarization.size != expected_size:
            raise ValueError(f"Polarization size {polarization.size} doesn't match expected size {expected_size}")
        if len(polarization.shape) == 1:
            polarization = polarization.reshape(nx, ny, nz, 3)

    # Convert inputs to device arrays if needed
    polarization = cp.asarray(polarization, dtype=cp.complex64)
    interaction_matrix = cp.asarray(interaction_matrix, dtype=cp.complex64)
    
    # Initialize output array
    out_array = torch.zeros((nx*2, ny, nz, 3), dtype=torch.complex64, device='cuda')
    out_array[:nx] = torch.from_dlpack(polarization)
    out_array = torch.fft.fft(out_array, dim=0)
    
    # Setup chunk parameters
    chunk_params = setup_chunked_values(nx, ny, nz)
    
    # Process in chunks
    chunked_yz = torch.zeros((chunk_params[0]['chunk_size'], 2*ny, 2*nz, 3), dtype=torch.complex64, device='cuda')
    for params in chunk_params:
        chunk_slice = slice(params['chunk_start'], params['chunk_start'] + params['chunk_size'])
        chunked_yz[:params['chunk_size'], :ny, :nz] = out_array[chunk_slice]
        chunked_yz = torch.fft.fft2(chunked_yz, dim=(1,2))
        chunked_yz = cp.from_dlpack(chunked_yz)
        chunk_work_flat = cp.zeros_like(chunked_yz).ravel()
        
        total_size = params['chunk_size'] * 2 * ny * 2 * nz * 3
        grid_size = (total_size + params['threads_per_block'] - 1) // params['threads_per_block']
        
        chunked_dot_product_reduced_kernel(
            (params['blocks'],), 
            (params['threads_per_block'],),
            (interaction_matrix, chunked_yz.ravel(), chunk_work_flat,
             nx, ny, nz, params['chunk_start'], params['chunk_size'], ny)
        )
        
        cp.cuda.Stream.null.synchronize()
        chunked_yz = torch.from_dlpack(chunk_work_flat.reshape(chunked_yz.shape))
        chunked_yz = torch.fft.ifft2(chunked_yz, dim=(1,2))
        out_array[chunk_slice] = chunked_yz[:params['chunk_size'], :ny, :nz]
        chunked_yz.zero_()
        torch.cuda.synchronize()
    
    out_array = torch.fft.ifft(out_array, dim=0)[:nx]
    result = cp.from_dlpack(out_array)
    
    return result  # Returns array of shape [nx,ny,nz,3]

