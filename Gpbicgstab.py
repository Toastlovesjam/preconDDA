import cupy as cp
import numpy as np
import torch
def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func   
precondition_kernel_reduced = cp.ElementwiseKernel(
    'raw T preconditioner, raw T y, int32 n',
    'T result',
    '''
    int idx = i / n;
    int col = i % n;
    T sum = 0;
    for (int k = 0; k < n; ++k) {
        T matrix_element;
        if (k == col) {
            // Diagonal elements
            if (k == 0) matrix_element = preconditioner[idx * 6 + 0];      // xx
            else if (k == 1) matrix_element = preconditioner[idx * 6 + 3]; // yy
            else matrix_element = preconditioner[idx * 6 + 5];             // zz
        } else {
            // Off-diagonal elements
            if ((k == 0 && col == 1) || (k == 1 && col == 0)) matrix_element = preconditioner[idx * 6 + 1]; // xy
            else if ((k == 0 && col == 2) || (k == 2 && col == 0)) matrix_element = preconditioner[idx * 6 + 2]; // xz
            else matrix_element = preconditioner[idx * 6 + 4]; // yz
        }
        sum += matrix_element * y[idx * n + k];
    }
    result = sum;
    ''',
    'matmul_kernel_reduced'
)





optimized_dot_product_reduced_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void optimized_dot_product_reduced_kernel(
        const complex<float>* interaction_matrix,
        const complex<float>* p_fft,
        complex<float>* result,
        int nx, int ny, int nz
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int total_size = 2 * nx * 2 * ny * 2 * nz * 3;
        if (i >= total_size) return;
        
        int full_idx = i / 3;
        int component = i % 3;

        int ix = full_idx / (2 * ny * 2 * nz);
        int iy = (full_idx / (2 * nz)) % (2 * ny);
        int iz = full_idx % (2 * nz);
        
        // Determine which region we're in for each dimension
        bool in_x_mirror = (ix > nx);
        bool in_y_mirror = (iy > ny);
        bool in_z_mirror = (iz > nz);
        
        // Calculate mirrored indices
        int mx = in_x_mirror ? (2*nx - ix) : ix;
        int my = in_y_mirror ? (2*ny - iy) : iy;
        int mz = in_z_mirror ? (2*nz - iz) : iz;
        
        // Clamp to reduced matrix size
        mx = min(mx, nx);
        my = min(my, ny);
        mz = min(mz, nz);
        
        int reduced_idx = mx * (ny+1) * (nz+1) + my * (nz+1) + mz;
        
        // Calculate overall sign for each off-diagonal component
        float xy_sign = ((in_x_mirror != in_y_mirror) ? -1.0f : 1.0f);
        float xz_sign = ((in_x_mirror != in_z_mirror) ? -1.0f : 1.0f);
        float yz_sign = ((in_y_mirror != in_z_mirror) ? -1.0f : 1.0f);
        
        complex<float> sum = 0;
        
        if (component == 0) {  // x component
            sum += interaction_matrix[reduced_idx * 6 + 0] * p_fft[full_idx * 3];
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[full_idx * 3 + 1];
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[full_idx * 3 + 2];
        } 
        else if (component == 1) {  // y component
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[full_idx * 3];
            sum += interaction_matrix[reduced_idx * 6 + 3] * p_fft[full_idx * 3 + 1];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[full_idx * 3 + 2];
        } 
        else {  // z component
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[full_idx * 3];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[full_idx * 3 + 1];
            sum += interaction_matrix[reduced_idx * 6 + 5] * p_fft[full_idx * 3 + 2];
        }
        
        result[i] = sum;
    }
    ''', 'optimized_dot_product_reduced_kernel')

optimized_dot_product_reduced_kernel_odd = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void optimized_dot_product_reduced_kernel_odd(
        const complex<float>* interaction_matrix,
        const complex<float>* p_fft,
        complex<float>* result,
        int nx, int ny, int nz,
        int full_nx, int full_ny, int full_nz
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int total_size = full_nx * full_ny * full_nz * 3;
        if (i >= total_size) return;
        
        int full_idx = i / 3;
        int component = i % 3;

        int ix = full_idx / (full_ny * full_nz);
        int iy = (full_idx / full_nz) % full_ny;
        int iz = full_idx % full_nz;
        
        // Determine which region we're in for each dimension
        bool in_x_mirror = (ix >= nx);
        bool in_y_mirror = (iy >= ny);
        bool in_z_mirror = (iz >= nz);
        
        // Calculate mirrored indices
        int mx = in_x_mirror ? (nx - (ix - nx) - 1) : ix;
        int my = in_y_mirror ? (ny - (iy - ny) - 1) : iy;
        int mz = in_z_mirror ? (nz - (iz - nz) - 1) : iz;

        
        int reduced_idx = mx * ny * nz + my * nz + mz;
        
        // Calculate overall sign for each off-diagonal component
        float xy_sign = ((in_x_mirror != in_y_mirror) ? -1.0f : 1.0f);
        float xz_sign = ((in_x_mirror != in_z_mirror) ? -1.0f : 1.0f);
        float yz_sign = ((in_y_mirror != in_z_mirror) ? -1.0f : 1.0f);
        
        complex<float> sum = 0;
        
        if (component == 0) {  // x component
            sum += interaction_matrix[reduced_idx * 6 + 0] * p_fft[full_idx * 3];
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[full_idx * 3 + 1];
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[full_idx * 3 + 2];
        } 
        else if (component == 1) {  // y component
            sum += xy_sign * interaction_matrix[reduced_idx * 6 + 1] * p_fft[full_idx * 3];
            sum += interaction_matrix[reduced_idx * 6 + 3] * p_fft[full_idx * 3 + 1];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[full_idx * 3 + 2];
        } 
        else {  // z component
            sum += xz_sign * interaction_matrix[reduced_idx * 6 + 2] * p_fft[full_idx * 3];
            sum += yz_sign * interaction_matrix[reduced_idx * 6 + 4] * p_fft[full_idx * 3 + 1];
            sum += interaction_matrix[reduced_idx * 6 + 5] * p_fft[full_idx * 3 + 2];
        }
        
        result[i] = sum;
    }
    ''', 'optimized_dot_product_reduced_kernel_odd')


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

final_reduction_kernel2 = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void final_reduction_kernel(const complex<float>* partial_results, 
                          complex<float>* zeta_eta_results,
                          int num_blocks) {
    const int tid = threadIdx.x;
    
    if(tid >= 6) return;  // Only need 6 threads per block
    
    complex<float> sum(0.0f, 0.0f);
    for(int i = 0; i < num_blocks; i++) {
        sum += partial_results[i * 6 + tid];
    }
    
    __shared__ complex<float> final_sums[6];
    final_sums[tid] = sum;
    __syncthreads();
    
    if(tid == 0) {
        complex<float> sum_y_squared = final_sums[0];
        complex<float> sum_t_squared = final_sums[1];
        complex<float> sum_y_t = final_sums[2];
        complex<float> sum_t_y = final_sums[3];
        complex<float> sum_y_s = final_sums[4];
        complex<float> sum_t_s = final_sums[5];
        
        complex<float> denom = sum_t_squared * sum_y_squared - sum_y_t * sum_t_y;
        complex<float> zeta = (sum_y_squared * sum_t_s - sum_y_s * sum_t_y) / denom;
        complex<float> eta = (sum_t_squared * sum_y_s - sum_y_t * sum_t_s) / denom;
        
        zeta_eta_results[0] = zeta;
        zeta_eta_results[1] = eta;
    }
}
''', 'final_reduction_kernel')

partial_dots_kernel2 = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void partial_dots_kernel(const complex<float>* y, const complex<float>* t, const complex<float>* s,
                        complex<float>* partial_results, int n) {
    __shared__ complex<float> sdata[512];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    int idx = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    complex<float> sum_y_squared(0.0f, 0.0f);
    complex<float> sum_t_squared(0.0f, 0.0f);
    complex<float> sum_y_t(0.0f, 0.0f);
    complex<float> sum_t_y(0.0f, 0.0f);
    complex<float> sum_y_s(0.0f, 0.0f);
    complex<float> sum_t_s(0.0f, 0.0f);
    
    while (idx < n) {
        complex<float> yi = y[idx];
        complex<float> ti = t[idx];
        complex<float> si = s[idx];
        
        sum_y_squared += conj(yi) * yi;
        sum_t_squared += conj(ti) * ti;
        sum_y_t += conj(yi) * ti;
        sum_t_y += conj(ti) * yi;
        sum_y_s += conj(yi) * si;
        sum_t_s += conj(ti) * si;
        
        idx += stride;
    }
    
    sdata[tid] = sum_y_squared;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 0] = sdata[0];
    
    sdata[tid] = sum_t_squared;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 1] = sdata[0];
    
    sdata[tid] = sum_y_t;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 2] = sdata[0];
    
    sdata[tid] = sum_t_y;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 3] = sdata[0];
    
    sdata[tid] = sum_y_s;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 4] = sdata[0];
    
    sdata[tid] = sum_t_s;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0) partial_results[bid * 6 + 5] = sdata[0];
}
''', 'partial_dots_kernel')

def calculate_zeta_eta_twostep(y, t, s):
    n = y.size
    
    block_size = 512
    num_blocks = min(1024, (n + block_size - 1) // block_size)
    
    partial_results = cp.zeros((num_blocks, 6), dtype=cp.complex64)
    
    partial_dots_kernel2(
        (num_blocks,), (block_size,),
        (y, t, s, partial_results.reshape(-1), n)
    )
    
    final_results = cp.zeros(2, dtype=cp.complex64)
    
    final_reduction_kernel2(
        (1,), (6,),
        (partial_results.reshape(-1), final_results, num_blocks)
    )
    
    return final_results[0], final_results[1]


vdot_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void vdot_kernel(const complex<float>* __restrict__ a, 
                 const complex<float>* __restrict__ b,
                 complex<float>* __restrict__ block_results, 
                 int n) {
    __shared__ complex<float> sum[256];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    int idx = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    complex<float> local_sum(0.0f, 0.0f);
    
    while(idx < n) {
        complex<float> a_val = a[idx];
        complex<float> b_val = b[idx];
        local_sum += conj(a_val) * b_val;
        idx += stride;
    }
    
    sum[tid] = local_sum;
    __syncthreads();
    
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        block_results[bid] = sum[0];
    }
}
''', 'vdot_kernel')

sum_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void sum_kernel(const complex<float>* block_results, complex<float>* result, 
                int n_blocks) {
    __shared__ complex<float> sum[32];
    const int tid = threadIdx.x;
    
    if(tid < n_blocks) {
        sum[tid] = block_results[tid];
    } else {
        sum[tid] = complex<float>(0.0f, 0.0f);
    }
    __syncthreads();
    
    if(tid < 16) { sum[tid] += sum[tid + 16]; } __syncthreads();
    if(tid < 8)  { sum[tid] += sum[tid + 8];  } __syncthreads();
    if(tid < 4)  { sum[tid] += sum[tid + 4];  } __syncthreads();
    if(tid < 2)  { sum[tid] += sum[tid + 2];  } __syncthreads();
    if(tid < 1)  { sum[tid] += sum[tid + 1];  }
    
    if(tid == 0) {
        result[0] = sum[0];
    }
}
''', 'sum_kernel')

def inner_product(a, b):
    n = a.size
    
    block_size = 256
    num_blocks = min(32, max(8, (n + block_size - 1) // block_size))
    
    block_results = cp.zeros(num_blocks, dtype=cp.complex64)
    result = cp.zeros(1, dtype=cp.complex64)
    
    vdot_kernel((num_blocks,), (block_size,),
                (a, b, block_results, n))
    
    sum_kernel((1,), (32,),
               (block_results, result, num_blocks))
    
    return result[0]

interaction_combine2 = cp.ElementwiseKernel(
    'T alpha_inv, T x, T ifft_result',
    'T output',
    'output = (alpha_inv * x - ifft_result)',
    'interaction_combine'
)

@cp.fuse()
def interaction_combine(alpha_inv, x, ifft_result):
    return alpha_inv * x - ifft_result


@cp.fuse
def compute_s(r0, alpha, v):
    return r0 - alpha * v

@cp.fuse
def compute_y(s_prev, s, alpha, w):
    return s_prev - s - alpha * w

@cp.fuse
def compute_r0_initial(s, zeta, t):
    return s - zeta * t

@cp.fuse
def compute_u_initial(zeta, v):
    return zeta * v

@cp.fuse
def compute_z_initial(zeta, s):
    return zeta * s

@cp.fuse
def compute_u(zeta, v, eta, s_prev, r0, beta, u_prev):
    return zeta * v + eta * (s_prev - r0 + beta * u_prev)

@cp.fuse
def compute_z(zeta, r0, eta, z_prev, alpha, u):
    return zeta * r0 + eta * z_prev - alpha * u

@cp.fuse
def compute_r0(s, eta, y, zeta, t):
    return s - eta * y - zeta * t

@cp.fuse
def update_p_current(p_current, alpha, phat, zhat):
    return p_current + alpha * phat + zhat
    
@cp.fuse
def compute_w(t, beta, v):
    return t + beta * v
    
@cp.fuse
def compute_p(r0, beta, p_prev, u):
    return r0 + beta * (p_prev - u)
    
@cp.fuse()
def norm_fused(a):
    return cp.sqrt(cp.sum(cp.abs(a)**2))
    
@cp.fuse()
def inner_product_fused(a, b):
    return cp.sum(cp.conj(a) * b)

def apply_preconditioner_loops(inverse_matrices, p, nx, ny, nz, precon):    

    # Get target dimensions
    target_x = inverse_matrices.shape[0]
    target_y = inverse_matrices.shape[1]
    
    # Initial transformation, preserving p
    result = cp.reshape(p, (nx, ny, nz, 3))
    if precon==False:
        return result    
    # Pad if needed
    if nx < target_x or ny < target_y:
        result = cp.pad(result, 
                       ((0, target_x - nx), 
                        (0, target_y - ny),
                        (0, 0),
                        (0, 0)),
                       mode='constant')

    # Transform for FFT
    result = cp.transpose(result, (0, 1, 3, 2))
    result = cp.reshape(result, (result.shape[0], result.shape[1], -1))

    # Forward FFT
    result = torch.from_dlpack(result)
    result = torch.fft.fft2(result, dim=(0, 1))
    result = cp.from_dlpack(result)

    # Matrix multiplication
    result = cp.matmul(inverse_matrices, result[:, :, :, cp.newaxis])
    result = result.squeeze(-1)

    # Inverse FFT
    result = torch.from_dlpack(result)
    result = torch.fft.ifft2(result, dim=(0, 1))
    result = cp.from_dlpack(result)

    # Extract and reshape to final form
    result = result[:nx, :ny]
    result = cp.reshape(result, (nx, ny, 3, nz))
    result = cp.transpose(result, (0, 1, 3, 2))
    
    return result





def fft_matvec4(in_array, out_array, matrix, mask, chunk_params, nx, ny, nz, precon):
    # Calculate full matrix shape
    if precon==False:
        return in_array.ravel()
    full_nx, full_ny, full_nz = [(d-1)*2 for d in matrix.shape[:3]]
    
    in_array = in_array.reshape(nx, ny, nz, 3)
    # Pad only x dimension initially
    out_array = torch.zeros((full_nx, ny, nz, 3), dtype=torch.complex64, device='cuda')
    out_array[:nx] = torch.from_dlpack(in_array)
    out_array = torch.fft.fft(out_array, dim=0)
    
    # Create chunked array with full y,z dimensions for FFT
    chunked_yz = torch.zeros((chunk_params[0]['chunk_size'], full_ny, full_nz, 3), dtype=torch.complex64, device='cuda')
    
    for params in chunk_params:
        chunk_slice = slice(params['chunk_start'], params['chunk_start'] + params['chunk_size'])
        
        # Pad y and z dimensions for this chunk
        chunked_yz[:params['chunk_size'], :ny, :nz] = out_array[chunk_slice]
        chunked_yz = torch.fft.fft2(chunked_yz, dim=(1,2))
        
        chunked_yz = cp.from_dlpack(chunked_yz)
        chunk_work_flat = cp.zeros_like(chunked_yz).ravel()
        total_size = params['chunk_size'] * full_ny * full_nz * 3
        grid_size = (total_size + params['threads_per_block'] - 1) // params['threads_per_block']
        
        chunked_dot_product_reduced_kernel(
            (params['blocks'],), 
            (params['threads_per_block'],),
            (matrix, chunked_yz.ravel(), chunk_work_flat,
             full_nx//2, full_ny//2, full_nz//2, params['chunk_start'], params['chunk_size'], full_ny//2)
        )
        cp.cuda.Stream.null.synchronize()
        
        chunked_yz = torch.from_dlpack(chunk_work_flat.reshape(chunked_yz.shape))
        # IFFT y and z, then unslice y and z
        chunked_yz = torch.fft.ifft2(chunked_yz, dim=(1,2))
        out_array[chunk_slice] = chunked_yz[:params['chunk_size'], :ny, :nz]
        chunked_yz.zero_()
        torch.cuda.synchronize()
    
    # IFFT x and unslice x
    out_array = torch.fft.ifft(out_array, dim=0)[:nx]
    
    result = cp.from_dlpack(out_array) * mask
    result = result.ravel()
    in_array = in_array.ravel()
    return result






def fft_matvec3(in_array, out_array, preconditioner, mask, chunk_params, nx, ny, nz, precon):
    if precon==False:
        return in_array.ravel()
    # Calculate full preconditioner shape
    full_nx, full_ny, full_nz = [(d-1)*2 for d in preconditioner.shape[:3]]
    
    # Create padded array
    padded = torch.zeros((full_nx, full_ny, full_nz, 3), dtype=torch.complex64, device='cuda')
    
    # Place in_array at the start
    padded[:nx, :ny, :nz] = torch.from_dlpack(in_array).reshape(nx, ny, nz, 3)

    y = torch.fft.fftn(padded, dim=(0, 1, 2))
    y_cp = cp.from_dlpack(y.reshape(-1))

    kernel_output = cp.empty_like(y_cp)
    
    threads_per_block = 256
    blocks = (y_cp.size + threads_per_block - 1) // threads_per_block
    
    optimized_dot_product_reduced_kernel(
        (blocks,), (threads_per_block,),
        (preconditioner.astype(cp.complex64).ravel(),
         y_cp,
         kernel_output,
         full_nx//2, full_ny//2, full_nz//2)
    )

    y = torch.from_dlpack(kernel_output.reshape(padded.shape))
    y = torch.fft.ifftn(y, dim=(0, 1, 2))

    # Extract the result from start
    result = y[:nx, :ny, :nz]
    result *= torch.from_dlpack(mask)

    return cp.from_dlpack(result.ravel())





def apply_precond_with_kernel(x, y, preconditioner, mask, chunk_params, nx, ny, nz):
    x = x.reshape(nx, ny, nz, 3)

    sx = (preconditioner.shape[0] - x.shape[0])//2
    sy = (preconditioner.shape[1] - x.shape[1])//2
    sz = (preconditioner.shape[2] - x.shape[2])//2
    
    padded = cp.zeros((*preconditioner.shape[:3], 3), dtype=cp.complex64)
    padded[sx:sx+x.shape[0], sy:sy+x.shape[1], sz:sz+x.shape[2]] = x

    y = cp.fft.fftn(padded, axes=(0, 1, 2))

    y = precondition_kernel_reduced(preconditioner.astype(cp.complex64).reshape(-1, 6), 
                                    y.reshape(-1, 3), 
                                    3, 
                                    cp.empty_like(y.reshape(-1, 3), dtype=cp.complex64))

    y = y.reshape(padded.shape)
    y = cp.fft.ifftn(y, axes=(0, 1, 2))
    
    y = y[sx:sx+x.shape[0], sy:sy+x.shape[1], sz:sz+x.shape[2]]

    y *= mask

    x = x.ravel()
    y = y.ravel()

    return y



    
def fft_matvec2(in_array, out_array, matrix, mask, chunk_params, nx, ny, nz):
    in_array = in_array.reshape(nx, ny, nz, 3)
    out_array = torch.zeros((nx*2, ny, nz, 3), dtype=torch.complex64, device='cuda')
    start_idx_x = (nx + 1) // 2
    out_array[start_idx_x:start_idx_x+nx] = torch.from_dlpack(in_array)
    out_array = torch.fft.fft(out_array, dim=0)
    
    chunked_yz = torch.zeros((chunk_params[0]['chunk_size'], 2*ny, 2*nz, 3), dtype=torch.complex64, device='cuda')
    start_idx_y = (ny + 1) // 2
    start_idx_z = (nz + 1) // 2
    
    for params in chunk_params:
        chunk_slice = slice(params['chunk_start'], params['chunk_start'] + params['chunk_size'])
        chunked_yz[:params['chunk_size'], start_idx_y:start_idx_y+ny, start_idx_z:start_idx_z+nz] = out_array[chunk_slice]
        chunked_yz = torch.fft.fft2(chunked_yz, dim=(1,2))
        chunked_yz = cp.from_dlpack(chunked_yz)
        chunk_work_flat = cp.zeros_like(chunked_yz).ravel()
        total_size = params['chunk_size'] * 2 * ny * 2 * nz * 3
        grid_size = (total_size + params['threads_per_block'] - 1) // params['threads_per_block']
        chunked_dot_product_reduced_kernel(
            (params['blocks'],), 
            (params['threads_per_block'],),
            (matrix, chunked_yz.ravel(), chunk_work_flat,
             nx, ny, nz, params['chunk_start'], params['chunk_size'], ny)
        )
        chunked_yz = torch.from_dlpack(chunk_work_flat.reshape(chunked_yz.shape))
        chunked_yz = torch.fft.ifft2(chunked_yz, dim=(1,2))
        out_array[chunk_slice] = chunked_yz[:params['chunk_size'], start_idx_y:start_idx_y+ny, start_idx_z:start_idx_z+nz]
        chunked_yz.zero_()
    
    out_array = torch.fft.ifft(out_array, dim=0)
    result = cp.from_dlpack(out_array[start_idx_x:start_idx_x+nx]) * mask
    result = result.ravel()
    in_array = in_array.ravel()
    return result    

def fft_matvec(in_array, out_array, matrix, mask, chunk_params, nx, ny, nz):
    in_array = in_array.reshape(nx, ny, nz, 3)
    out_array = torch.zeros((nx*2, ny, nz, 3), dtype=torch.complex64, device='cuda')
    out_array[:nx] = torch.from_dlpack(in_array)
    out_array = torch.fft.fft(out_array, dim=0)
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
            (matrix, chunked_yz.ravel(), chunk_work_flat,
             nx, ny, nz, params['chunk_start'], params['chunk_size'], ny)
        )
        cp.cuda.Stream.null.synchronize()
        chunked_yz = torch.from_dlpack(chunk_work_flat.reshape(chunked_yz.shape))
        chunked_yz = torch.fft.ifft2(chunked_yz, dim=(1,2))
        out_array[chunk_slice] = chunked_yz[:params['chunk_size'], :ny, :nz]
        chunked_yz.zero_()
        torch.cuda.synchronize()
    out_array = torch.fft.ifft(out_array, dim=0)[:nx]
    result = cp.from_dlpack(out_array)* mask
    result = result.ravel()
    in_array=in_array.ravel()
    return result

optimized_extract_kernel = cp.RawKernel(r'''
#include <cuComplex.h>
extern "C" __global__
void optimized_extract(const cuFloatComplex* __restrict__ input, 
                      cuFloatComplex* __restrict__ output,
                      const int* __restrict__ indices, 
                      int num_nonzero) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid < num_nonzero) {
        output[tid] = input[indices[tid]];
    }
}
''', 'optimized_extract')

optimized_place_kernel = cp.RawKernel(r'''
#include <cuComplex.h>
extern "C" __global__
void optimized_place(const cuFloatComplex* __restrict__ input, 
                    cuFloatComplex* __restrict__ output,
                    const int* __restrict__ indices, 
                    int num_nonzero) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid < num_nonzero) {
        output[indices[tid]] = input[tid];
    }
}
''', 'optimized_place')

def extract_to_1d(full_array, indices, num_nonzero):
    compressed_data = cp.zeros(num_nonzero, dtype=cp.complex64)
    
    threadsperblock = 256
    blockspergrid = min(65535, (num_nonzero + threadsperblock - 1) // threadsperblock)
    
    optimized_extract_kernel((blockspergrid,), (threadsperblock,),
                           (full_array.ravel(), compressed_data, indices, num_nonzero))
    
    return compressed_data

def place_to_4d(compressed_data, indices, num_nonzero, shape):
    full_array = cp.zeros(shape[0] * shape[1] * shape[2] * shape[3], dtype=cp.complex64)
    
    threadsperblock = 256
    blockspergrid = min(65535, (num_nonzero + threadsperblock - 1) // threadsperblock)
    
    optimized_place_kernel((blockspergrid,), (threadsperblock,),
                         (compressed_data, full_array, indices, num_nonzero))
    
    return full_array.reshape(shape)



def interaction_mult(in_array, out_array, interaction_matrix, alpha_inv, mask, chunk_params, nx, ny, nz):
    out_array = fft_matvec(in_array, out_array, interaction_matrix, mask, chunk_params, nx, ny, nz)
    return interaction_combine(alpha_inv, in_array, out_array)

def setup_chunked_values(nx, ny, nz):
    total_length = 2*nx
    target_chunk_size = 40
    min_chunk_size = 10
    
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
import random
@free_gpu_memory 
def solve_gpbicgstab(
    grid_size, 
    inv_alpha, 
    interaction_matrix, 
    preconditioner, 
    E_inc, 
    mask, 
    ratio, 
    max_iter, 
    target=1e-5, 
    initial_p=None, 
    is_2d=False,
    precon=True
):
    """
    Solve a system using a GP-BiCGSTAB-like iterative method.

    Parameters
    ----------
    grid_size          : tuple (nx, ny, nz)
        The dimensions of the grid.
    inv_alpha         : array-like
        1D or broadcastable array for the inverse of alpha.
    interaction_matrix : array-like
        Matrix used for the interaction multiplication (frequency-space or real-space).
    preconditioner    : array-like
        The preconditioner used by matvec_func or apply_preconditioner_loops.
    E_inc             : array-like
        The incident field (or right-hand side) plus any initial guess or offsets.
    mask              : array-like
        Binary or float mask that indicates which grid points to include.
    ratio             : float
        Ratio used to trigger a reset if current norm < previous_lowest / ratio.
    max_iter          : int
        Maximum number of iterations.
    target            : float
        Convergence target as a fraction of the initial norm.
    initial_p         : unused
        Placeholder for an initial guess (not used here).
    is_2d             : bool
        If True, use a 2D solver path. Otherwise, the 3D approach.
    precon            : bool
        Whether to use preconditioner.

    Returns
    -------
    (iters, solution, lowest_norm, first_norm)
    """
    nx, ny, nz = grid_size

    # Convert all inputs to CuPy single precision
    mask = cp.asarray(mask, dtype=cp.float32)
    mask = mask[..., None]

    r0 = cp.asarray(E_inc, dtype=cp.complex64) * mask
    r0 = r0.ravel()

    inv_alpha = cp.asarray(inv_alpha, dtype=cp.complex64)
    inv_alpha = cp.repeat(inv_alpha, 3)
    print('inv alpha')
    print(inv_alpha.shape)
    
    interaction_matrix = cp.asarray(interaction_matrix, dtype=cp.complex64)
    preconditioner = cp.asarray(preconditioner, dtype=cp.complex64)

    print('pre mean')
    print(cp.mean(preconditioner))

    # Setup chunk parameters
    chunk_params = setup_chunked_values(nx, ny, nz)

    preconditioner_shape = [((d - 1)) for d in preconditioner.shape[:3]]
    print(preconditioner_shape)
    chunk_params2 = setup_chunked_values(*preconditioner_shape)

    cp.get_default_memory_pool().free_all_blocks()

    # Initialize dependent vectors
    r0_hat = r0.copy()
    p = r0.copy()
    p_current = cp.zeros_like(r0)
    v = cp.zeros_like(r0)
    t = cp.zeros_like(r0)
    s = cp.zeros_like(r0)
    phat = cp.zeros_like(r0)
    shat = cp.zeros_like(r0)
    zhat = cp.zeros_like(r0)
    u = cp.zeros_like(r0)
    z = cp.zeros_like(r0)
    w = cp.zeros_like(r0)
    y = cp.zeros_like(r0)
    s_prev = cp.zeros_like(r0)

    # Scalar values 
    initial_norm = cp.linalg.norm(r0).get()
    rho_1 = inner_product(r0_hat, r0)
    rho_0 = rho_1.copy()
    alpha = cp.array(0, dtype=r0.dtype)
    beta = cp.array(0, dtype=r0.dtype)
    zeta = cp.array(0, dtype=r0.dtype)
    eta = cp.array(0, dtype=r0.dtype)

    # Tracking variables
    lowest_norm = initial_norm
    current_norm = lowest_norm
    previous_lowest = lowest_norm
    first_norm = 0

    # For best solution tracking (store p_current, with no reversion to older states)
    best_p_current = cp.zeros_like(r0)

    # Calculate target convergence norms
    target_norms = initial_norm * target
    print('target norm')
    print(target_norms)

    # Reset detection parameters
    resetcount = 0
    reset_tracker = 0
    was_reset = False 
    last_change = 0

    # Choose matvec function
    matvec_func = fft_matvec3

    for i in range(max_iter):
        # Preconditioning
        if not is_2d:
            #p = place_to_4d(p, nonzero_indices, num_nonzero, (nx, ny, nz, 3))  # Example toggled line
            phat = matvec_func(p, phat, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        else:
            #p = place_to_4d(p, nonzero_indices, num_nonzero, (nx, ny, nz, 3))
            phat = apply_preconditioner_loops(preconditioner, p, nx, ny, nz, precon)
            phat *= mask

        # Matrix-vector product with the interaction matrix
        v = interaction_mult(phat.ravel(), v, interaction_matrix, inv_alpha, mask, chunk_params, nx, ny, nz)
        #v = extract_to_1d(v, nonzero_indices, num_nonzero)  # Example toggled line
        #p = extract_to_1d(p, nonzero_indices, num_nonzero)  # Example toggled line
        
        alpha = (rho_1 / inner_product(r0_hat, v))
        s = compute_s(r0, alpha, v)
        y = compute_y(s_prev, s, alpha, w)

        # Precondition s
        if not is_2d:
            #s = place_to_4d(s, nonzero_indices, num_nonzero, (nx, ny, nz, 3))  # Example toggled line
            shat = matvec_func(s, shat, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        else:
            #s = place_to_4d(s, nonzero_indices, num_nonzero, (nx, ny, nz, 3))
            shat = apply_preconditioner_loops(preconditioner, s, nx, ny, nz, precon)
            shat *= mask

        # Matrix-vector product for t
        t = interaction_mult(shat.ravel(), t, interaction_matrix, inv_alpha, mask, chunk_params, nx, ny, nz)
        #t = extract_to_1d(t, nonzero_indices, num_nonzero)  # Example toggled line
        #s = extract_to_1d(s, nonzero_indices, num_nonzero)  # Example toggled line

        # Special case for the first iteration
        if (i == 0):# or i%2==0:  # 10% chance to reinitialize
            zeta = inner_product(t, s) / inner_product(t, t)
            eta = cp.array(0)
            u = compute_u_initial(zeta, v)
            z = compute_z_initial(zeta, s)
            r0 = compute_r0_initial(s, zeta, t)
        else:
            if was_reset:  # If we just did a reset, re-initialize
                zeta = inner_product(t, s) / inner_product(t, t)
                eta = cp.array(0)
                u = compute_u_initial(zeta, v)
                z = compute_z_initial(zeta, s)
                r0 = compute_r0_initial(s, zeta, t)
            else:
                zeta, eta = calculate_zeta_eta_twostep(y, t, s)
                u = compute_u(zeta, v, eta, s_prev, r0, beta, u)
                z = compute_z(zeta, r0, eta, z, alpha, u)
                r0 = compute_r0(s, eta, y, zeta, t)
    
        p_current = update_p_current(p_current, alpha, p, z)

        if i==1:  
            # Step 1: Apply preconditioner to p_current
            phat = matvec_func(p_current, phat, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
            
            # Store previous r0 for comparison
            r0_old = r0.copy()
            
            # Step 2: Compute new r0
            r0 = (cp.asarray(E_inc, dtype=cp.complex64) * mask).ravel() - interaction_mult(phat.ravel(), r0, interaction_matrix, inv_alpha, mask, chunk_params, nx, ny, nz)
            
            # Calculate core comparison metrics
            diff = r0 - r0_old
            #print(f"\nIteration {i} Comparison:")
            #print(f"Current L2 norm:  {norm_fused(r0).get():.2e}")
            #print(f"Previous L2 norm: {norm_fused(r0_old).get():.2e}")
            #print(f"Change in L2:     {norm_fused(diff).get():.2e}")
        
        current_norm = norm_fused(r0).get()  # Convert to NumPy
        
        # Record the norm at the first iteration
        if i == 0:
            first_norm = current_norm
        
        # Print progress every ~10% of max_iter or at least once

        progress_interval = max(1, max_iter // 10)
        if i % progress_interval == 0:
            relative_current = current_norm / (initial_norm )
            relative_lowest = lowest_norm / (initial_norm )
            print(
                f'\n{"-"*120}\n'
                f'Iter: {i:4d}  |  Resets: {resetcount:3d}  |  '
                f'Target: {initial_norm*target:.2e}  |  '
                f'Current: {current_norm:.2e} (Ratio: {relative_current:.2e})  |  '
                f'Best: {lowest_norm:.2e} (Ratio: {relative_lowest:.2e})'
            )

        last_change += 1
        
        # Update best known solution if improved
        if (not np.isnan(current_norm)) and (np.isnan(lowest_norm) or current_norm < lowest_norm):
            last_change = 0
            lowest_norm = current_norm
            best_p_current = p_current.copy()  # stays on CuPy
        
        # Check for NaN
        if np.isnan(current_norm)or last_change>=5000:
            print(f"NAN detected after {i + 1} iterations.")
            break
        
                # Check for convergence
        elif current_norm < (target * initial_norm):
            print(f"\nConvergence achieved after {i + 1} iterations.")
            break
        elif last_change >=5000:
            print(f"\n Stagnation detected after {i + 1} iterations.")
            break            
        # Prepare for possible reset
        reset = False

        # Save old rho, compute new rho
        rho_0 = rho_1.copy()    
        rho_1 = inner_product(r0_hat, r0)
        #(i<=5 and i >0 and current_norm<previous_lowest)
        #(i <= 50 and current_norm<previous_lowest and i%2)
        # Ratio-based reset AND forced reset at i == 1
        # If ratio=0, do not do either reset (including i == 1).
        if ratio != 0:
            if i==2 or (i > 2 and current_norm < previous_lowest / ratio):
                print(f"\nRESET TRIGGERED:")
                print(f"  Iteration: {i}")
                print(f"  Reason: {'First iteration (i=1)' if i==2 else 'Ratio threshold'}")
                print(f"  Current value: {current_norm}")
                print(f"  Previous lowest was: {previous_lowest}")
                print(f"  Reset threshold was: {previous_lowest / ratio}")
                reset = True
                previous_lowest = current_norm
                print(f"  Next trigger value will be: {previous_lowest / ratio}")   
                last_change = 0


        was_reset = False
        if reset:
            if i % 100 != 0:  # Every 100 iterations
                phat = matvec_func(p_current, phat, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
                r0 = (cp.asarray(E_inc, dtype=cp.complex64) * mask).ravel() - interaction_mult(phat.ravel(), r0, interaction_matrix, inv_alpha, mask, chunk_params, nx, ny, nz)
            # Perform a reset
            r0_hat = r0.copy()
            beta *= 0
            w = cp.zeros_like(w)
            s_prev = s.copy()
            was_reset = True
            p = r0.copy()
            resetcount += 1
        else:
            # No reset: continue the usual update sequence
            beta = alpha / zeta * rho_1 / rho_0
            w = compute_w(t, beta, v)
            s_prev = s.copy()
            p = compute_p(r0, beta, p, u)

    # First check for NaN or failure conditions that triggered the break
    # First check for NaN or failure conditions that triggered the break
    if np.isnan(current_norm) or last_change >= 5000:
        if not np.isnan(lowest_norm):
            percent_of_initial_norm = (lowest_norm / initial_norm) * 100
            print(
                f"Final result is NaN. Returning the value associated with the "
                f"lowest norm ({percent_of_initial_norm:.2f}% of initial_norm)."
            )
        else:
            print("Final result is NaN and lowest norm is also NaN. Returning NaN.")
        best_p_current = matvec_func(best_p_current, t, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        best_p_current = best_p_current.reshape(nx, ny, nz, 3)
        return -(i+1), best_p_current, lowest_norm/initial_norm, first_norm/initial_norm
    
    # Safety check: if we're below max_iter-1 but haven't converged, something went wrong
    elif i < (max_iter - 1) and current_norm >= target * initial_norm:
        print(
            f"WARNING: Iteration terminated before max_iter but didn't converge. "
            f"Likely missed failure condition. Treating as failure at iteration {i+1}."
        )
        best_p_current = matvec_func(best_p_current, t, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        best_p_current = best_p_current.reshape(nx, ny, nz, 3)
        return -(i+1), best_p_current, lowest_norm/initial_norm, first_norm/initial_norm
    
    # Then check for normal non-convergence
    elif current_norm >= target * initial_norm:
        percent_of_initial_norm = (lowest_norm / initial_norm) * 100
        print(
            f"Did not converge. Returning {percent_of_initial_norm:.2f}% of initial_norm "
            "instead of target."
        )
        best_p_current = matvec_func(best_p_current, t, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        best_p_current = best_p_current.reshape(nx, ny, nz, 3)
        return max_iter, best_p_current, lowest_norm/initial_norm, first_norm/initial_norm  # Return max_iter instead of i+1
    
    # Finally, handle successful convergence
    else:
        print(f"Convergence achieved after {i+1} iterations.")
        p_current = matvec_func(p_current, t, preconditioner, mask, chunk_params2, nx, ny, nz, precon)
        p_current = p_current.reshape(nx, ny, nz, 3)
        return i+1, p_current, lowest_norm/initial_norm, first_norm/initial_norm
