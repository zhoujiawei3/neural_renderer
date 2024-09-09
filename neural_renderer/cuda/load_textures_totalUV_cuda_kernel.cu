#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)

#else
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
	if (val==0.0)
      return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif

namespace{
template <typename scalar_t>
__global__ void forward_face_index_map_UV_cuda_kernel_1(
        const scalar_t* __restrict__ faces,
        scalar_t* __restrict__ faces_inv,
        int num_faces,
        int image_size) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //i为face中的一个元素
    if (i >= num_faces) {
        return;
    }
    const int is = image_size;
    const scalar_t* face = &faces[i * 6]; //face即为当前的面
    scalar_t* face_inv_g = &faces_inv[i * 9];

    /* return if backside */
    if ((face[5] - face[1]) * (face[2] - face[0]) < (face[3] - face[1]) * (face[4] - face[0]))
        return;

    /* p[num][xy]: x, y is normalized from [-1, 1] to [0, is - 1]. */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            //0.5 * (face[3 * num + dim] * is + is - 1);
            p[num][dim] =face[num*2+dim]* is;
            // p[num][dim] = 0.5 * (face[3 * num + dim] * is + is - 1);
        }
    }
    // printf("p[0][0]: %f, p[0][1]: %f \n p[1][0]:%f p[1][1]: %f \n  p[2][0]:%f p[2][1]:%f \n", p[0][0], p[0][1],p[1][0], p[1][1], p[2][0], p[2][1]);
    //p就记载了某一个面的三个点在输出的图中的坐标点
    /* compute face_inv 计算该面片的逆矩阵，是在屏幕中作运算，可能用于后续逆变换或者纹理绘制*/
    // [x1, y1, 1]
    // [x2, y2, 1]
    // [x3, y3, 1]
    scalar_t face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    
    //打印face_inv
    // printf("face_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f\n", face_inv[0], face_inv[1], face_inv[2]);
    // printf("face_inv[3]: %f, face_inv[4]: %f, face_inv[5]: %f\n", face_inv[3], face_inv[4], face_inv[5]);
    // printf("face_inv[6]: %f, face_inv[7]: %f, face_inv[8]: %f\n", face_inv[6], face_inv[7], face_inv[8]);
    
    scalar_t face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    // printf("face_inv_denominatorbefore: %f\n face_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f \n face_inv[3]: %f face_inv[4]: %f, face_inv[5]: %f \n face_inv[6]: %f  face_inv[7]: %f, face_inv[8]: %f\n", face_inv_denominator,face_inv[0], face_inv[1], face_inv[2], face_inv[3], face_inv[4], face_inv[5], face_inv[6], face_inv[7], face_inv[8]);

   
    if (face_inv_denominator == 0) {
        printf("face_inv_denominator is zero");
    }
    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }

    // printf("face_inv_denominatorafter: %f\n face_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f \n face_inv[3]: %f  face_inv[4]: %f, face_inv[5]: %f \n face_inv[6]: %f  face_inv[7]: %f, face_inv[8]: %f\n", face_inv_denominator,face_inv[0], face_inv[1], face_inv[2], face_inv[3], face_inv[4], face_inv[5], face_inv[6], face_inv[7], face_inv[8]);

    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv_g[k] = face_inv[k];
    }
}

template <typename scalar_t>
__global__ void forward_face_index_map_UV_cuda_kernel_2(
        const scalar_t* faces,
        scalar_t* faces_inv,
        int32_t* __restrict__ face_index_map,
        scalar_t* __restrict__ weight_map,
        int num_faces,
        int image_size
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    // const int bn = i / (is * is);
    const int pn = i % (is * is);
    // const scalar_t yi = 1.0*(pn / is);
    // const scalar_t xi = 1.0*(pn % is);
    // const scalar_t yp = (2. * yi + 1 - is) / is; //会是一个浮点数
    // const scalar_t xp = (2. * xi + 1 - is) / is; //归一化到-1到1
    
    const int yi = pn / is;
    const int xi = pn % is;
    const scalar_t yp = 1.0*yi / is;
    const scalar_t xp = 1.0*xi / is;
    const scalar_t* face = &faces[ 0] - 6;
    scalar_t* face_inv = &faces_inv[ 0] - 9;
    // scalar_t depth_min = far;
    int face_index_min = -1;

    scalar_t weight_min[3];
    scalar_t face_inv_min[9];
    for (int fn = 0; fn < nf; fn++) {
        /* go to next face */
        face +=6;
        face_inv += 9;
    
        /* return if backside */
        if ((face[5] - face[1]) * (face[2] - face[0]) < (face[3] - face[1]) * (face[4] - face[0]))
            // printf("真的有背面");
            continue;
    
        /* check [py, px] is inside the face 判断xp，yp是否在面片里面,在线上也是inside*/
        if (((yp - face[1]) * (face[2] - face[0]) < (xp - face[0]) * (face[3] - face[1])) ||
            ((yp - face[3]) * (face[4] - face[2]) < (xp - face[2]) * (face[5] - face[3])) ||
            ((yp - face[5]) * (face[0] - face[4]) < (xp - face[4]) * (face[1] - face[5])))
            continue;
    
        /* compute w = face_inv * p */
        scalar_t w[3];
        w[0] = face_inv[3 * 0 + 0] * xi + face_inv[3 * 0 + 1] * yi + face_inv[3 * 0 + 2];
        w[1] = face_inv[3 * 1 + 0] * xi + face_inv[3 * 1 + 1] * yi + face_inv[3 * 1 + 2];
        w[2] = face_inv[3 * 2 + 0] * xi + face_inv[3 * 2 + 1] * yi + face_inv[3 * 2 + 2];

        // printf("w[0]: %f, w[1]: %f, w[2]: %f\nface_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f\nface_inv[3]: %f, face_inv[4]: %f, face_inv[5]: %f\nface_inv[6]: %f, face_inv[7]: %f, face_inv[8]: %f\n", w[0], w[1], w[2], face_inv[0], face_inv[1], face_inv[2], face_inv[3], face_inv[4], face_inv[5], face_inv[6], face_inv[7], face_inv[8]);

        // /* sum(w) -> 1, 0 < w < 1 */
        // printf("w[0]: %f, w[1]: %f, w[2]: %f\n", w[0], w[1], w[2]);
        // //打印face_inv
        // printf("face_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f\n", face_inv[0], face_inv[1], face_inv[2]);
        // printf("face_inv[3]: %f, face_inv[4]: %f, face_inv[5]: %f\n", face_inv[3], face_inv[4], face_inv[5]);
        // printf("face_inv[6]: %f, face_inv[7]: %f, face_inv[8]: %f\n", face_inv[6], face_inv[7], face_inv[8]);
        // printf("w[0]: %f, w[1]: %f, w[2]: %f\nface_inv[0]: %f, face_inv[1]: %f, face_inv[2]: %f\nface_inv[3]: %f, face_inv[4]: %f, face_inv[5]: %f\nface_inv[6]: %f, face_inv[7]: %f, face_inv[8]: %f\n", w[0], w[1], w[2], face_inv[0], face_inv[1], face_inv[2], face_inv[3], face_inv[4], face_inv[5], face_inv[6], face_inv[7], face_inv[8]);
        scalar_t w_sum = 0;
        // scalar_t w_min =m min()
        for (int k = 0; k < 3; k++) {
            w[k] = min(max(w[k], 0.), 1.);
            w_sum += w[k];
        }
        for (int k = 0; k < 3; k++) {
            w[k] /= w_sum;
        }
       
        face_index_min = fn;
        for (int k = 0; k < 3; k++) {
                weight_min[k] = w[k];
            }
    }
    
    /* set to global memory */
    if (0 <= face_index_min) {
        
        face_index_map[i] = face_index_min;
        for (int k = 0; k < 3; k++) {
            weight_map[3 * i + k] = weight_min[k];
        }
        
    }
}

template <typename scalar_t>
__global__ void load_textures_cuda_kernel(
        const scalar_t* image,
        const scalar_t* faces,
		scalar_t* __restrict__ textures,
		const int32_t* face_index_map,
		const scalar_t* weight_map,
		int32_t* sampling_index_map,
        scalar_t* sampling_weight_map,
        scalar_t* texture_total_weight,
        const int32_t* is_update,
        int texture_wrapping,
        bool use_bilinear,
        int num_faces,
        int image_size,
        int texture_size,
        float eps){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = pn / is;
    const int xi = pn % is;

    const int face_index = face_index_map[i];
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
        */
        const int nf = num_faces;
        const int ts = texture_size;
        const scalar_t* face = &faces[face_index * 6];
        scalar_t* texture = &textures[face_index * ts * ts * ts * 3];
        scalar_t* texture_weight= &texture_total_weight[face_index* ts * ts * ts];
        const scalar_t* pixel = &image[i * 3];
        const scalar_t* weight = &weight_map[i * 3];
        int32_t* sampling_indices = &sampling_index_map[i * 8];
        scalar_t* sampling_weights = &sampling_weight_map[i * 8];
        // printf("sampling_weights type: %s\n", sampling_weights.type().toString().c_str());
        // printf("sampling_weight_map type: %s\n", sampling_weight_map.type().toString().c_str());

        /* get texture index (float) */
        scalar_t texture_index_float[3];
        for (int k = 0; k < 3; k++) { scalar_t tif = weight[k] * (ts - 1) ;
            tif = max(tif, 0.);
            tif = min(tif, ts - 1 - eps);
            texture_index_float[k] = tif;
        }
    
        /* blend */
        // scalar_t new_pixel[3] = {0, 0, 0};
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = 1.0;                         // weight
            int texture_index_int[3];            // index in source (int)
            for (int k = 0; k < 3; k++) {
                if ((pn >> k) % 2 == 0) {
                    w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                    texture_index_int[k] = (int)texture_index_float[k];
                }
                else {
                    w *= texture_index_float[k] - (int)texture_index_float[k];
                    texture_index_int[k] = (int)texture_index_float[k] + 1;
                }
            }
    
            int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
            for (int k = 0; k < 3; k++)
                atomicAdd(&texture[isc * 3 + k], w * pixel[k]);
            atomicAdd(&texture_weight[isc], w);
                // new_pixel[k] += w * ;
            sampling_indices[pn] = isc;
            sampling_weights[pn] = w;
        }
    // printf("sampling_weights type: %s\n", sampling_weights.type().toString().c_str());

        // for (int k = 0; k < 3; k++)
        //     pixel[k] = new_pixel[k];
    }

}

// template <typename scalar_t>
// __global__ void load_textures_backward_cuda_kernel(
//         const int32_t* face_index_map,
//         scalar_t* sampling_weight_map,
//         int32_t* sampling_index_map,
//         scalar_t* texture_total_weight,
//         scalar_t* grad_out,
//         scalar_t* grad_textures,
//         size_t num_faces,
//         int image_size,
//         size_t texture_size) {
    
//     // printf("load_textures_backward_cuda_kernel\n");
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= image_size * image_size) {
//         return;
//     }
//     const int face_index = face_index_map[i];
//     if (0 <= face_index) {
//         int is = image_size;
//         int nf = num_faces;
//         int ts = texture_size;
//         int bn = i / (is * is);    // batch number [0 -> bs]
    
//         scalar_t* grad_texture = &grad_textures[(face_index) * ts * ts * ts * 3];
//         // scalar_t* texture_weight = &texture_total_weight[face_index * ts * ts * ts];
//         scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
//         int* sampling_index_map_p = &sampling_index_map[i * 8];
//         for (int pn = 0; pn < 8; pn++) {
//             scalar_t w_part = *sampling_weight_map_p++;
//             // scalar_t ts1=sampling_index_map_p/ts/ts;
//             // scalar_t ts2=(sampling_index_map_p-ts1*ts*ts)/ts;
//             // scalar_t ts3=sampling_index_map_p-ts1*ts*ts-ts2*ts;
//             scalar_t w_total = texture_total_weight[face_index*ts*ts*ts+*sampling_index_map_p++];
//             scalar_t w= w_part/ w_total;
//             int isc = *sampling_index_map_p++;
//             scalar_t* grad_texture_p = &grad_texture[isc * 3];
//             scalar_t* grad_rgb_map_p = &grad_out[i * 3];
//             for (int k = 0; k < 3; k++)
//                 atomicAdd(grad_rgb_map_p++, w * *grad_texture_p++);
//         }
//     }
// }

template <typename scalar_t>
__global__ void load_textures_backward_cuda_kernel(
        const int32_t* face_index_map,
        scalar_t* sampling_weight_map,
        int32_t* sampling_index_map,
        scalar_t* texture_total_weight,
        scalar_t* grad_out,
        scalar_t* grad_textures,
        size_t num_faces,
        int image_size,
        size_t texture_size) {
    
    // printf("load_textures_backward_cuda_kernel\n");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    if (0 <= face_index) {
        int is = image_size;
        int nf = num_faces;
        int ts = texture_size;
        int bn = i / (is * is);    // batch number [0 -> bs]
    
        scalar_t* grad_texture = &grad_textures[(face_index) * ts * ts * ts * 3];
        // scalar_t* texture_weight = &texture_total_weight[face_index * ts * ts * ts];
        scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
        int* sampling_index_map_p = &sampling_index_map[i * 8];
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w_part = *sampling_weight_map_p++;
            // scalar_t ts1=sampling_index_map_p/ts/ts;
            // scalar_t ts2=(sampling_index_map_p-ts1*ts*ts)/ts;
            // scalar_t ts3=sampling_index_map_p-ts1*ts*ts-ts2*ts;
            int isc= *sampling_index_map_p++;
            scalar_t w_total = texture_total_weight[face_index*ts*ts*ts+isc];
            
            // printf("w_total: %f  w_part: %f \n", w_total,w_part);
            scalar_t w= w_part*w_total; // / w_total;
            // printf("w_total: %f  w_part: %f w_all: %f \n", w_total,w_part,w);
            // int isc = *sampling_index_map_p++;
            scalar_t* grad_texture_p = &grad_texture[isc * 3];
            scalar_t* grad_rgb_map_p = &grad_out[i * 3];
            for (int k = 0; k < 3; k++)
                atomicAdd(grad_rgb_map_p++, w * *grad_texture_p++);
        }
    }
}

// template <typename scalar_t>
// __global__ void load_textures_backward_cuda_kernel(
//         const int32_t* face_index_map,
//         scalar_t* sampling_weight_map,
//         int32_t* sampling_index_map,
//         scalar_t* texture_total_weight,
//         scalar_t* grad_out,
//         scalar_t* grad_textures,
//         size_t num_faces,
//         int image_size,
//         size_t texture_size) {
    
//     // printf("load_textures_backward_cuda_kernel\n");
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= image_size * image_size) {
//         return;
//     }
//     const int face_index = face_index_map[i];
//     if (0 <= face_index) {
//         int is = image_size;
//         int nf = num_faces;
//         int ts = texture_size;
//         int bn = i / (is * is);    // batch number [0 -> bs]
    
//         scalar_t* grad_texture = &grad_textures[(face_index) * ts * ts * ts * 3];
//         scalar_t* texture_weight = &texture_total_weight[face_index * ts * ts * ts];
//         scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
//         int* sampling_index_map_p = &sampling_index_map[i * 8];
//         for (int pn = 0; pn < 8; pn++) {
//             scalar_t w = *sampling_weight_map_p++;
//             w= w* *texture_weight;
//             if ((*texture_weight)>1e-4){printf("texture_weight: %f\n", *texture_weight);}
                
            
//             int isc = *sampling_index_map_p++;
//             scalar_t* grad_texture_p = &grad_texture[isc * 3];
//             scalar_t* grad_rgb_map_p = &grad_out[i * 3];
//             for (int k = 0; k < 3; k++)
//                 atomicAdd(grad_rgb_map_p++, w * *grad_texture_p++);
//         }
//     }
// }


}
std::vector<at::Tensor> forward_face_index_map_UV_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor faces_inv,
        int image_size
        ) {

    // const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(0);
    const int threads = 512;
    const dim3 blocks_1 ((num_faces - 1) / threads +1);
    //这个forward_face_index_map_UV_cuda_1是找到每一个p[3][2]，即3个顶点在屏幕图片的坐标点位置，之后根据该坐标点，构建屏幕三角形的逆矩阵，放到faces中
    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_UV_cuda_1", ([&] {
      forward_face_index_map_UV_cuda_kernel_1<scalar_t><<<blocks_1, threads>>>(
          faces.data<scalar_t>(),
          faces_inv.data<scalar_t>(),
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map_UV_1: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((image_size * image_size - 1) / threads +1);
    
    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_UV_cuda_2", ([&] {
      forward_face_index_map_UV_cuda_kernel_2<scalar_t><<<blocks_2, threads>>>(
          faces.data<scalar_t>(),
          faces_inv.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          (int) num_faces,
          (int) image_size
          );
      }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map_UV_2: %s\n", cudaGetErrorString(err));
    return {face_index_map, weight_map};
}





std::vector<at::Tensor> load_textures_cuda( 
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor texture_total_weight,
        at::Tensor is_update,
        int texture_wrapping,
        int use_bilinear,
        int image_size,
        float eps
        ) {
    // const auto image_size= image.size(0);
    const auto num_faces = faces.size(0);
    const auto texture_size = textures.size(2);
    const int threads = 512;

    const dim3 blocks ((image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "load_textures_cuda", ([&] {
      load_textures_cuda_kernel<scalar_t><<<blocks, threads>>>(
          image.data<scalar_t>(),
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
		  sampling_index_map.data<int32_t>(),
		  sampling_weight_map.data<scalar_t>(),
          texture_total_weight.data<scalar_t>(),
          is_update.data<int32_t>(),
          texture_wrapping,
          use_bilinear,
		  num_faces,
          image_size,
          texture_size,
          eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_texture_sampling: %s\n", cudaGetErrorString(err));

    return {textures, sampling_index_map, sampling_weight_map,texture_total_weight};
}


at::Tensor load_textures_backward_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor texture_total_weight,
        at::Tensor grad_out,
        at::Tensor grad_textures,
        int num_faces) {
    // printf("load_textures_backward_cuda\n");
    const auto image_size = face_index_map.size(0);
    const auto texture_size = grad_textures.size(1);
    const int threads = 512;
    const dim3 blocks ((image_size * image_size - 1) / threads + 1);
    // 打印sampling_weight_map.type()
    // printf("sampling_weight_map type: %s\n", sampling_weight_map.type().toString().c_str());
    AT_DISPATCH_FLOATING_TYPES(sampling_weight_map.type(), "load_textures_backward_cuda", ([&] {
      load_textures_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
          face_index_map.data<int32_t>(),
          sampling_weight_map.data<scalar_t>(),
          sampling_index_map.data<int32_t>(),
          texture_total_weight.data<scalar_t>(),
          grad_out.data<scalar_t>(),
          grad_textures.data<scalar_t>(),
          num_faces,
          image_size,
          texture_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in backward_textures: %s\n", cudaGetErrorString(err));

    return grad_out;
}