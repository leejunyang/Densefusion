#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx) {
    // 检查维度
    TORCH_CHECK(ref.dim() == 3, "ref must be 3-dimensional");
    TORCH_CHECK(query.dim() == 3, "query must be 3-dimensional");
    
    long batch = ref.size(0);
    long dim = ref.size(1);
    long ref_nb = ref.size(2);
    long query_nb = query.size(2);
    long k = idx.size(1);

    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = idx.data_ptr<long>();

    if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
        // 使用ATen分配器创建临时距离矩阵
        auto options = ref.options().dtype(at::kFloat);
        at::Tensor dist_tensor = at::empty({ref_nb, query_nb}, options);
        float* dist_dev = dist_tensor.data_ptr<float>();

        for (int b = 0; b < batch; b++) {
            knn_device(
                ref_dev + b * dim * ref_nb,    // 当前batch的参考点
                ref_nb,                       // 每个batch的参考点数量
                query_dev + b * dim * query_nb, // 当前batch的查询点
                query_nb,                     // 每个batch的查询点数量
                dim,                          // 点的维度
                k,                            // 最近邻数量
                dist_dev,                     // 距离矩阵
                idx_dev + b * k * query_nb,   // 结果索引
                at::cuda::getCurrentCUDAStream() // 当前CUDA流
            );
        }

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA error in knn: ", cudaGetErrorString(err));
        }
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    } else {
        // CPU实现保持不变
        float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
        long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
        
        for (int b = 0; b < batch; b++) {
            knn_cpu(
                ref_dev + b * dim * ref_nb,
                ref_nb,
                query_dev + b * dim * query_nb,
                query_nb,
                dim,
                k,
                dist_dev,
                idx_dev + b * k * query_nb,
                ind_buf
            );
        }

        free(dist_dev);
        free(ind_buf);
    }

    return 1;
}