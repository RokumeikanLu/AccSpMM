/**
 * @file mma_tf32.cu
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#include "mmio.hpp"
#include "tf32_comp.hpp"

namespace {

inline void CheckCublasStatus(cublasStatus_t status, const char* step)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error %d at %s\n", static_cast<int>(status), step);
        exit(EXIT_FAILURE);
    }
}

inline void RowMajorToColumnMajor(const MAT_VAL_TYPE* src,
                                  MAT_VAL_TYPE*       dst,
                                  vint                rows,
                                  vint                cols)
{
    for (vint r = 0; r < rows; ++r) {
        for (vint c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

inline void ColumnMajorToRowMajor(const MAT_VAL_TYPE* src,
                                  MAT_VAL_TYPE*       dst,
                                  vint                rows,
                                  vint                cols)
{
    for (vint r = 0; r < rows; ++r) {
        for (vint c = 0; c < cols; ++c) {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
}

struct ErrorStats {
    double total_abs_error;
    double max_abs_error;
};

ErrorStats ComputeErrorStats(const std::vector<MAT_VAL_TYPE>& reference,
                             const std::vector<MAT_VAL_TYPE>& test)
{
    ErrorStats stats{0.0, 0.0};
    if (reference.size() != test.size()) {
        printf("Size mismatch when computing error: reference=%zu test=%zu\n",
               reference.size(),
               test.size());
        return stats;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = std::fabs(static_cast<double>(reference[i]) - static_cast<double>(test[i]));
        stats.total_abs_error += diff;
        stats.max_abs_error = std::max(stats.max_abs_error, diff);
    }
    return stats;
}

float RunCublasBaseline(const std::vector<MAT_VAL_TYPE>& denseA_row_major,
                        const std::vector<MAT_VAL_TYPE>& denseB_row_major,
                        vint                             numNodes,
                        vint                             feature_dim,
                        std::vector<MAT_VAL_TYPE>&       cublas_output_row_major)
{
    const size_t rowsA = static_cast<size_t>(numNodes);
    const size_t colsA = static_cast<size_t>(numNodes);
    const size_t colsC = static_cast<size_t>(feature_dim);

    std::vector<MAT_VAL_TYPE> denseA_col_major(rowsA * colsA);
    std::vector<MAT_VAL_TYPE> denseB_col_major(colsA * colsC);
    std::vector<MAT_VAL_TYPE> denseC_col_major(rowsA * colsC, static_cast<MAT_VAL_TYPE>(0));

    RowMajorToColumnMajor(denseA_row_major.data(), denseA_col_major.data(), numNodes, numNodes);
    RowMajorToColumnMajor(denseB_row_major.data(), denseB_col_major.data(), numNodes, feature_dim);

    MAT_VAL_TYPE* dA = nullptr;
    MAT_VAL_TYPE* dB = nullptr;
    MAT_VAL_TYPE* dC = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dA, denseA_col_major.size() * sizeof(MAT_VAL_TYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&dB, denseB_col_major.size() * sizeof(MAT_VAL_TYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&dC, denseC_col_major.size() * sizeof(MAT_VAL_TYPE)));

    CHECK_CUDA_ERROR(cudaMemcpy(dA,
                                denseA_col_major.data(),
                                denseA_col_major.size() * sizeof(MAT_VAL_TYPE),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dB,
                                denseB_col_major.data(),
                                denseB_col_major.size() * sizeof(MAT_VAL_TYPE),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dC,
                                denseC_col_major.data(),
                                denseC_col_major.size() * sizeof(MAT_VAL_TYPE),
                                cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CheckCublasStatus(cublasCreate(&handle), "cublasCreate");
    CheckCublasStatus(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode");

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CheckCublasStatus(
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    numNodes,
                    feature_dim,
                    numNodes,
                    &alpha,
                    dA,
                    numNodes,
                    dB,
                    numNodes,
                    &beta,
                    dC,
                    numNodes),
        "cublasSgemm");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    CHECK_CUDA_ERROR(cudaMemcpy(denseC_col_major.data(),
                                dC,
                                denseC_col_major.size() * sizeof(MAT_VAL_TYPE),
                                cudaMemcpyDeviceToHost));

    cublas_output_row_major.resize(rowsA * colsC);
    ColumnMajorToRowMajor(denseC_col_major.data(),
                          cublas_output_row_major.data(),
                          numNodes,
                          feature_dim);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return elapsed_ms;
}

}  // namespace

__host__
void tf32_spmm(
    METCFBit<MAT_VAL_TYPE>& metcf_bit,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute,
    const MAT_VAL_TYPE* denseMatBHost,
    std::vector<MAT_VAL_TYPE>& host_output
) {
    vint rowWindowOffsetSize    =       metcf_bit.rowWindowOffset.size();
    vint tcOffsetSize           =       metcf_bit.tcOffset.size();
    vint sparseA2BSize          =       metcf_bit.sparseA2B.size();
    vint tcLocalBitSize         =       metcf_bit.tcLocalBit.size();
    vint dataSize               =       metcf_bit.data.size();

    vint numBlocks              =       rowWindowOffsetSize - 1;
    vint denseC_size            =       numBlocks * ROW_WINDOW * feature_dim;
    
    /*---------------- CPU Malloc ------------------*/
    vint* ptr_rowWindowOffset           =   (vint*)malloc(sizeof(vint) * rowWindowOffsetSize);
    vint* ptr_sparseA2B                 =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    vint* ptr_tcOffset                  =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    TCLOCAL_TYPE* ptr_tcLocalBit        =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data              =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(metcf_bit.rowWindowOffset.begin(), metcf_bit.rowWindowOffset.end(), ptr_rowWindowOffset);
    std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);
    std::copy(metcf_bit.tcOffset.begin(), metcf_bit.tcOffset.end(), ptr_tcOffset);
    std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);
    size_t denseB_elems = static_cast<size_t>(numNodes) * feature_dim;
    memcpy(DenseMatB, denseMatBHost, denseB_elems * sizeof(MAT_VAL_TYPE));
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_rowWindowOffset, *d_sparseA2B, *d_tcOffset;
    MAT_VAL_TYPE* d_data;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE*   d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowWindowOffset, ptr_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.compute(
        d_tcLocalBit, d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC, 
        d_rowWindowOffset, d_tcOffset,
        numNodes, numBlocks, feature_dim, timer
    );

    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
    const size_t valid_output = static_cast<size_t>(numNodes) * feature_dim;
    const size_t copy_elems   = std::min(valid_output, static_cast<size_t>(denseC_size));
    host_output.assign(DenseMatC, DenseMatC + copy_elems);

    free(ptr_rowWindowOffset);
    free(ptr_sparseA2B);
    free(ptr_tcOffset);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_rowWindowOffset);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcOffset);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

__host__
void adp_tf32_spmm(
    AdpBME<MAT_VAL_TYPE>& adpbme,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute,
    const MAT_VAL_TYPE* denseMatBHost,
    std::vector<MAT_VAL_TYPE>& host_output
) {
    vint groupOffsetSize    =   adpbme.groupOffset.size();
    vint tcOffsetSize       =   adpbme.tcOffset.size();
    vint rowIndicesSize     =   adpbme.rowIndices.size();
    vint sparseA2BSize      =   adpbme.sparseA2B.size();
    vint tcLocalBitSize     =   adpbme.tcLocalBit.size();
    vint dataSize           =   adpbme.data.size();

    vint numBlocks          =   groupOffsetSize - 1;
    vint denseC_size        =   std::max(numBlocks * ROW_WINDOW * feature_dim, (adpbme.rowIndices.back() + 8) * feature_dim);

    /*---------------- CPU Malloc ------------------*/
    vint* ptr_groupOffset           =   (vint*)malloc(sizeof(vint) * groupOffsetSize);
    vint* ptr_tcOffset              =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    vint* ptr_rowIndices            =   (vint*)malloc(sizeof(vint) * rowIndicesSize);
    vint* ptr_sparseA2B             =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    TCLOCAL_TYPE* ptr_tcLocalBit    =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data          =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_groupOffset);
    std::copy(adpbme.tcOffset.begin(), adpbme.tcOffset.end(), ptr_tcOffset);
    std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_rowIndices);
    std::copy(adpbme.sparseA2B.begin(), adpbme.sparseA2B.end(), ptr_sparseA2B);
    std::copy(adpbme.tcLocalBit.begin(), adpbme.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(adpbme.data.begin(), adpbme.data.end(), ptr_data);

    size_t denseB_elems = static_cast<size_t>(numNodes) * feature_dim;
    memcpy(DenseMatB, denseMatBHost, denseB_elems * sizeof(MAT_VAL_TYPE));
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_groupOffset, *d_tcOffset, *d_rowIndices, *d_sparseA2B;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE* d_data, *d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_groupOffset, sizeof(vint) * groupOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowIndices, sizeof(vint) * rowIndicesSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_groupOffset, ptr_groupOffset, sizeof(vint) * groupOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowIndices, ptr_rowIndices, sizeof(vint) * rowIndicesSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.adpBalanceCompute(
        d_groupOffset, d_tcOffset, d_rowIndices, d_tcLocalBit, 
        d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC,
        numBlocks, numNodes, 
        feature_dim, timer
    );
    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
    const size_t valid_output = static_cast<size_t>(numNodes) * feature_dim;
    const size_t copy_elems   = std::min(valid_output, static_cast<size_t>(denseC_size));
    host_output.assign(DenseMatC, DenseMatC + copy_elems);

    free(ptr_groupOffset);
    free(ptr_tcOffset);
    free(ptr_rowIndices);
    free(ptr_sparseA2B);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_groupOffset);
    cudaFree(d_tcOffset);
    cudaFree(d_rowIndices);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

__host__
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./mma_tf32 <matrix.mtx> <feature_dim>\n");
        return ERROR_ARGS;
    }
    
    const char* filename = argv[1];
    const int feature_dim = atoi(argv[2]);
        
    const std::string mtx_name = match_filename(std::string(filename));
    
    // Load original matrix
    COO<MAT_VAL_TYPE>* coo = (COO<MAT_VAL_TYPE>*)malloc(sizeof(COO<MAT_VAL_TYPE>));
    int read_status = read_from_mtx<MAT_VAL_TYPE>(filename, coo);
    if(read_status != SUCCESS) {
        return ERROR_READ_MTX;
    }
    
    CSR<MAT_VAL_TYPE> csr = CSR<MAT_VAL_TYPE>(coo);
    vint numNodes = coo->rows;
    vint numEdges = coo->nnz;

    if (csr.cols != numNodes) {
        printf("Non-square matrices are not supported (rows=%u, cols=%u).\n", numNodes, csr.cols);
        free(coo);
        return ERROR_NOT_MATCH;
    }

    std::vector<MAT_VAL_TYPE> denseMatB(static_cast<size_t>(numNodes) * feature_dim);
    init_vecB(static_cast<vint>(denseMatB.size()), denseMatB.data(), 1.0);
    //init_vec1(static_cast<vint>(denseMatB.size()), denseMatB.data(), 10.0);
    std::vector<MAT_VAL_TYPE> spmm_output;
    spmm_output.reserve(static_cast<size_t>(numNodes) * feature_dim);

    std::vector<MAT_VAL_TYPE> denseA_row_major(static_cast<size_t>(numNodes) * numNodes, 0.0f);
    for (vint row = 0; row < numNodes; ++row) {
        for (vint idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx) {
            vint col = csr.col_idx[idx];
            denseA_row_major[row * numNodes + col] = csr.data[idx];
        }
    }

    METCFBit<MAT_VAL_TYPE> metcf_bit;
    metcf_bit.convertFromCSR(csr);
    
    bool load_balance = balanceStrategy(metcf_bit, 
                                        metcf_bit.tcOffset.size() - 1, 
                                        metcf_bit.rowWindowOffset.size() - 1);
    float elapsed_time = 0.0, spmm_throughput = 0.0;
    
    GpuTimer timer;
    TF32Compute tf32_compute;

    //load_balance = false;

    if(load_balance) {
        // Adaptive balance Tensor core operations
        AdpBME<MAT_VAL_TYPE> adpbme;
        adpbme.CSR2AdpBME(csr, 3, 1, 2);
        TF32Compute tf32_compute(10, 10, 128);
        adp_tf32_spmm(adpbme,
                      numNodes,
                      numEdges,
                      elapsed_time,
                      feature_dim,
                      timer,
                      tf32_compute,
                      denseMatB.data(),
                      spmm_output);
            
    } else {
        // Tensor core operations
        TF32Compute tf32_compute;
        tf32_spmm(metcf_bit,
                  numNodes,
                  numEdges,
                  elapsed_time,
                  feature_dim,
                  timer,
                  tf32_compute,
                  denseMatB.data(),
                  spmm_output);
    }
    spmm_throughput = (float(numEdges) * float(feature_dim) * 2.0 * 1000.) 
                    / (elapsed_time * 1000. * 1000. * 1000.);

    std::vector<MAT_VAL_TYPE> cublas_output;
    float cublas_time = RunCublasBaseline(denseA_row_major, denseMatB, numNodes, feature_dim, cublas_output);
    ErrorStats error_stats = ComputeErrorStats(cublas_output, spmm_output);
    printf("cuBLAS TF32 time: %.4f ms | SpMM time: %.4f ms\n", cublas_time, elapsed_time);
    printf("cuBLAS vs SpMM absolute error: total=%.6e max=%.6e\n",
           error_stats.total_abs_error,
           error_stats.max_abs_error);
    std::ofstream outFile("result.csv", std::ios::app);
    if (!outFile) {
        std::cerr << "Error Opening result.csv" << std::endl;
    }
    outFile << mtx_name << ","<< feature_dim << "," << elapsed_time << "," << spmm_throughput << "\n";
    outFile.close();  

    free(coo);
    return 0;
}
