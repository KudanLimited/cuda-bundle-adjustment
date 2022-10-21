


template <typename T>
void SparseCholesky<T>::init(cusolverSpHandle_t handle)
{
    handle_ = handle;

    // create info
    CHECK_CUSOLVER(cusolverSpCreateCsrcholInfo(&info_));
}

template <typename T>
void SparseCholesky<T>::allocateBuffer(const SparseSquareMatrixCSR<T>& A)
{
}

template <>
inline void SparseCholesky<float>::allocateBuffer(const SparseSquareMatrixCSR<float>& A)
{
    size_t internalData, workSpace;

    CHECK_CUSOLVER(cusolverSpScsrcholBufferInfo(
        handle_,
        A.size(),
        A.nnz(),
        A.desc(),
        A.val(),
        A.rowPtr(),
        A.colInd(),
        info_,
        &internalData,
        &workSpace));

    buffer_.resize(workSpace);
}

template <>
inline void SparseCholesky<double>::allocateBuffer(const SparseSquareMatrixCSR<double>& A)
{
    size_t internalData, workSpace;

    CHECK_CUSOLVER(cusolverSpDcsrcholBufferInfo(
        handle_,
        A.size(),
        A.nnz(),
        A.desc(),
        A.val(),
        A.rowPtr(),
        A.colInd(),
        info_,
        &internalData,
        &workSpace));

    buffer_.resize(workSpace);
}

template <typename T>
bool SparseCholesky<T>::hasZeroPivot(int* position) const
{
    return false;
}

template <>
inline bool SparseCholesky<float>::hasZeroPivot(int* position) const
{
    const float tol = static_cast<float>(1e-14);
    int singularity = -1;

    CHECK_CUSOLVER(cusolverSpScsrcholZeroPivot(handle_, info_, tol, &singularity));

    if (position)
    {
        *position = singularity;
    }
    return singularity >= 0;
}

template <>
inline bool SparseCholesky<double>::hasZeroPivot(int* position) const
{
    const double tol = static_cast<double>(1e-14);
    int singularity = -1;

    CHECK_CUSOLVER(cusolverSpDcsrcholZeroPivot(handle_, info_, tol, &singularity));

    if (position)
    {
        *position = singularity;
    }
    return singularity >= 0;
}

template <typename T>
bool SparseCholesky<T>::analyze(const SparseSquareMatrixCSR<T>& A)
{
    CHECK_CUSOLVER(cusolverSpXcsrcholAnalysis(
        handle_, A.size(), A.nnz(), A.desc(), A.rowPtr(), A.colInd(), info_));
    allocateBuffer(A);
    return true;
}

template <typename T>
bool SparseCholesky<T>::factorize(SparseSquareMatrixCSR<T>& A)
{
    return false;
}

template <>
inline bool SparseCholesky<float>::factorize(SparseSquareMatrixCSR<float>& A)
{
    CHECK_CUSOLVER(cusolverSpScsrcholFactor(
        handle_,
        A.size(),
        A.nnz(),
        A.desc(),
        A.val(),
        A.rowPtr(),
        A.colInd(),
        info_,
        buffer_.data()));
    return !hasZeroPivot();
}

template <>
inline bool SparseCholesky<double>::factorize(SparseSquareMatrixCSR<double>& A)
{
    CHECK_CUSOLVER(cusolverSpDcsrcholFactor(
        handle_,
        A.size(),
        A.nnz(),
        A.desc(),
        A.val(),
        A.rowPtr(),
        A.colInd(),
        info_,
        buffer_.data()));
    return !hasZeroPivot();
}

template <typename T>
void SparseCholesky<T>::solve(int size, const T* b, T* x)
{
}

template <>
inline void SparseCholesky<float>::solve(int size, const float* b, float* x)
{
    CHECK_CUSOLVER(cusolverSpScsrcholSolve(handle_, size, b, x, info_, buffer_.data()));
}

template <>
inline void SparseCholesky<double>::solve(int size, const double* b, double* x)
{
    CHECK_CUSOLVER(cusolverSpDcsrcholSolve(handle_, size, b, x, info_, buffer_.data()));
}

template <typename T>
void SparseCholesky<T>::destroy()
{
    CHECK_CUSOLVER(cusolverSpDestroyCsrcholInfo(info_));
}

template <typename T>
SparseCholesky<T>::~SparseCholesky()
{
    destroy();
}

template <typename T>
CuSparseCholeskySolver<T>::CuSparseCholeskySolver(int size)
{
    init();

    if (size > 0)
    {
        resize(size);
    }
}

template <typename T>
void CuSparseCholeskySolver<T>::init()
{
    cholesky.init(cusolver);
    doOrdering = false;
    information = Info::SUCCESS;
}

template <typename T>
void CuSparseCholeskySolver<T>::resize(int size)
{
    Acsr.resize(size);
    d_y.resize(size);
    d_z.resize(size);
}

template <typename T>
void CuSparseCholeskySolver<T>::setPermutaion(int size, const int* P, cudaStream_t stream)
{
    h_PT.resize(size);
    for (int i = 0; i < size; i++)
    {
        h_PT[P[i]] = i;
    }

    d_P.assignAsync(size, P, stream);
    d_PT.assignAsync(size, h_PT.data(), stream);
    doOrdering = true;
}

template <typename T>
void CuSparseCholeskySolver<T>::analyze(
    int nnz, const int* csrRowPtr, const int* csrColInd, const CudaDeviceInfo& deviceInfo)
{
    const int size = Acsr.size();
    Acsr.resizeNonZeros(nnz);

    if (doOrdering)
    {
        d_tmpRowPtr.assign(size + 1, csrRowPtr);
        d_tmpColInd.assign(nnz, csrColInd);
        d_nnzPerRow.resize(size + 1);
        d_map.resize(nnz);

        gpu::twistCSR(
            size,
            nnz,
            d_tmpRowPtr,
            d_tmpColInd,
            d_PT,
            Acsr.rowPtr(),
            Acsr.colInd(),
            d_map,
            d_nnzPerRow,
            deviceInfo);
    }
    else
    {
        Acsr.uploadAsync(nullptr, csrRowPtr, csrColInd, deviceInfo.stream);
    }

    cholesky.analyze(Acsr);
}

template <typename T>
void CuSparseCholeskySolver<T>::factorize(const T* d_A)
{
    if (doOrdering)
    {
        permute(Acsr.nnz(), d_A, Acsr.val(), d_map);
    }
    else
    {
        cudaMemcpy(Acsr.val(), d_A, sizeof(Scalar) * Acsr.nnz(), cudaMemcpyDeviceToDevice);
    }

    // M = L * LT
    if (!cholesky.factorize(Acsr))
    {
        information = Info::NUMERICAL_ISSUE;
    }
}

template <typename T>
void CuSparseCholeskySolver<T>::solve(const T* d_b, T* d_x)
{
    if (doOrdering)
    {
        // y = P * b
        permute(Acsr.size(), d_b, d_y, d_P);

        // solve A * z = y
        cholesky.solve(Acsr.size(), d_y, d_z);

        // x = PT * z
        permute(Acsr.size(), d_z, d_x, d_PT);
    }
    else
    {
        // solve A * x = b
        cholesky.solve(Acsr.size(), d_b, d_x);
    }
}

template <typename T>
void CuSparseCholeskySolver<T>::permute(int size, const T* src, T* dst, const int* P)
{
    gpu::permute(size, src, dst, P);
}

template <typename T>
void CuSparseCholeskySolver<T>::reordering(
    int size, int nnz, const int* csrRowPtr, const int* csrColInd, int* P) const
{
    CHECK_CUSOLVER(cusolverSpXcsrmetisndHost(
        cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, nullptr, P));
}

template <typename T>
typename CuSparseCholeskySolver<T>::Info CuSparseCholeskySolver<T>::info() const
{
    return information;
}

template <typename T>
void CuSparseCholeskySolver<T>::downloadCSR(int* csrRowPtr, int* csrColInd, cudaStream_t stream)
{
    Acsr.downloadAsync(nullptr, csrRowPtr, csrColInd, stream);
}

/**********************************************************************************************************
 * Dense cholesky functions
 *********************************************************************************************************/
template <typename T>
void DenseCholesky<T>::init(cusolverDnHandle_t dnHandle, cusparseHandle_t spHandle)
{
    dnHandle_ = dnHandle;
    spHandle_ = spHandle;
}

template <typename T>
void DenseCholesky<T>::allocateBuffer(SparseSquareMatrixCSR<T>& A, DenseSquareMatrix<T>& B)
{
}

template <>
inline void
DenseCholesky<float>::allocateBuffer(SparseSquareMatrixCSR<float>& A, DenseSquareMatrix<float>& B)
{
    CHECK_CUSPARSE(cusparseCreateCsr(
        &spMatDescr,
        A.rows(),
        A.cols(),
        A.nnz(),
        A.rowPtr(),
        A.colInd(),
        A.val(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &dnMatDescr, B.rows(), B.cols(), B.ld(), B.val(), CUDA_R_32F, CUSPARSE_ORDER_COL));

#ifndef USE_TOOLKIT_10_2
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
        spHandle_, spMatDescr, dnMatDescr, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));
    denseBuffer_.resize(bufferSize);
#endif

    int sytBufferSize = 0;
    CHECK_CUSOLVER(
        cusolverDnSsytrf_bufferSize(dnHandle_, B.rows(), B.val(), B.ld(), &sytBufferSize));
    buffer_.resize(sytBufferSize);
    info_.resize(1);
    ipiv_.resize(B.rows());
}

template <>
inline void DenseCholesky<double>::allocateBuffer(
    SparseSquareMatrixCSR<double>& A, DenseSquareMatrix<double>& B)
{
    CHECK_CUSPARSE(cusparseCreateCsr(
        &spMatDescr,
        A.rows(),
        A.cols(),
        A.nnz(),
        A.rowPtr(),
        A.colInd(),
        A.val(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &dnMatDescr, B.rows(), B.cols(), B.ld(), B.val(), CUDA_R_64F, CUSPARSE_ORDER_COL));

#ifndef USE_TOOLKIT_10_2
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
        spHandle_, spMatDescr, dnMatDescr, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));
    denseBuffer_.resize(bufferSize);
#endif

    int sytBufferSize = 0;
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(
        dnHandle_, B.rows(), B.cols(), B.val(), B.ld(), &sytBufferSize));
    buffer_.resize(sytBufferSize);
    info_.resize(1);
    ipiv_.resize(B.rows());
}

#ifdef USE_TOOLKIT_10_2
template <typename T>
inline void
DenseCholesky<T>::sparseToDense(const SparseSquareMatrixCSR<T>& A, DenseSquareMatrix<T>& B)
{
}

template <>
inline void DenseCholesky<float>::sparseToDense(
    const SparseSquareMatrixCSR<float>& A, DenseSquareMatrix<float>& B)
{
    // CHECK_CUSPARSE(cusparseCcsr2dense(spHandle_, A.rows(), A.cols(), A.desc(), A.val(),
    // A.rowPtr(), A.colInd(), B.val(), A.rows()));
}

template <>
inline void DenseCholesky<double>::sparseToDense(
    const SparseSquareMatrixCSR<double>& A, DenseSquareMatrix<double>& B)
{
    CHECK_CUSPARSE(cusparseDcsr2dense(
        spHandle_,
        A.rows(),
        A.cols(),
        A.desc(),
        A.val(),
        A.rowPtr(),
        A.colInd(),
        B.val(),
        A.rows()));
}
#else
template <typename T>
inline void
DenseCholesky<T>::sparseToDense(const SparseSquareMatrixCSR<T>& A, DenseSquareMatrix<T>& B)
{
    CHECK_CUSPARSE(cusparseSparseToDense(
        spHandle_,
        spMatDescr,
        dnMatDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        denseBuffer_.data()));
}
#endif

template <typename T>
bool DenseCholesky<T>::factorize(DenseSquareMatrix<T>& A)
{
    return false;
}

template <>
inline bool DenseCholesky<float>::factorize(DenseSquareMatrix<float>& A)
{
    CHECK_CUSOLVER(cusolverDnSsytrf(
        dnHandle_,
        CUBLAS_FILL_MODE_LOWER,
        A.rows(),
        A.val(),
        A.ld(),
        ipiv_.data(),
        buffer_.data(),
        buffer_.size(),
        info_.data()));
    CUDA_CHECK(cudaMemcpy(&h_info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        return false;
    }
    return true;
}

template <>
inline bool DenseCholesky<double>::factorize(DenseSquareMatrix<double>& A)
{
    CHECK_CUSOLVER(cusolverDnDgetrf(
        dnHandle_,
        A.rows(),
        A.cols(),
        A.val(),
        A.ld(),
        buffer_.data(),
        ipiv_.data(),
        info_.data()));
    CUDA_CHECK(cudaMemcpy(&h_info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        return false;
    }
    return true;
}

template <typename T>
void DenseCholesky<T>::solve(const DenseSquareMatrix<T>& A, const T* b, T* x)
{
}

template <>
inline void DenseCholesky<float>::solve(const DenseSquareMatrix<float>& A, const float* b, float* x)
{
    int size = A.rows();
    int ld = A.ld();

    CUDA_CHECK(cudaMemcpy(x, b, sizeof(float) * size, cudaMemcpyDeviceToDevice));
    CHECK_CUSOLVER(cusolverDnSgetrs(
        dnHandle_, CUBLAS_OP_N, size, 1, A.val(), ld, ipiv_.data(), x, size, info_.data()));
}

template <>
inline void
DenseCholesky<double>::solve(const DenseSquareMatrix<double>& A, const double* b, double* x)
{
    int size = A.rows();
    int ld = A.ld();

    CUDA_CHECK(cudaMemcpy(x, b, sizeof(double) * size, cudaMemcpyDeviceToDevice));
    CHECK_CUSOLVER(cusolverDnDgetrs(
        dnHandle_, CUBLAS_OP_N, size, 1, A.val(), ld, ipiv_.data(), x, size, info_.data()));
}

template <typename T>
DenseCholesky<T>::~DenseCholesky()
{
}

template <typename T>
CuDenseCholeskySolver<T>::CuDenseCholeskySolver(int size)
{
    init();

    if (size > 0)
    {
        resize(size);
    }
}

template <typename T>
void CuDenseCholeskySolver<T>::init()
{
    cholesky.init(cusolver, cusparse);
    information = Info::SUCCESS;
}

template <typename T>
void CuDenseCholeskySolver<T>::resize(int size)
{
    Acsr.resize(size);
    Adense.resize(size);
}

template <typename T>
void CuDenseCholeskySolver<T>::allocate(int nnz, const int* csrRowPtr, const int* csrColInd, cudaStream_t stream)
{
    Acsr.resizeNonZeros(nnz);
    Acsr.uploadAsync(nullptr, csrRowPtr, csrColInd, stream);
    cholesky.allocateBuffer(Acsr, Adense);
}

template <typename T>
void CuDenseCholeskySolver<T>::factorize(const T* d_A)
{
    CUDA_CHECK(cudaMemcpy(Acsr.val(), d_A, sizeof(Scalar) * Acsr.nnz(), cudaMemcpyDeviceToDevice));
    cholesky.sparseToDense(Acsr, Adense);

    // A = L * LH
    if (!cholesky.factorize(Adense))
    {
        information = Info::NUMERICAL_ISSUE;
    }
}

template <typename T>
void CuDenseCholeskySolver<T>::solve(const T* d_b, T* d_x)
{
    // solve A * x = b
    cholesky.solve(Adense, d_b, d_x);
}

template <typename T>
typename CuDenseCholeskySolver<T>::Info CuDenseCholeskySolver<T>::info() const
{
    return information;
}