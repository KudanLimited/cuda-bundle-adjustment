

template<typename T>
void SparseCholesky<T>::init(cusolverSpHandle_t handle)
{
    handle_ = handle;

    // create info
    cusolverSpCreateCsrcholInfo(&info_);
}

template<typename T>
void SparseCholesky<T>::allocateBuffer(const SparseSquareMatrixCSR<T>& A)
{
    size_t internalData, workSpace;

    if constexpr (is_value_type_32f<T>())
        cusolverSpScsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
            A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

    if constexpr (is_value_type_64f<T>())
        cusolverSpDcsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
            A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

    buffer_.resize(workSpace);
}

template<typename T>
bool SparseCholesky<T>::hasZeroPivot(int* position) const
{
    const T tol = static_cast<T>(1e-14);
    int singularity = -1;

    if constexpr (is_value_type_32f<T>())
        cusolverSpScsrcholZeroPivot(handle_, info_, tol, &singularity);

    if constexpr (is_value_type_64f<T>())
        cusolverSpDcsrcholZeroPivot(handle_, info_, tol, &singularity);

    if (position)
        *position = singularity;
    return singularity >= 0;
}

template<typename T>
bool SparseCholesky<T>::analyze(const SparseSquareMatrixCSR<T>& A)
{
    cusolverSpXcsrcholAnalysis(handle_, A.size(), A.nnz(), A.desc(), A.rowPtr(), A.colInd(), info_);
    allocateBuffer(A);
    return true;
}

template<typename T>
bool SparseCholesky<T>::factorize(SparseSquareMatrixCSR<T>& A)
{
    if constexpr (is_value_type_32f<T>())
        cusolverSpScsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
            A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

    if constexpr (is_value_type_64f<T>())
        cusolverSpDcsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
            A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

    return !hasZeroPivot();
}

template<typename T>
void SparseCholesky<T>::solve(int size, const T* b, T* x)
{
    if constexpr (is_value_type_32f<T>())
        cusolverSpScsrcholSolve(handle_, size, b, x, info_, buffer_.data());

    if constexpr (is_value_type_64f<T>())
        cusolverSpDcsrcholSolve(handle_, size, b, x, info_, buffer_.data());
}

template<typename T>
void SparseCholesky<T>::destroy()
{
    cusolverSpDestroyCsrcholInfo(info_);
}

template<typename T>
SparseCholesky<T>::~SparseCholesky() { destroy(); }

template<typename T>
CuSparseCholeskySolver<T>::CuSparseCholeskySolver(int size)
{
    init();

    if (size > 0)
        resize(size);
}

template<typename T>
void CuSparseCholeskySolver<T>::init()
{
    cholesky.init(cusolver);
    doOrdering = false;
    information = Info::SUCCESS;
}

template<typename T>
void CuSparseCholeskySolver<T>::resize(int size)
{
    Acsr.resize(size);
    d_y.resize(size);
    d_z.resize(size);
}

template<typename T>
void CuSparseCholeskySolver<T>::setPermutaion(int size, const int* P)
{
    h_PT.resize(size);
    for (int i = 0; i < size; i++)
        h_PT[P[i]] = i;

    d_P.assign(size, P);
    d_PT.assign(size, h_PT.data());
    doOrdering = true;
}

template<typename T>
void CuSparseCholeskySolver<T>::analyze(int nnz, const int* csrRowPtr, const int* csrColInd)
{
    const int size = Acsr.size();
    Acsr.resizeNonZeros(nnz);

    if (doOrdering)
    {
        d_tmpRowPtr.assign(size + 1, csrRowPtr);
        d_tmpColInd.assign(nnz, csrColInd);
        d_nnzPerRow.resize(size + 1);
        d_map.resize(nnz);

        gpu::twistCSR(size, nnz, d_tmpRowPtr, d_tmpColInd, d_PT,
            Acsr.rowPtr(), Acsr.colInd(), d_map, d_nnzPerRow);
    }
    else
    {
        Acsr.upload(nullptr, csrRowPtr, csrColInd);
    }

    cholesky.analyze(Acsr);
}

template<typename T>
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
        information = Info::NUMERICAL_ISSUE;
}

template<typename T>
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

template<typename T>
void CuSparseCholeskySolver<T>::permute(int size, const T* src, T* dst, const int* P)
{
    gpu::permute(size, src, dst, P);
}

template<typename T>
void CuSparseCholeskySolver<T>::reordering(int size, int nnz, const int* csrRowPtr, const int* csrColInd, int* P) const
{
    //cusolverSpXcsrsymrcmHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
    //cusolverSpXcsrsymamdHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
    //cusolverSpXcsrsymmdqHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
    cusolverSpXcsrmetisndHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, nullptr, P);
}

template<typename T>
typename CuSparseCholeskySolver<T>::Info CuSparseCholeskySolver<T>::info() const
{
    return information;
}

template<typename T>
void CuSparseCholeskySolver<T>::downloadCSR(int* csrRowPtr, int* csrColInd)
{
    Acsr.download(nullptr, csrRowPtr, csrColInd);
}
