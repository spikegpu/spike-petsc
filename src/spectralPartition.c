static char help[] = "Partition the graph of a matrix using spectral partitioning.\n\
Input parameters include:\n\
  -mat <matrix file> : the matrix in PETSc binary format\n\n";
/*
Matrices are at http://www.cise.ufl.edu/research/sparse/matrices/list_by_dimension.html

Matrix Market files can be converted using:

  src/mat/examples/tests/ex72 -fin matrix.mtx -fout matrix.bin

This code could be integrated.
*/
#include <petscsnes.h>
#include <petscblaslapack.h>

typedef struct {
  char      matFilename[PETSC_MAX_PATH_LEN];
  char      matOrdtype[256];
  PetscBool showSpectrum, showFiedler;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscFunctionList ordlist;
  char              tname[256];
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  options->matFilename[0] = '\0';
  ierr = PetscStrncpy(options->matOrdtype, MATORDERINGRCM, 256);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Reordering Test Options", "Mat");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat", "Path for matrix input file", "main", options->matFilename, options->matFilename, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must provide an input matrix using -mat <file>");
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsList("-mat_ordering_type", "Reordering for matrix", "main", ordlist, options->matOrdtype, tname, 256, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscStrncpy(options->matOrdtype, tname, 256);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-show_spectrum", "Print the graph Laplacian spectrum", "main", options->showSpectrum, &options->showSpectrum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_fielder", "Print the Fiedler vector", "main", options->showFiedler, &options->showFiedler, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "MatLaplacian"
/*@
  MatLaplacian - Form the matrix Laplacian, with all values in the matrix less than the tolerance set to zero

  Input Parameters:
+ A   - The matrix
- tol - The zero tolerance

  Output Parameters:
. L - The graph Laplacian matrix

  Level: intermediate

.seealso: MatChop()
 @*/
PetscErrorCode MatLaplacian(Mat A, PetscReal tol, Mat *L)
{
  PetscScalar   *newVals;
  PetscInt      *newCols;
  PetscInt       rStart, rEnd, r, colMax = 0;
  PetscInt      *dnnz, *onnz;
  PetscInt       m, n, M, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject) A), L);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
  ierr = MatSetSizes(*L, m, n, M, N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,PetscInt,&dnnz,m,PetscInt,&onnz);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    dnnz[r-rStart] = onnz[r-rStart] = 0;
    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0, newcols = 0; c < ncols; ++c) {
      if (cols[c] == r) {
        ++newcols;
        hasdiag = PETSC_TRUE;
        ++dnnz[r-rStart];
      } else if (PetscAbsScalar(vals[c]) >= tol) {
        if ((cols[c] >= rStart) && (cols[c] < rEnd)) ++dnnz[r-rStart];
        else                                         ++onnz[r-rStart];
        ++newcols;
      }
    }
    if (!hasdiag) {++newcols; ++dnnz[r-rStart];}
    colMax = PetscMax(colMax, newcols);CHKERRQ(ierr);
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(*L);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(*L, 1, dnnz, onnz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*L);CHKERRQ(ierr);
  ierr = PetscMalloc2(colMax,PetscInt,&newCols,colMax,PetscScalar,&newVals);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0, newcols = 0; c < ncols; ++c) {
      if (cols[c] == r) {
        newCols[newcols] = cols[c];
        newVals[newcols] = dnnz[r-rStart]+onnz[r-rStart]-1;
        ++newcols;
        hasdiag = PETSC_TRUE;
      } else if (PetscAbsScalar(vals[c]) >= tol) {
        newCols[newcols] = cols[c];
        newVals[newcols] = -1.0;
        ++newcols;
      }
      if (newcols > colMax) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Overran work space");
    }
    if (!hasdiag) {
      newCols[newcols] = r;
      newVals[newcols] = dnnz[r-rStart]+onnz[r-rStart]-1;
      ++newcols;
    }
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    ierr = MatSetValues(*L, 1, &r, newcols, newCols, newVals, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(newCols,newVals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSubMatrixBanded"
/*@
  MatCreateSubMatrixBanded - Extract the banded subset B of A such that ||Vec(B)||_1 >= frac ||Vec(A)||_1

  Input Parameters:
+ A   - The matrix
. kmax - The maximum half-bandwidth, so 2k+1 diagonals may be extracted
- frac - The norm fraction for the extracted band

  Output Parameters:
. B - The banded submatrix

  Level: intermediate

.seealso: MatChop()
 @*/
PetscErrorCode MatCreateSubMatrixBanded(Mat A, PetscInt kmax, PetscReal frac, Mat *B)
{
  Vec            weight;
  PetscScalar   *w, *newVals;
  PetscReal      normA = 0.0, normB = 0.0;
  PetscInt       rStart, rEnd, r;
  PetscInt      *dnnz, *onnz, *newCols;
  PetscInt       m, n, M, N, k, maxcols = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create weight vector */
  ierr = MatGetVecs(A, NULL, &weight);CHKERRQ(ierr);
  ierr = VecSet(weight, 0.0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = VecGetArray(weight, &w);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, c;

    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0; c < ncols; ++c) {
      w[abs(r - cols[c])] += PetscAbsScalar(vals[c]);
      normA += PetscAbsScalar(vals[c]);
    }
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(weight, &w);CHKERRQ(ierr);
  /* Determine bandwidth */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "||Vec(A)||_1: %g\n", normA);CHKERRQ(ierr);
  ierr = VecGetArray(weight, &w);CHKERRQ(ierr);
  for (k = 0; k < kmax; ++k) {
    normB += w[k];
    if (normB >= frac*normA) break;
  }
  ierr = VecRestoreArray(weight, &w);CHKERRQ(ierr);
  ierr = VecDestroy(&weight);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Bandwidth of %d%% band: %d frac: %g\n", (PetscInt) (frac*100), k, normB/normA);CHKERRQ(ierr);
  /* Extract band */
  ierr = MatCreate(PetscObjectComm((PetscObject) A), B);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, m, n, M, N);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,PetscInt,&dnnz,m,PetscInt,&onnz);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, c;

    dnnz[r-rStart] = onnz[r-rStart] = 0;
    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0; c < ncols; ++c) {
      if (abs(cols[c] - r) > k) continue;
      if ((cols[c] >= rStart) && (cols[c] < rEnd)) ++dnnz[r-rStart];
      else                                         ++onnz[r-rStart];
    }
    maxcols = PetscMax(ncols, maxcols);
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(*B);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(*B, 1, dnnz, onnz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxcols,PetscInt,&newCols,maxcols,PetscScalar,&newVals);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;

    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0, newcols = 0; c < ncols; ++c) {
      if (abs(cols[c] - r) > k) continue;
      newCols[newcols] = cols[c];
      newVals[newcols] = vals[c];
      ++newcols;
      if (newcols > maxcols) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Overran work space");
    }
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    ierr = MatSetValues(*B, 1, &r, newcols, newCols, newVals, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(newCols, newVals);CHKERRQ(ierr);
  ierr = PetscFree2(dnnz, onnz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
  Mat            A, L;
  AppCtx         ctx;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args, (char *) 0, help);CHKERRQ(ierr);
  ierr = ProcessOptions(&ctx);CHKERRQ(ierr);
  /* Load matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, ctx.matFilename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatLoad(A, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  /* Make graph Laplacian from matrix */
  ierr = MatLaplacian(A, 1.0e-12, &L);CHKERRQ(ierr);
  /* Check Laplacian */
  PetscReal norm;
  Vec       x, y;

  ierr = MatGetVecs(L, &x, NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
  ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  ierr = MatMult(L, x, y);CHKERRQ(ierr);
  ierr = VecNorm(y, NORM_INFINITY, &norm);CHKERRQ(ierr);
  if (norm > 1.0e-10) SETERRQ(PetscObjectComm((PetscObject) y), PETSC_ERR_PLIB, "Invalid graph Laplacian");
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  /* Compute Fiedler vector, and perhaps more vectors */
  Mat          LD;
  PetscScalar *a, *realpart, *imagpart, *eigvec, *work, sdummy;
  PetscBLASInt bn, bN, lwork, lierr, idummy;
  PetscInt     n, i;

  ierr = MatConvert(L, MATDENSE, MAT_INITIAL_MATRIX, &LD);CHKERRQ(ierr);
  ierr = MatGetLocalSize(LD, &n, NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(LD, &a);CHKERRQ(ierr);

  ierr = PetscBLASIntCast(n, &bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n, &bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*n,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(1,&idummy);CHKERRQ(ierr);
  ierr = PetscMalloc4(n,PetscScalar,&realpart,n,PetscScalar,&imagpart,n*n,PetscScalar,&eigvec,lwork,PetscScalar,&work);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCall("LAPACKgeev", LAPACKgeev_("N","V",&bn,a,&bN,realpart,imagpart,&sdummy,&idummy,eigvec,&bN,work,&lwork,&lierr));
  if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscReal *r, *c;
  PetscInt  *perm;

  ierr = PetscMalloc3(n,PetscInt,&perm,n,PetscReal,&r,n,PetscReal,&c);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) perm[i] = i;
  ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  for (i = 0; i < n; ++i) {
    realpart[i] = r[i];
    imagpart[i] = c[i];
  }
  /* Output spectrum */
  if (ctx.showSpectrum) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Spectrum\n");CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = PetscPrintf(PETSC_COMM_SELF, "%d: Real %g Imag %g\n", i, realpart[i], imagpart[i]);CHKERRQ(ierr);}
  }
  /* Check lowest eigenvalue and eigenvector */
  PetscInt evInd = perm[0];

  if ((realpart[0] > 1.0e-12) || (imagpart[0] > 1.0e-12)) SETERRQ(PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have lowest eigenvalue 0");
  for (i = 0; i < n; ++i) {
    if (fabs(eigvec[evInd*n+i] - eigvec[evInd*n+0]) > 1.0e-10) SETERRQ3(PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have constant lowest eigenvector ev_%d %g != ev_0 %g", i, eigvec[evInd*n+i], eigvec[evInd*n+0]);
  }
  /* Output Fiedler vector */
  evInd = perm[1];
  if (ctx.showFiedler) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Fiedler vector, Re{ev} %g\n", realpart[1]);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = PetscPrintf(PETSC_COMM_SELF, "%d: %g\n", i, eigvec[evInd*n+i]);CHKERRQ(ierr);}
  }
  /* Construct Fiedler partition */
  IS        fIS, fIS2;
  PetscInt *fperm, *fperm2, pos, neg, posSize = 0;

  ierr = PetscMalloc(n * sizeof(PetscInt), &fperm);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    if (eigvec[evInd*n+i] > 0.0) ++posSize;
  }

  ierr = PetscMalloc(n * sizeof(PetscInt), &fperm2);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) fperm[i] = i;
  ierr = PetscSortRealWithPermutation(n, &eigvec[evInd*n], fperm);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) fperm2[n-1-i] = fperm[i];

  for (i = 0, pos = 0, neg = posSize; i < n; ++i) {
    if (eigvec[evInd*n+i] > 0.0) fperm[pos++] = i;
    else                         fperm[neg++] = i;
  }

  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) L), n, fperm, PETSC_OWN_POINTER, &fIS);CHKERRQ(ierr);
  ierr = ISSetPermutation(fIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) L), n, fperm2, PETSC_OWN_POINTER, &fIS2);CHKERRQ(ierr);
  ierr = ISSetPermutation(fIS2);CHKERRQ(ierr);

  ierr = PetscFree3(perm,r,c);CHKERRQ(ierr);
  ierr = PetscFree4(realpart,imagpart,eigvec,work);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(LD, &a);CHKERRQ(ierr);
  ierr = MatDestroy(&LD);CHKERRQ(ierr);
  ierr = MatDestroy(&L);CHKERRQ(ierr);
  /* Permute matrix */
  Mat AR, AR2;

  ierr = MatPermute(A, fIS, fIS, &AR);CHKERRQ(ierr);
  ierr = MatView(A,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatView(AR, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&fIS);CHKERRQ(ierr);

  ierr = MatPermute(A, fIS2, fIS2, &AR2);CHKERRQ(ierr);
  ierr = MatView(AR2, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&fIS2);CHKERRQ(ierr);
  ierr = MatDestroy(&AR);CHKERRQ(ierr);
  AR   = AR2;
  /* Extract blocks and reorder */
  Mat               AP, AN, APR, ANR;
  IS                ispos, isneg, rpermpos, cpermpos, rpermneg, cpermneg;
  PetscInt          bw, bwr;

  ierr = ISCreateStride(PETSC_COMM_SELF, posSize, 0, 1, &ispos);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, n - posSize, posSize, 1, &isneg);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(AR, ispos, ispos, MAT_INITIAL_MATRIX, &AP);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(AR, isneg, isneg, MAT_INITIAL_MATRIX, &AN);CHKERRQ(ierr);
  ierr = ISDestroy(&ispos);CHKERRQ(ierr);
  ierr = ISDestroy(&isneg);CHKERRQ(ierr);
  ierr = MatGetOrdering(AP, ctx.matOrdtype, &rpermpos, &cpermpos);CHKERRQ(ierr);
  ierr = MatGetOrdering(AN, ctx.matOrdtype, &rpermneg, &cpermneg);CHKERRQ(ierr);
  ierr = MatPermute(AP, rpermpos, cpermpos, &APR);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(AP, 0.0, &bw);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(APR, 0.0, &bwr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Reduced positive bandwidth from %d to %d\n", bw, bwr);CHKERRQ(ierr);
  ierr = MatPermute(AN, rpermneg, cpermneg, &ANR);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(AN, 0.0, &bw);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(ANR, 0.0, &bwr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Reduced negative bandwidth from %d to %d\n", bw, bwr);CHKERRQ(ierr);
  ierr = MatView(AP,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatView(APR, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatView(AN,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatView(ANR, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  /* Reorder original matrix */
  Mat             ARR;
  IS              rperm, cperm;
  PetscInt       *idx;
  const PetscInt *cidx;

  ierr = PetscMalloc(n * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  ierr = ISGetIndices(rpermpos, &cidx);CHKERRQ(ierr);
  for (i = 0; i < posSize; ++i) idx[i] = cidx[i];
  ierr = ISRestoreIndices(rpermpos, &cidx);CHKERRQ(ierr);
  ierr = ISGetIndices(rpermneg, &cidx);CHKERRQ(ierr);
  for (i = posSize; i < n; ++i) idx[i] = cidx[i-posSize] + posSize;
  ierr = ISRestoreIndices(rpermneg, &cidx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, idx, PETSC_OWN_POINTER, &rperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(rperm);CHKERRQ(ierr);
  ierr = PetscMalloc(n * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  ierr = ISGetIndices(cpermpos, &cidx);CHKERRQ(ierr);
  for (i = 0; i < posSize; ++i) idx[i] = cidx[i];
  ierr = ISRestoreIndices(cpermpos, &cidx);CHKERRQ(ierr);
  ierr = ISGetIndices(cpermneg, &cidx);CHKERRQ(ierr);
  for (i = posSize; i < n; ++i) idx[i] = cidx[i-posSize] + posSize;
  ierr = ISRestoreIndices(cpermneg, &cidx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, idx, PETSC_OWN_POINTER, &cperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(cperm);CHKERRQ(ierr);
  ierr = MatPermute(AR, rperm, cperm, &ARR);CHKERRQ(ierr);
  ierr = MatView(ARR, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&rperm);CHKERRQ(ierr);
  ierr = ISDestroy(&cperm);CHKERRQ(ierr);
  ierr = ISDestroy(&rpermpos);CHKERRQ(ierr);
  ierr = ISDestroy(&cpermpos);CHKERRQ(ierr);
  ierr = ISDestroy(&rpermneg);CHKERRQ(ierr);
  ierr = ISDestroy(&cpermneg);CHKERRQ(ierr);
  ierr = MatDestroy(&AP);CHKERRQ(ierr);
  ierr = MatDestroy(&AN);CHKERRQ(ierr);
  ierr = MatDestroy(&APR);CHKERRQ(ierr);
  ierr = MatDestroy(&ANR);CHKERRQ(ierr);
  /* Compare bands */
  Mat B, BR;

  ierr = MatCreateSubMatrixBanded(A,   50, 0.95, &B);CHKERRQ(ierr);
  ierr = MatCreateSubMatrixBanded(ARR, 50, 0.95, &BR);CHKERRQ(ierr);
  ierr = MatView(B,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatView(BR, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&BR);CHKERRQ(ierr);
  /* Cleanup */
  ierr = MatDestroy(&ARR);CHKERRQ(ierr);
  ierr = MatDestroy(&AR);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
