static char help[] = "Checks banded preconditioner with reordering\n";
/*
Matrices are at http://www.cise.ufl.edu/research/sparse/matrices/list_by_dimension.html

Matrix Market files can be converted using:

  src/mat/examples/tests/ex72 -fin matrix.mtx -fout matrix.bin

This code could be integrated.
*/

#include <petscksp.h>

typedef struct {
  char      matFilename[PETSC_MAX_PATH_LEN];
  char      matOrdtype[256];
  char      matOrdtype2[256];
  PetscBool bandedPreMat;
  PetscBool randomSol;
  PetscBool viewSol, viewMat;
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
  ierr = PetscStrncpy(options->matOrdtype,  MATORDERINGNATURAL, 256);CHKERRQ(ierr);
  ierr = PetscStrncpy(options->matOrdtype2, MATORDERINGNATURAL, 256);CHKERRQ(ierr);
  options->bandedPreMat   = PETSC_FALSE;
  options->randomSol      = PETSC_FALSE;
  options->viewSol        = PETSC_FALSE;
  options->viewMat        = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Reordering Test Options", "Mat");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat", "Path for matrix input file", "main", options->matFilename, options->matFilename, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must provide an input matrix using -mat <file>");
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_ordering_type", "Reordering for matrix", "main", ordlist, options->matOrdtype, tname, 256, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscStrncpy(options->matOrdtype, tname, 256);CHKERRQ(ierr);}
  ierr = PetscOptionsFList("-mat_ordering_type2", "Second reordering for matrix", "main", ordlist, options->matOrdtype2, tname, 256, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscStrncpy(options->matOrdtype2, tname, 256);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-banded_pre_mat", "Use a banded preconditioner matrix", "main", options->bandedPreMat, &options->bandedPreMat, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-random_exact_sol", "Use uniform random entries for exact solution u", "main", options->randomSol, &options->randomSol, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_exact_sol", "Output exact solution u", "main", options->viewSol, &options->viewSol, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_mat", "Output matrix A", "main", options->viewMat, &options->viewMat, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatGetOrdering_WBM(Mat, MatOrderingType, IS *, IS *);
extern PetscErrorCode MatGetOrdering_AWBM(Mat, MatOrderingType, IS *, IS *);

#undef __FUNCT__
#define __FUNCT__ "LoadModules"
PetscErrorCode LoadModules(AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatOrderingRegister("wbm",  MatGetOrdering_WBM);CHKERRQ(ierr);
  ierr = MatOrderingRegister("awbm", MatGetOrdering_AWBM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSubMatrixBanded"
/*@
  MatCreateSubMatrixBanded - Extract the banded subset B of A such that ||Vec(B)||_1 >= frac ||Vec(A)||_1

  Input Parameters:
+ A    - The matrix
. kmax - The maximum half-bandwidth, so 2k+1 diagonals may be extracted
- frac - The norm fraction limit for the extracted band

  Output Parameters:
+ kmax - The actual half-bandwidth of the extracted matrix 
. frac - The norm fraction of the extracted matrix
- B    - The banded submatrix

  Level: intermediate

.seealso: MatChop()
 @*/
PetscErrorCode MatCreateSubMatrixBanded(Mat A, PetscInt *kmax, PetscReal *frac, Mat *B)
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
  ierr = VecGetArray(weight, &w);CHKERRQ(ierr);
  for (k = 0; k < *kmax; ++k) {
    normB += w[k];
    if (normB >= (*frac)*normA) break;
  }
  ierr = VecRestoreArray(weight, &w);CHKERRQ(ierr);
  ierr = VecDestroy(&weight);CHKERRQ(ierr);
  /* Extract band */
  ierr = MatCreate(PetscObjectComm((PetscObject) A), B);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, m, n, M, N);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,&dnnz,m,&onnz);CHKERRQ(ierr);
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
  ierr = PetscMalloc2(maxcols,&newCols,maxcols,&newVals);CHKERRQ(ierr);
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
  *kmax = k;
  *frac = normB/normA;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  AppCtx         ctx;
  Mat            A;        /* system matrix */
  Mat            B;        /* preconditioner matrix (the preconditioner is built from this) */
  Vec            x, b, u;  /* approx solution, RHS, exact solution */
  KSP            ksp;      /* linear solver */
  PetscReal      error;    /* norm of solution error */
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args, (char *) 0, help);CHKERRQ(ierr);
  ierr = ProcessOptions(&ctx);CHKERRQ(ierr);
  ierr = LoadModules(&ctx);CHKERRQ(ierr);
  /* Load matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, ctx.matFilename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatLoad(A, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  /* Reorder matrix */
  if (ctx.matOrdtype) {
    Mat      PA;
    IS       rperm, cperm;
    PetscInt bw, pbw;

    ierr = MatGetOrdering(A, ctx.matOrdtype, &rperm, &cperm);CHKERRQ(ierr);
    ierr = MatPermute(A, rperm, cperm, &PA);CHKERRQ(ierr);
    ierr = ISDestroy(&rperm);CHKERRQ(ierr);
    ierr = ISDestroy(&cperm);CHKERRQ(ierr);
    ierr = MatComputeBandwidth(A, 0.0, &bw);CHKERRQ(ierr);
    ierr = MatComputeBandwidth(PA, 0.0, &pbw);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Reordered matrix using %s\n", ctx.matOrdtype);CHKERRQ(ierr);
    if (pbw < bw) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Reduced bandwidth from %d to %d\n", bw, pbw);CHKERRQ(ierr);}
    else          {ierr = PetscPrintf(PETSC_COMM_WORLD, "Increased bandwidth from %d to %d\n", bw, pbw);CHKERRQ(ierr);}
    if (ctx.viewMat) {
      ierr = MatView(A,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = MatView(PA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }
    {
      Vec       DA, DPA;
      PetscReal norm, pnorm;

      ierr = MatGetVecs(A, &DA, &DPA);CHKERRQ(ierr);
      ierr = MatGetDiagonal(A, DA);CHKERRQ(ierr);
      ierr = MatGetDiagonal(PA, DPA);CHKERRQ(ierr);
      ierr = VecNorm(DA,  NORM_1, &norm);CHKERRQ(ierr);
      ierr = VecNorm(DPA, NORM_1, &pnorm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of diagonal %g perm diagonal %g\n", norm, pnorm);CHKERRQ(ierr);
      ierr = VecView(DA,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = VecView(DPA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = VecDestroy(&DA);CHKERRQ(ierr);
      ierr = VecDestroy(&DPA);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = PA;
  }
  if (ctx.matOrdtype2) {
    Mat      PA;
    IS       rperm, cperm;
    PetscInt bw, pbw;

    ierr = MatGetOrdering(A, ctx.matOrdtype2, &rperm, &cperm);CHKERRQ(ierr);
    ierr = MatPermute(A, rperm, cperm, &PA);CHKERRQ(ierr);
    ierr = ISDestroy(&rperm);CHKERRQ(ierr);
    ierr = ISDestroy(&cperm);CHKERRQ(ierr);
    ierr = MatComputeBandwidth(A, 0.0, &bw);CHKERRQ(ierr);
    ierr = MatComputeBandwidth(PA, 0.0, &pbw);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Reordered matrix using %s\n", ctx.matOrdtype2);CHKERRQ(ierr);
    if (pbw < bw) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Reduced bandwidth from %d to %d\n", bw, pbw);CHKERRQ(ierr);}
    else          {ierr = PetscPrintf(PETSC_COMM_WORLD, "Increased bandwidth from %d to %d\n", bw, pbw);CHKERRQ(ierr);}
    if (ctx.viewMat) {
      ierr = MatView(A,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = MatView(PA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }
    {
      Vec       DA, DPA;
      PetscReal norm, pnorm;
      PetscScalar *a;
      PetscInt  numZeros = 0, n, r;

      ierr = MatGetVecs(A, &DA, &DPA);CHKERRQ(ierr);
      ierr = MatGetDiagonal(A, DA);CHKERRQ(ierr);
      ierr = MatGetDiagonal(PA, DPA);CHKERRQ(ierr);

      //ierr = MatSetOption(PA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
      //ierr = MatShift(PA, 0.0);CHKERRQ(ierr);
      ierr = VecGetLocalSize(DPA, &n);CHKERRQ(ierr);
      ierr = VecGetArray(DPA, &a);CHKERRQ(ierr);
      for (r = 0; r < n; ++r) {
        if (a[r] == 0.0) ++numZeros;
      }
      ierr = VecRestoreArray(DPA, &a);CHKERRQ(ierr);
      if (numZeros) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Zeros on the permuted diagonal: %d\n", numZeros);CHKERRQ(ierr);}

      ierr = VecNorm(DA,  NORM_1, &norm);CHKERRQ(ierr);
      ierr = VecNorm(DPA, NORM_1, &pnorm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of diagonal %g perm diagonal %g\n", norm, pnorm);CHKERRQ(ierr);
      ierr = VecView(DA,  PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = VecView(DPA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      ierr = VecDestroy(&DA);CHKERRQ(ierr);
      ierr = VecDestroy(&DPA);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = PA;
  }
  /* Create banded preconditioner */
  if (ctx.bandedPreMat) {
    PetscInt  k    = 50;
    PetscReal frac = 0.95;

    ierr = MatCreateSubMatrixBanded(A, &k, &frac, &B);CHKERRQ(ierr);
    if (ctx.viewMat) {ierr = MatView(B, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Bandwidth of 95%% band: %d norm fraction: %g\n", k, frac);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject) A);CHKERRQ(ierr);
    B    = A;
  }
  /* Create problem */
  ierr = MatGetVecs(A, &u, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &x);CHKERRQ(ierr);
  if (ctx.randomSol) {
    PetscRandom rctx;

    ierr = PetscRandomCreate(PetscObjectComm((PetscObject) u), &rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u, rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u, 1.0);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  if (ctx.viewSol) {ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  /* Create linear solver */
  ierr = KSPCreate(PetscObjectComm((PetscObject) A), &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, B);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
  /* Check the error */
  ierr = VecAXPY(x, -1.0, u);CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_2, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject) A), "Error in solution: %g\n", error);CHKERRQ(ierr);
  /* Cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
