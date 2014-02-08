static char help[] = "Checks banded preconditioner with reordering\n";

/*
./testbed2 -mat ../share/matrices/31770-lhs.bin -ksp_view -ksp_max_it 10 -ksp_converged_reason -ksp_monitor -pc_type fieldsplit -pc_fieldsplit_detect_saddle_point -pc_fieldsplit_type schur -pc_fieldsplit_schur_precondition full -pc_fieldsplit_schur_fact_type upper -fieldsplit_0_ksp_converged_reason -fieldsplit_0_pc_type ilu -fieldsplit_0_ksp_type reorder -fieldsplit_0_mat_ordering_type rcm -fieldsplit_0_reorder_pc_type lu -fieldsplit_0_reorder_ksp_type preonly -fieldsplit_1_ksp_converged_reason -fieldsplit_1_ksp_type gmres -fieldsplit_1_pc_type ilu 
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

extern PetscErrorCode PCCreate_Banded(PC);
extern PetscErrorCode KSPCreate_Reorder(KSP);

#undef __FUNCT__
#define __FUNCT__ "LoadModules"
PetscErrorCode LoadModules(AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatOrderingRegister("wbm",  MatGetOrdering_WBM);CHKERRQ(ierr);
  ierr = MatOrderingRegister("awbm", MatGetOrdering_AWBM);CHKERRQ(ierr);

  ierr = PCRegister("banded", PCCreate_Banded);CHKERRQ(ierr);
  ierr = KSPRegister("reorder", KSPCreate_Reorder);CHKERRQ(ierr);
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
  PetscBool      isSym;
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
  ierr = MatIsSymmetric(A, 1.0e-10, &isSym);CHKERRQ(ierr);
  if (isSym) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Matrix is symmetric\n");CHKERRQ(ierr);}
  else       {ierr = PetscPrintf(PETSC_COMM_WORLD, "Matrix is non-symmetric\n");CHKERRQ(ierr);}
  /* Things to make accessible from options
       - Banded PC (Done)
       - Reordering
     so that they can be accessible inside FieldSplit, and we can attack Dan's matrices the right way.
  */
  {
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
  ierr = KSPSetOperators(ksp, A, B, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
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
