#include <petsc-private/kspimpl.h>   /*I "petscksp.h" I*/

typedef struct {
  KSP  ksp;             /* The embedded KSP */
  char ordertype[256];  /* The type of ordering */
  IS   rorder, corder;  /* The row and column orderings */
} KSP_Reorder;

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_Reorder"
PetscErrorCode KSPSetUp_Reorder(KSP ksp)
{
  KSP_Reorder   *r = (KSP_Reorder *) ksp->data;
  Mat            A, M, PA, PM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(ksp, &A, &M, NULL);CHKERRQ(ierr);
  ierr = MatGetOrdering(M, r->ordertype, &r->rorder, &r->corder);CHKERRQ(ierr);
  ierr = MatPermute(M, r->rorder, r->corder, &PM);CHKERRQ(ierr);
  if (A != M) {ierr = MatPermute(A, r->rorder, r->corder, &PA);CHKERRQ(ierr);}
  else        {PA   = PM;}
  ierr = KSPSetOperators(r->ksp, PA, PM, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(r->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&PM);CHKERRQ(ierr);
  if (A != M) {ierr = MatDestroy(&PA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_Reorder"
PetscErrorCode  KSPSolve_Reorder(KSP ksp)
{
#if 0
  KSP_Reorder   *r = (KSP_Reorder *) ksp->data;
  Vec            x, b;
  PetscBool      diagonalscale;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc, &diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject) ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  ierr = VecDuplicate(ksp->vec_sol, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(ksp->vec_rhs, &b);CHKERRQ(ierr);
  ierr = VecCopy(ksp->vec_sol, x);CHKERRQ(ierr);
  ierr = VecCopy(ksp->vec_rhs, b);CHKERRQ(ierr);
  ierr = VecPermute(x, r->corder, PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecPermute(b, r->rorder, PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPSolve(r->ksp, b, x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(r->ksp, &ksp->reason);CHKERRQ(ierr);
  ierr = VecCopy(x, ksp->vec_sol);CHKERRQ(ierr);
  ierr = VecPermute(ksp->vec_sol, r->corder, PETSC_TRUE);CHKERRQ(ierr);
  {
    Mat       A, PA, tmpA;
    Vec       rv;
    PetscReal res;

    ierr = VecDuplicate(r->ksp->vec_rhs, &rv);CHKERRQ(ierr);
    ierr = KSPGetOperators(r->ksp, &PA, NULL, NULL);CHKERRQ(ierr);
    ierr = MatMult(PA, r->ksp->vec_sol, rv);CHKERRQ(ierr);
    ierr = VecAXPY(rv, -1.0, r->ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "By Hand Permuted Residual: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    ierr = KSPBuildResidual(r->ksp, NULL, NULL, &rv);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "PETSc Permuted Residual: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    ierr = VecDuplicate(ksp->vec_rhs, &rv);CHKERRQ(ierr);
    ierr = KSPGetOperators(ksp, &A, NULL, NULL);CHKERRQ(ierr);
    ierr = MatMult(A, ksp->vec_sol, rv);CHKERRQ(ierr);
    ierr = VecAXPY(rv, -1.0, ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "By Hand Original Residual: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    ierr = KSPBuildResidual(ksp, NULL, NULL, &rv);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Original Residual: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    /* Check rhs */
    ierr = VecDuplicate(ksp->vec_rhs, &rv);CHKERRQ(ierr);
    ierr = VecCopy(ksp->vec_rhs, rv);CHKERRQ(ierr);
    ierr = VecPermute(rv, r->rorder, PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(rv, -1.0, r->ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Rhs Difference: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    /* Check sol */
    ierr = VecDuplicate(ksp->vec_sol, &rv);CHKERRQ(ierr);
    ierr = VecCopy(ksp->vec_sol, rv);CHKERRQ(ierr);
    ierr = VecPermute(rv, r->corder, PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(rv, -1.0, r->ksp->vec_sol);CHKERRQ(ierr);
    ierr = VecNorm(rv, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Sol Difference: %g\n", res);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);

    /* Check A */
    ierr = MatPermute(A, r->rorder, r->corder, &tmpA);CHKERRQ(ierr);
    ierr = MatAXPY(tmpA, -1.0, PA, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(tmpA, NORM_FROBENIUS, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "A Difference: %g\n", res);CHKERRQ(ierr);
    ierr = MatDestroy(&tmpA);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  KSP_Reorder   *r = (KSP_Reorder *) ksp->data;
  Vec            x = ksp->vec_sol;
  Vec            b = ksp->vec_rhs;
  PetscBool      diagonalscale;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc, &diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject) ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  ierr = VecPermute(x, r->corder, PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecPermute(b, r->rorder, PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPSolve(r->ksp, b, x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(r->ksp, &ksp->reason);CHKERRQ(ierr);
  ierr = VecPermute(x, r->corder, PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecPermute(b, r->rorder, PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_Reorder"
PetscErrorCode KSPSetFromOptions_Reorder(KSP ksp)
{
  KSP_Reorder      *r = (KSP_Reorder *) ksp->data;
  PetscFunctionList ordlist;
  char              tname[256];
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Reorder Options");CHKERRQ(ierr);
  ierr = PetscStrncpy(r->ordertype, MATORDERINGNATURAL, 256);CHKERRQ(ierr);
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_ordering_type", "Reordering for matrix", "main", ordlist, r->ordertype, tname, 256, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscStrncpy(r->ordertype, tname, 256);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(r->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_Reorder"
PetscErrorCode KSPView_Reorder(KSP ksp, PetscViewer viewer)
{
  KSP_Reorder   *r = (KSP_Reorder *) ksp->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  reordering type = %s\n", r->ordertype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(r->ksp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_Reorder"
PetscErrorCode KSPDestroy_Reorder(KSP ksp)
{
  KSP_Reorder   *r = (KSP_Reorder *) ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&r->rorder);CHKERRQ(ierr);
  ierr = ISDestroy(&r->corder);CHKERRQ(ierr);
  ierr = KSPDestroy(&r->ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  KSPREORDER - Solve of a non-symmetrically permuted system

  Level: beginner

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP
M*/

#undef __FUNCT__
#define __FUNCT__ "KSPCreate_Reorder"
PETSC_EXTERN PetscErrorCode KSPCreate_Reorder(KSP ksp)
{
  KSP_Reorder   *r;
  const char    *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp, &r);CHKERRQ(ierr);
  ksp->data = (void *) r;

  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_Reorder;
  ksp->ops->solve          = KSPSolve_Reorder;
  ksp->ops->destroy        = KSPDestroy_Reorder;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->view           = KSPView_Reorder;
  ksp->ops->setfromoptions = KSPSetFromOptions_Reorder;

  ierr = KSPCreate(PetscObjectComm((PetscObject) ksp), &r->ksp);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(r->ksp, prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(r->ksp, "reorder_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
