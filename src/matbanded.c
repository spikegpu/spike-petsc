#include "matbanded.h"

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

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/

typedef struct {
  PetscInt  kmax, k; /* The maximum and actual half-bandwidth, so 2k+1 diagonals may be extracted */
  PetscReal frac, f; /* The norm fraction limit and actual for the extracted band */
  Mat       B;       /* The banded approximation */
  PC        pc;      /* The embedded PC */
} PC_Banded;

#undef __FUNCT__
#define __FUNCT__ "PCReset_Banded"
PetscErrorCode PCReset_Banded(PC pc)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);
  ierr = PCReset(b->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Banded"
PetscErrorCode PCDestroy_Banded(PC pc)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Banded(pc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCBandedSetMaxHalfBandwidth_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCBandedSetNormFraction_C",     NULL);CHKERRQ(ierr);
  ierr = PCDestroy(&b->pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_Banded"
PetscErrorCode PCSetFromOptions_Banded(PC pc)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Banded options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_banded_kmax", "Set the maximum half-bandwidth", "PCBandedSetMaxHalfBandwith", b->kmax, &b->kmax, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_banded_frac", "Set the norm fraction limit for the extracted band", "PCBandedSetNormFraction", b->frac, &b->frac, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PCSetFromOptions(b->pc);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Banded"
PetscErrorCode PCSetUp_Banded(PC pc)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pc->setupcalled == 0) {
    b->k = b->kmax;
    b->f = b->frac;
    ierr = MatCreateSubMatrixBanded(pc->pmat, &b->k, &b->f, &b->B);CHKERRQ(ierr);
    ierr = PetscInfo2(pc, "PCBANDED: half-bandwidth: %d norm fraction: %g\n", b->k, b->f);CHKERRQ(ierr);
    ierr = PCSetOperators(b->pc, pc->mat, b->B, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = PCSetUp(b->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_Banded"
PetscErrorCode PCApply_Banded(PC pc, Vec x, Vec y)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApply(b->pc, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_Banded"
PetscErrorCode PCView_Banded(PC pc, PetscViewer viewer)
{
  PC_Banded     *b = (PC_Banded *) pc->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Banded: k = %d (%d max), frac = %g (%g max)\n", b->k, b->kmax, b->f, b->frac);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PCView(b->pc, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBandedSetMaxHalfBandwidth_Banded"
static PetscErrorCode PCBandedSetMaxHalfBandwidth_Banded(PC pc, PetscInt kmax)
{
  PC_Banded *b = (PC_Banded *) pc->data;

  PetscFunctionBegin;
  b->kmax = kmax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBandedSetNormFraction_Banded"
static PetscErrorCode PCBandedSetNormFraction_Banded(PC pc, PetscReal frac)
{
  PC_Banded *b = (PC_Banded *) pc->data;

  PetscFunctionBegin;
  b->frac = frac;
  PetscFunctionReturn(0);
}

/*MC
  PCBANDED - Preconditioning with a banded approximation

  Options Database Key:
+ -pc_banded_kmax - the maximum half-bandwidth, so 2k+1 diagonals may be extracted
- -pc_banded_frac - the norm fraction limit for the extracted band

   Level: beginner

  Concepts: banded, preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBandedSetMaxHalfBandwith(), PCBandedSetNormFraction()
M*/
#undef __FUNCT__
#define __FUNCT__ "PCCreate_Banded"
PetscErrorCode PCCreate_Banded(PC pc)
{
  PC_Banded     *b;
  const char    *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc, &b);CHKERRQ(ierr);
  pc->data = (void *) b;

  b->kmax = 50;
  b->frac = 0.95;

  pc->ops->apply               = PCApply_Banded;
  pc->ops->applytranspose      = NULL;
  pc->ops->setup               = PCSetUp_Banded;
  pc->ops->reset               = PCReset_Banded;
  pc->ops->destroy             = PCDestroy_Banded;
  pc->ops->setfromoptions      = PCSetFromOptions_Banded;
  pc->ops->view                = PCView_Banded;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;

  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCBandedSetMaxHalfBandwidth_C", PCBandedSetMaxHalfBandwidth_Banded);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCBandedSetNormFraction_C",     PCBandedSetNormFraction_Banded);CHKERRQ(ierr);

  ierr = PCCreate(PetscObjectComm((PetscObject) pc), &b->pc);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) pc, &prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) b->pc, prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject) b->pc, "banded_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBandedSetMaxHalfBandwith"
/*@
  PCBandedSetMaxHalfBandwith - Set the maximum half-bandwidth, so 2k+1 diagonals may be extracted

  Logical Collective on PC

  Input Parameters:
+ pc - the preconditioner context
- k  - the maximum half-bandwidth

  Options Database Key:
. -pc_banded_kmax

  Level: intermediate

  Concepts: banded preconditioner

.seealso: PCBandedSetNormFraction()
@*/
PetscErrorCode PCBandedSetMaxHalfBandwith(PC pc, PetscInt kmax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc, "PCBandedSetMaxHalfBandwith_C", (PC, PetscInt), (pc, kmax));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBandedSetNormFraction"
/*@
  PCBandedSetNormFraction - Set the norm fraction limit for the extracted band

  Logical Collective on PC

  Input Parameters:
+ pc   - the preconditioner context
- frac - the norm fraction limit

  Options Database Key:
. -pc_banded_frac

  Level: intermediate

  Concepts: banded preconditioner

.seealso: PCBandedSetMaxHalfBandwith()
@*/
PetscErrorCode PCBandedSetNormFraction(PC pc, PetscReal frac)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc, "PCBandedSetMaxHalfBandwith_C", (PC, PetscReal), (pc, frac));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
