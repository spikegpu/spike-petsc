#include <petscmat.h>
#include <petsc-private/matorderimpl.h>

extern void hslmc73_(PetscInt *n, PetscInt *lirn, const PetscInt *ia, const PetscInt *ja, const PetscScalar *a, PetscInt *order, PetscInt *inprof, PetscInt *outprof, PetscInt *inbw, PetscInt *outbw, PetscErrorCode *ierr);

/*
  MatGetOrdering_Fielder - Find the symmetric reordering given by the Fiedler vector of the graph. This is MC73 in the Harwell-Boeing library.
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetOrdering_Fiedler"
PETSC_EXTERN PetscErrorCode MatGetOrdering_Fiedler(Mat mat, MatOrderingType type, IS *row, IS *col)
{
  const PetscInt *ia, *ja;
  PetscScalar    *a;
  PetscInt       *perm, *iperm;
  PetscInt        nrow, nnz, i;
  PetscInt        inprof, outprof, inbw, outbw;
  PetscBool       done;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  nnz  = ia[nrow];
  if (!done) SETERRQ(PetscObjectComm((PetscObject) mat), PETSC_ERR_SUP, "Cannot get rows for matrix");
  ierr = MatSeqAIJGetArray(mat, &a);CHKERRQ(ierr);

  ierr = PetscMalloc1(nrow,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrow,&iperm);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PetscObjectComm((PetscObject) mat), PETSC_ERR_SUP, "Fielder using MC73 does not support complex numbers");
#else
  #if 0
  {
    PetscInt    exia[9]  = {1,  6,  8,  10,  12,  14,  15,  17,  18};
    PetscInt    exja[17] = {3,  1,  5,  6,  7,  2,  8,  3,  7,  4,  5,  6,  5,  8,  7,  8,  8};
    PetscScalar exa[17]  = {1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0};
    nrow = 8;
    nnz  = 17;
    hslmc73_(&nrow, &nnz, exia, exja, exa, perm, &inprof, &outprof, &inbw, &outbw, &ierr);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, " Permutation : ");CHKERRQ(ierr);
    for (i = 0; i < nrow; ++i) {ierr = PetscPrintf(PETSC_COMM_WORLD, "%4d", perm[i]);CHKERRQ(ierr);}
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }
  #else
  hslmc73_(&nrow, &nnz, ia, ja, a, perm, &inprof, &outprof, &inbw, &outbw, &ierr);CHKERRQ(ierr);
  #endif
#endif
  /* We have to invert this permutation, I hate you HSL */
  for (i = 0; i < nrow; ++i) iperm[perm[i]-1] = i;
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, " Profile before/after mc73   : %4d %4d\n", inprof, outprof);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, " Bandwidth before/after mc73 : %4d %4d\n", inbw, outbw);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, NULL, &ia, &ja, &done);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nrow, iperm, PETSC_OWN_POINTER, row);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) *row);CHKERRQ(ierr);
  *col = *row;
  PetscFunctionReturn(0);
}
