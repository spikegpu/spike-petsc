#ifndef __MATBANDED_H
#define __MATBANDED_H
#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatCreateSubMatrixBanded(Mat, PetscInt *, PetscReal *, Mat *);

#endif /* __MATBANDED_H */
