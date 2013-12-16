#include <petsc.h>

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/*
   Multiplicative max: c_ij = log a_j - log a_ij
   Additiive max:      c_ij = a_j - | a_ij |
   \bar c_ij = c_ij - u_i - v_j

  Initial guess:
  u_i = min_j c_ij
  v_j = min_i c_ij - u_i

This can be done by scanning the set COL(j) for each column node j to see whether it contains an unmatched row node i for which \bar c+ij = 0.
If such a node i exists, edge (i, j) is added to the initial matching M.

Then, for each remaining unmatched column node j,
  every row node i \in COL(j) is considered for which \bar cij = 0,
  and that is matched to a column node other than j, say, j1. So (i,j1) \in M.
  If a row node i1 \in COL(j1) can be found that is not yet matched and for which \bar c_i1j1 = 0, then edge (i,j1) in M is replaced by edges (i,j) and (i1, j1).
After having repeated this for all unmatched columns, the search for shortest augmenting paths starts with respect to the current matching.
*/
#undef __FUNCT__
#define __FUNCT__ "MatComputeMatching_SeqAIJ"
static PetscErrorCode MatComputeMatching_SeqAIJ(Mat A, IS *permR, Vec *scalR, IS *permC, Vec *scalC)
{
  /* EVERYTHING IS WRITTEN AS IF THE MATRIX WERE COLUMN-MAJOR */
  Mat_SeqAIJ      *aij = (Mat_SeqAIJ *) A->data;
  PetscInt         n   = A->rmap->n; /* Number of local columns */
  PetscInt         m   = A->cmap->n; /* Number of local rows */
  PetscInt        *match;            /* The row matched to each column, and inverse column permutation */
  PetscInt        *matchR;           /* The column matched to each row */
  PetscInt        *p;                /* The column permutation */
  const PetscInt  *ia  = aij->i;
  const PetscInt  *ja  = aij->j;
  const MatScalar *a   = aij->a;
  Vec              colMax;
  PetscScalar     *a_j, *sr, *sc;
  PetscReal       *weights /* c_ij */, *u /* u_i */, *v /* v_j */, eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt         r, c, r1, c1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatGetVecs(A, NULL, &colMax);CHKERRQ(ierr);
  ierr = MatGetRowMax(A, colMax, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &match);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &matchR);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &p);CHKERRQ(ierr);
  ierr = PetscCalloc3(m, &u, n, &v, ia[n], &weights);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) match[c] = -1;
  /* Compute weights */
  ierr = VecGetArray(colMax, &a_j);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    for (r = ia[c]; r < ia[c+1]; ++r) {
      weights[r] = log(a_j[c]/PetscRealPart(a[r]));
    }
  }
  /* Compute local row weights */
  for (r = 0; r < m; ++r) u[r] = PETSC_MAX_REAL;
  for (c = 0; c < n; ++c) {
    for (r = ia[c]; r < ia[c+1]; ++r) {
      u[ja[r]] = PetscMin(u[ja[r]], weights[r]);
    }
  }
  /* Compute local column weights */
  for (c = 0; c < n; ++c) {
    v[c] = PETSC_MAX_REAL;
    for (r = ia[c]; r < ia[c+1]; ++r) {
      v[c] = PetscMin(v[c], weights[r] - u[ja[r]]);
    }
  }
  /* Match columns */
  for (r = 0; r < m; ++r) matchR[r] = -1;
  for (c = 0; c < n; ++c) {
    for (r = ia[c]; r < ia[c+1]; ++r) {
      PetscReal weight = weights[r] - u[ja[r]] - v[c];
      if ((weight <= eps) && (matchR[ja[r]] < 0)) {
        match[c]      = ja[r];
        matchR[ja[r]] = c;
        break;
      }
    }
  }
  /* Deal with unmatched columns */
  for (c = 0; c < n; ++c) {
    if (match[c] >= 0) continue;
    for (r = ia[c]; r < ia[c+1]; ++r) {
      PetscReal weight = weights[r] - u[ja[r]] - v[c];
      if (weight > eps) continue;
      /* \bar c_ij = 0 and (r, j1) \in M */
      c1 = matchR[ja[r]];
      for (r1 = ia[c1]; r1 < ia[c1+1]; ++r1) {
        PetscReal weight1 = weights[r1] - u[ja[r1]] - v[c1];
        if ((matchR[ja[r1]] < 0) && (weight1 <= eps)) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          match[c]       = ja[r];
          matchR[ja[r]]  = c;
          match[c1]      = ja[r1];
          matchR[ja[r1]] = c1;
          break;
        }
      }
    }
  }
  /* Check matching */
  for (c = 0; c < n; ++c) {
    if (match[c] <  0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d unmatched", c);
    if (match[c] >= n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d matched to invalid row %d", c, match[c]);
  }
  /* Make permutation */
  for (c = 0; c < n; ++c) {p[match[c]] = c;}
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, p, PETSC_OWN_POINTER, permR);CHKERRQ(ierr);
  ierr = ISSetPermutation(*permR);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, permC);CHKERRQ(ierr);
  ierr = ISSetPermutation(*permC);CHKERRQ(ierr);
  ierr = PetscFree(match);CHKERRQ(ierr);
  ierr = PetscFree(matchR);CHKERRQ(ierr);
  /* Make scaling */
  ierr = VecCreateSeq(PETSC_COMM_SELF, n, scalR);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, n, scalC);CHKERRQ(ierr);
  ierr = VecGetArray(*scalR, &sr);CHKERRQ(ierr);
  ierr = VecGetArray(*scalC, &sc);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    sr[c] = PetscExpReal(v[c])/a_j[c];
    sc[c] = PetscExpReal(u[c]);
  }
  ierr = VecRestoreArray(*scalR, &sr);CHKERRQ(ierr);
  ierr = VecRestoreArray(*scalC, &sc);CHKERRQ(ierr);
  ierr = VecRestoreArray(colMax, &a_j);CHKERRQ(ierr);
  ierr = VecDestroy(&colMax);CHKERRQ(ierr);
  ierr = PetscFree3(u,v,weights);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatComputeMatching_MPIAIJ"
static PetscErrorCode MatComputeMatching_MPIAIJ(Mat A, IS *permR, Vec *scalR, IS *permC, Vec *scalC)
{
  /* EVERYTHING IS WRITTEN AS IF THE MATRIX WERE COLUMN-MAJOR */
  Mat_MPIAIJ      *aij  = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ      *aijA = (Mat_SeqAIJ *) aij->A->data;
  Mat_SeqAIJ      *aijB = (Mat_SeqAIJ *) aij->B->data;
  PetscInt         n    = aij->A->rmap->n; /* Number of local columns */
  PetscInt         m    = aij->B->cmap->n; /* Number of local rows */
  PetscInt        *match;                 /* The row matched to each column */
  const PetscInt  *iaA  = aijA->i;
  const PetscInt  *iaB  = aijB->i;
  const PetscInt  *jaA  = aijA->j;
  const PetscInt  *jaB  = aijB->j;
  const MatScalar *aA   = aijA->a;
  const MatScalar *aB   = aijB->a;
  Vec              colMax;
  PetscScalar     *a_j, *sr, *sc;
  PetscReal       *weightsA, *weightsB, *u, *v, eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt         rStart, lrow, r, c;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatGetVecs(A, NULL, &colMax);CHKERRQ(ierr);
  ierr = MatGetRowMax(A, colMax, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &match);CHKERRQ(ierr);
  ierr = PetscCalloc4(m, &u, n, &v, iaA[n], &weightsA, iaB[n], &weightsB);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) match[c] = -1;
  /* Compute weights */
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      weightsA[r] = log(a_j[c]/PetscRealPart(aA[r]));
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      weightsB[r] = log(a_j[c]/PetscRealPart(aB[r]));
    }
  }
  /* Compute local row weights */
  /* TODO: Must compute for both A and B */
  ierr = PetscMalloc2(n, &u, n, &v);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      u[jaA[r]-rStart] = PetscMin(u[jaA[r]-rStart], weightsA[c]);
    }
  }
  /* Reduce row weights */
  /* Compute local column weights */
  /* TODO: Must compute for both A and B */
  for (c = 0; c < n; ++c) {
    v[c] = PETSC_MAX_REAL;
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
#if defined(PETSC_USE_CTABLE)
      ierr = PetscTableFind(aij->colmap, jaA[r]+1, &lrow);CHKERRQ(ierr);
      lrow--;
#else
      lrow = aij->colmap[jaA[r]] - 1;
#endif
      v[c] = PetscMin(v[c], weightsA[r] - u[lrow]);
    }
  }
  /* Match columns */
  /* TODO: Must compute for both A and B */
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      if (weightsA[r] < eps) {match[c] = jaA[r]; break;}
    }
  }
  /* Deal with unmatched columns */
  /* TODO: Must compute for both A and B */
  for (c = 0; c < n; ++c) {
    if (match[c] >= 0) continue;
  }
  /* Check matching */
  for (c = 0; c < n; ++c) {
    if (match[c] <  0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d unmatched", c);
    if (match[c] >= n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d matched to invalid row %d", c, match[c]);
  }
  ierr = VecRestoreArray(colMax, &a_j);CHKERRQ(ierr);
  ierr = VecDestroy(&colMax);CHKERRQ(ierr);
  ierr = PetscFree4(u,v,weightsA,weightsB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatComputeMatching"
static PetscErrorCode MatComputeMatching(Mat A, IS *permR, Vec *scalR, IS *permC, Vec *scalC)
{
  PetscBool      isSeq, isMPI;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) A, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) A, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if      (isSeq) {ierr = MatComputeMatching_SeqAIJ(A, permR, scalR, permC, scalC);CHKERRQ(ierr);}
  else if (isMPI) {ierr = MatComputeMatching_MPIAIJ(A, permR, scalR, permC, scalC);CHKERRQ(ierr);}
  else SETERRQ(PetscObjectComm((PetscObject) A), PETSC_ERR_ARG_WRONG, "Cannot compute matching fir this matrix type");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMatrix"
static PetscErrorCode CreateMatrix(Mat *A)
{
  PetscInt       cols[2];
  PetscScalar    vals[2];
  PetscInt       n = 3, rStart, rEnd, r;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatCreate(PETSC_COMM_WORLD, A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A, n, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*A, &rStart, &rEnd);CHKERRQ(ierr);
  /* From HC64 documentation */
  r = 0;
  cols[0] = 1;   cols[1] = 2;
  vals[0] = 8.0; vals[1] = 3.0;
  ierr = MatSetValues(*A, 1, &r, 2, cols, vals, ADD_VALUES);CHKERRQ(ierr);
  r = 1;
  cols[0] = 1;   cols[1] = 2;
  vals[0] = 2.0; vals[1] = 1.0;
  ierr = MatSetValues(*A, 1, &r, 2, cols, vals, ADD_VALUES);CHKERRQ(ierr);
  r = 2;
  cols[0] = 0;
  vals[0] = 4.0;
  ierr = MatSetValues(*A, 1, &r, 1, cols, vals, ADD_VALUES);CHKERRQ(ierr);
  /* Assemble */
  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(*A, NULL, "-mat_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  Mat            A, P;
  IS             permR, permC;
  Vec            scalR, scalC;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = CreateMatrix(&A);CHKERRQ(ierr);
  ierr = MatComputeMatching(A, &permR, &scalR, &permC, &scalC);CHKERRQ(ierr);
  ierr = ISView(permR, NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(scalR, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(scalC, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = MatDiagonalScale(A, scalR, scalC);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A, NULL, "-mat_view");CHKERRQ(ierr);
  ierr = MatPermute(A, permR, permC, &P);CHKERRQ(ierr);
  ierr = MatViewFromOptions(P, NULL, "-mat_view");CHKERRQ(ierr);
  ierr = ISDestroy(&permR);CHKERRQ(ierr);
  ierr = ISDestroy(&permC);CHKERRQ(ierr);
  ierr = VecDestroy(&scalR);CHKERRQ(ierr);
  ierr = VecDestroy(&scalC);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
