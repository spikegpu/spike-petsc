#include <petscmat.h>
#include <petsc-private/matorderimpl.h>

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "CheckUnmatched"
static PetscErrorCode CheckUnmatched(PetscInt n, const PetscInt match[], const PetscInt matchR[])
{
  PetscInt       unc = 0, unr = 0, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (c = 0; c < n; ++c) {
    if (match[c]  <  0) ++unc;
    if (matchR[c] <  0) ++unr;
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "Unmatched columns %d rows %d\n", unc, unr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Ref: "ON ALGORITHMS FOR PERMUTING LARGE ENTRIES TO THE DIAGONAL OF A SPARSE MATRIX", I. S. DUFF and J. KOSTER, SIAM J. MATRIX ANAL. APPL.	22(4), pp. 973-996, 2001.
       "ALGORITHM 548: Solution of the Assignment Problem", GIORGIO CARPANETO and PAOLO TOTH, ACM TOMS, 6(1), 104-111, 1980. 

   Multiplicative max: c_ij = log a_j - log a_ij
   Additiive max:      c_ij = a_j - | a_ij |
   \bar c_ij = c_ij - u_i - v_j

  Initial guess:
  u_i = min_j c_ij
  v_j = min_i c_ij - u_i

  We scan the rows in COL(j) for each column node j to see whether it contains an unmatched row node i for which \bar c_ij = 0. If such a node i exists, edge (i,j)
  is added to the initial matching M. Then, for each remaining unmatched column node j, every row node i \in COL(j) is considered for which \bar c_ij = 0, and that
  is matched to a column node other than j, say, j1, so that (i,j1) \in M. If a row node i1 \in COL(j1) can be found that is not yet matched and for which
  \bar c_i1j1 = 0, then edge (i,j1) in M is replaced by edges (i,j) and (i1, j1).
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetOrdering_AWBM"
PETSC_EXTERN PetscErrorCode MatGetOrdering_AWBM(Mat A, MatOrderingType type, IS *permR, IS *permC)
{
  Vec *scalR, *scalC, scalRVec, scalCVec;
  scalR = &scalRVec; scalC = &scalCVec;

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
  PetscInt         debug = 0, r, c, r1, c1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(NULL, "-debug", &debug, NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A, NULL, &colMax);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A, colMax, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(n, &match, m, &matchR);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &p);CHKERRQ(ierr);
  ierr = PetscCalloc3(m, &u, n, &v, ia[n], &weights);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) match[c] = -1;
  /* Compute weights */
  ierr = VecGetArray(colMax, &a_j);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    for (r = ia[c]; r < ia[c+1]; ++r) {
      PetscReal ar = PetscAbsScalar(a[r]);

      if (ar == 0.0) weights[r] = PETSC_MAX_REAL;
      else           weights[r] = log(a_j[c]/ar);
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
  for (r = 0; r < m; ++r) matchR[r] = -1;
  /* Match columns */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    /* if (match[c] >= 0) continue; */
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Row %d\n  Weights:", c);CHKERRQ(ierr);}
    for (r = ia[c]; r < ia[c+1]; ++r) {
      PetscReal weight = weights[r] - u[ja[r]] - v[c];
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, " %g", weight);CHKERRQ(ierr);}
      if ((weight <= eps) && (matchR[ja[r]] < 0)) {
        if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Matched %d -- %d\n", c, ja[r]);CHKERRQ(ierr);}
        match[c]      = ja[r];
        matchR[ja[r]] = c;
        break;
      }
    }
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);}
  }
  /* Deal with unmatched columns */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
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
          if (debug) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "Replaced match %d -- %d\n", c1, ja[r]);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "  Added  match %d -- %d\n", c,  ja[r]);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "  Added  match %d -- %d\n", c1, ja[r1]);CHKERRQ(ierr);
          }
          match[c]       = ja[r];
          matchR[ja[r]]  = c;
          match[c1]      = ja[r1];
          matchR[ja[r1]] = c1;
          break;
        }
      }
      if (match[c] >= 0) break;
    }
  }
  /* Allow matching with non-optimal rows */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    if (match[c] >= 0) continue;
    for (r = ia[c]; r < ia[c+1]; ++r) {
      if (matchR[ja[r]] < 0) {
        if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Matched non-opt %d -- %d\n", c, ja[r]);CHKERRQ(ierr);}
        match[c]      = ja[r];
        matchR[ja[r]] = c;
        break;
      }
    }
  }
  /* Deal with non-optimal unmatched columns */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    if (match[c] >= 0) continue;
    for (r = ia[c]; r < ia[c+1]; ++r) {
      /* \bar c_ij = 0 and (r, j1) \in M */
      c1 = matchR[ja[r]];
      for (r1 = ia[c1]; r1 < ia[c1+1]; ++r1) {
        if (matchR[ja[r1]] < 0) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          if (debug) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "Replaced match %d -- %d\n", c1, ja[r]);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "  Added  match %d -- %d\n", c,  ja[r]);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "  Added  match %d -- %d\n", c1, ja[r1]);CHKERRQ(ierr);
          }
          match[c]       = ja[r];
          matchR[ja[r]]  = c;
          match[c1]      = ja[r1];
          matchR[ja[r1]] = c1;
          break;
        }
      }
      if (match[c] >= 0) break;
    }
  }
  /* Complete matching */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
  for (c = 0, r = 0; c < n; ++c) {
    if (match[c] >= n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d matched to invalid row %d", c, match[c]);
    if (match[c] <  0) {
      for (; r < n; ++r) {
        if (matchR[r] < 0) {
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Matched default %d -- %d\n", c, r);CHKERRQ(ierr);}
          match[c]  = r;
          matchR[r] = c;
          break;
        }
      }
    }
  }
  /* Check matching */
  ierr = CheckUnmatched(n, match, matchR);CHKERRQ(ierr);
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
  ierr = PetscFree2(match, matchR);CHKERRQ(ierr);
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

  ierr = VecDestroy(scalR);CHKERRQ(ierr);
  ierr = VecDestroy(scalC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
