#include <petsc.h>

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/*
  Ref: "ON ALGORITHMS FOR PERMUTING LARGE ENTRIES TO THE DIAGONAL OF A SPARSE MATRIX", I. S. DUFF and J. KOSTER, SIAM J. MATRIX ANAL. APPL.	22(4), pp. 973-996, 2001.
       "ALGORITHM 548: Solution of the Assignment Problem", GIORGIO CARPANETO and PAOLO TOTH, ACM TOMS, 6(1), 104-111, 1980. 

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
#define __FUNCT__ "CheckUnmatched_SeqAIJ"
static PetscErrorCode CheckUnmatched_SeqAIJ(PetscInt n, const PetscInt match[], const PetscInt matchR[])
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
  /* Match columns */
  ierr = CheckUnmatched_SeqAIJ(n, match, matchR);CHKERRQ(ierr);
  for (r = 0; r < m; ++r) matchR[r] = -1;
  for (c = 0; c < n; ++c) {
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
  ierr = CheckUnmatched_SeqAIJ(n, match, matchR);CHKERRQ(ierr);
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
  ierr = CheckUnmatched_SeqAIJ(n, match, matchR);CHKERRQ(ierr);
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
  ierr = CheckUnmatched_SeqAIJ(n, match, matchR);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetLocalColumn_MPIAIJ"
PETSC_STATIC_INLINE PetscErrorCode MatGetLocalColumn_MPIAIJ(Mat_MPIAIJ *aij, PetscInt gcol, PetscInt *lcol)
{
#if defined(PETSC_USE_CTABLE)
  PetscErrorCode ierr;
  ierr = PetscTableFind(aij->colmap, gcol+1, lcol);CHKERRQ(ierr);
  (*lcol)--;
#else
  *lcol = aij->colmap[gcol] - 1;
#endif
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatComputeMatching_MPIAIJ"
static PetscErrorCode MatComputeMatching_MPIAIJ(Mat A, IS *permR, Vec *scalR, IS *permC, Vec *scalC)
{
  /* EVERYTHING IS WRITTEN AS IF THE MATRIX WERE COLUMN-MAJOR */
  MPI_Comm         comm = PetscObjectComm((PetscObject) A);
  Mat_MPIAIJ      *aij  = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ      *aijA = (Mat_SeqAIJ *) aij->A->data;
  Mat_SeqAIJ      *aijB = (Mat_SeqAIJ *) aij->B->data;
  PetscInt         n    = aij->A->rmap->n;     /* Number of local columns */
  PetscInt         m    = n + aij->B->cmap->n; /* Number of local rows */
  PetscInt        *match;                      /* The row matched to each column */
  PetscInt        *matchR;                     /* The column matched to each row */
  PetscInt        *p;                          /* The column permutation */
  const PetscInt  *iaA  = aijA->i;
  const PetscInt  *iaB  = aijB->i;
  const PetscInt  *jaA  = aijA->j;
  const PetscInt  *jaB  = aijB->j;
  const MatScalar *aA   = aijA->a;
  const MatScalar *aB   = aijB->a;
  Vec              uVec, uTmp, colMax;
  PetscScalar     *a_j, *sr, *sc;
  PetscReal       *weightsA, *weightsB, *u, *v, weight, weight1, eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt         rStart, lrow, lrow1, r, r1, c, c1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!aij->colmap) {ierr = MatCreateColmap_MPIAIJ_Private(A);CHKERRQ(ierr);}
  ierr = MatGetOwnershipRange(A, &rStart, NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A, &uTmp, &colMax);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A, colMax, NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(aij->lvec, &uVec);CHKERRQ(ierr);
  ierr = VecSet(uVec, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &match);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &matchR);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &p);CHKERRQ(ierr);
  ierr = PetscCalloc3(n, &v, iaA[n], &weightsA, iaB[n], &weightsB);CHKERRQ(ierr);
  ierr = VecGetArray(uVec, &u);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) match[c] = -1;
  /* Compute weights */
  ierr = VecGetArray(colMax, &a_j);CHKERRQ(ierr);
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      PetscReal ar = PetscAbsScalar(aA[r]);

      if (ar == 0.0) weightsA[r] = PETSC_MAX_REAL;
      else           weightsA[r] = log(a_j[c]/ar);
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      PetscReal ar = PetscAbsScalar(aB[r]);

      if (ar == 0.0) weightsB[r] = PETSC_MAX_REAL;
      else           weightsB[r] = log(a_j[c]/ar);
    }
  }
  CHKMEMQ;
  /* Compute local row weights */
  for (r = 0; r < m; ++r) u[r] = PETSC_MAX_REAL;
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      u[jaA[r]-rStart] = PetscMin(u[jaA[r]-rStart], weightsA[r]);
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r], &lrow);CHKERRQ(ierr);
      lrow += n;
      u[lrow] = PetscMin(u[lrow], weightsB[r]);
    }
  }
  CHKMEMQ;
  /* Reduce row weights */
  /* TODO Replace with PetscSF and MPI_MIN */
  ierr = VecScatterBegin(aij->Mvctx, uVec, uTmp, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(aij->Mvctx, uVec, uTmp, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(aij->Mvctx, uTmp, uVec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(aij->Mvctx, uTmp, uVec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  CHKMEMQ;
  /* Compute local column weights */
  for (c = 0; c < n; ++c) {
    v[c] = PETSC_MAX_REAL;
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      v[c] = PetscMin(v[c], weightsA[r] - u[jaA[r]-rStart]);
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r], &lrow);CHKERRQ(ierr);
      v[c] = PetscMin(v[c], weightsB[r] - u[n+lrow]);
    }
  }
  CHKMEMQ;
  /* Match columns */
  for (r = 0; r < m; ++r) matchR[r] = -1;
  for (c = 0; c < n; ++c) {
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      lrow   = jaA[r]-rStart;
      weight = weightsA[r] - u[lrow] - v[c];
      if ((weight <= eps) && (matchR[lrow] < 0)) {
        ierr = PetscSynchronizedPrintf(comm, "Matched diag %d -- %d\n", c, lrow);CHKERRQ(ierr);
        match[c]     = lrow;
        matchR[lrow] = c;
        break;
      }
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r], &lrow);CHKERRQ(ierr);
      lrow += n;
      weight = weightsB[r] - u[lrow] - v[c];
      if ((weight <= eps) && (matchR[lrow] < 0)) {
        ierr = PetscSynchronizedPrintf(comm, "Matched off diag %d -- %d\n", c, lrow);CHKERRQ(ierr);
        match[c]     = lrow;
        matchR[lrow] = c;
        break;
      }
    }
  }
  CHKMEMQ;
  /* Deal with unmatched columns */
  for (c = 0; c < n; ++c) {
    if (match[c] >= 0) continue;
    for (r = iaA[c]; r < iaA[c+1]; ++r) {
      lrow   = jaA[r]-rStart;
      weight = weightsA[r] - u[lrow] - v[c];
      if (weight > eps) continue;
      /* \bar c_ij = 0 and (r, j1) \in M */
      c1 = matchR[lrow];
      for (r1 = iaA[c1]; r1 < iaA[c1+1]; ++r1) {
        lrow1   = jaA[r1]-rStart;
        weight1 = weightsA[r1] - u[lrow1] - v[c1];
        if ((matchR[lrow1] < 0) && (weight1 <= eps)) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          ierr = PetscSynchronizedPrintf(comm, "Replaced d-d match %d -- %d\n", c1, lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  d-d match %d -- %d\n", c,  lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  d-d match %d -- %d\n", c1, lrow1);CHKERRQ(ierr);
          match[c]      = lrow;
          matchR[lrow]  = c;
          match[c1]     = lrow1;
          matchR[lrow1] = c1;
          break;
        }
      }
      for (r1 = iaB[c1]; r1 < iaB[c1+1]; ++r1) {
        ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r1], &lrow1);CHKERRQ(ierr);
        lrow1  += n;
        weight1 = weightsB[r1] - u[lrow1] - v[c1];
        if ((matchR[lrow1] < 0) && (weight1 <= eps)) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          ierr = PetscSynchronizedPrintf(comm, "Replaced d-o match %d -- %d\n", c1, lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  d-o match %d -- %d\n", c,  lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  d-o match %d -- %d\n", c1, lrow1);CHKERRQ(ierr);
          match[c]      = lrow;
          matchR[lrow]  = c;
          match[c1]     = lrow1;
          matchR[lrow1] = c1;
          break;
        }
      }
    }
    for (r = iaB[c]; r < iaB[c+1]; ++r) {
      ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r], &lrow);CHKERRQ(ierr);
      lrow  += n;
      weight = weightsB[r] - u[lrow] - v[c];
      if (weight > eps) continue;
      /* \bar c_ij = 0 and (r, j1) \in M */
      c1 = matchR[lrow];
      for (r1 = iaA[c1]; r1 < iaA[c1+1]; ++r1) {
        lrow1   = jaA[r1]-rStart;
        weight1 = weightsA[r1] - u[lrow1] - v[c1];
        if ((matchR[lrow1] < 0) && (weight1 <= eps)) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          ierr = PetscSynchronizedPrintf(comm, "Replaced o-d match %d -- %d\n", c1, lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  o-d match %d -- %d\n", c,  lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  o-d match %d -- %d\n", c1, lrow1);CHKERRQ(ierr);
          match[c]      = lrow;
          matchR[lrow]  = c;
          match[c1]     = lrow1;
          matchR[lrow1] = c1;
          break;
        }
      }
      for (r1 = iaB[c1]; r1 < iaB[c1+1]; ++r1) {
        ierr = MatGetLocalColumn_MPIAIJ(aij, jaB[r1], &lrow1);CHKERRQ(ierr);
        lrow1  += n;
        weight1 = weightsB[r1] - u[lrow1] - v[c1];
        if ((matchR[lrow1] < 0) && (weight1 <= eps)) {
          /* (r, c1) in M is replaced by (r, c) and (r1, c1) */
          ierr = PetscSynchronizedPrintf(comm, "Replaced o-o match %d -- %d\n", c1, lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  o-o match %d -- %d\n", c,  lrow);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Added  o-o match %d -- %d\n", c1, lrow1);CHKERRQ(ierr);
          match[c]      = lrow;
          matchR[lrow]  = c;
          match[c1]     = lrow1;
          matchR[lrow1] = c1;
          break;
        }
      }
    }
  }
  CHKMEMQ;
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  /* Complete matching */
  for (c = 0, r = 0; c < n; ++c) {
    if (match[c] >= n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %d matched to invalid row %d", c, match[c]);
    if (match[c] <  0) {
      for (; r < n; ++r) {
        if (matchR[r] < 0) {
          match[c]  = r;
          matchR[r] = c;
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
  ierr = VecRestoreArray(uVec, &u);CHKERRQ(ierr);
  ierr = VecDestroy(&uVec);CHKERRQ(ierr);
  ierr = VecDestroy(&uTmp);CHKERRQ(ierr);
  ierr = PetscFree3(v,weightsA,weightsB);CHKERRQ(ierr);
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
  else SETERRQ(PetscObjectComm((PetscObject) A), PETSC_ERR_ARG_WRONG, "Cannot compute matching for this matrix type");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMatrix"
static PetscErrorCode CreateMatrix(Mat *A)
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscInt       cols[2];
  PetscScalar    vals[2];
  PetscInt       N = 3, rStart, rEnd, r;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatCreate(PETSC_COMM_WORLD, A);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, "-filename", filename, PETSC_MAX_PATH_LEN-1, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    ierr = MatLoad(*A, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else {
    ierr = MatSetSizes(*A, PETSC_DETERMINE, PETSC_DETERMINE, N, N);CHKERRQ(ierr);
    ierr = MatSetUp(*A);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(*A, &rStart, &rEnd);CHKERRQ(ierr);
    /* From HC64 documentation */
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) *A), &rank);CHKERRQ(ierr);
    if (!rank) {
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
    }
  }
  /* Assemble */
  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(*A, NULL, "-mat_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMatrix"
static PetscErrorCode OutputMatrix(Mat A)
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetString(NULL, "-outfilename", filename, PETSC_MAX_PATH_LEN-1, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);CHKERRQ(ierr);
    ierr = MatView(A, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckDiagonalWeight"
static PetscErrorCode CheckDiagonalWeight(Mat A, const char name[])
{
  Vec            diag;
  PetscScalar   *a;
  PetscReal      weight = 0.0, gweight;
  PetscInt       r, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetVecs(A, NULL, &diag);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A, diag);CHKERRQ(ierr);
  ierr = VecGetLocalSize(diag, &n);CHKERRQ(ierr);
  ierr = VecGetArray(diag, &a);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) {
    weight += PetscAbsScalar(a[r]);
  }
  ierr = MPI_Allreduce(&weight, &gweight, 1, MPIU_REAL, MPI_PROD, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Diagonal weight for %s matrix: %g\n", name, gweight);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag, &a);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  Mat            A, P;
  IS             permR, permC;
  Vec            scalR, scalC;
  PetscBool      scale = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = CreateMatrix(&A);CHKERRQ(ierr);
  ierr = MatComputeMatching(A, &permR, &scalR, &permC, &scalC);CHKERRQ(ierr);
  //ierr = ISView(permR, NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(scalR, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(scalC, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, "-scale", &scale, NULL);CHKERRQ(ierr);
  if (scale) {ierr = MatDiagonalScale(A, scalR, scalC);CHKERRQ(ierr);}
  ierr = MatViewFromOptions(A, NULL, "-mat_view");CHKERRQ(ierr);
  ierr = MatPermute(A, permR, permC, &P);CHKERRQ(ierr);
  ierr = MatViewFromOptions(P, NULL, "-mat_view");CHKERRQ(ierr);
  ierr = CheckDiagonalWeight(A, scale ? "scaled" : "original");CHKERRQ(ierr);
  ierr = CheckDiagonalWeight(P, scale ? "permuted and scaled" : "permuted");CHKERRQ(ierr);
  ierr = ISDestroy(&permR);CHKERRQ(ierr);
  ierr = ISDestroy(&permC);CHKERRQ(ierr);
  ierr = VecDestroy(&scalR);CHKERRQ(ierr);
  ierr = VecDestroy(&scalC);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = OutputMatrix(P);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
