/* hsl_mc64d.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/
#include <petscsys.h>
#include <petsc-private/matorderimpl.h>

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode HSLmc64BD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num,
                                      PetscInt *jperm, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64SD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx,
                                      PetscInt *w, PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc, PetscInt *iw, PetscInt *iw4, PetscScalar *rlx, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64UD(PetscInt *id, PetscInt *mod, PetscInt *m, PetscInt *n, const PetscInt *irn, PetscInt *lirn, const PetscInt *ip, PetscInt *lenc,
                                      PetscInt *fc, PetscInt *iperm, PetscInt *num, PetscInt *numx, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode HSLmc64WD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num,
                                      PetscInt *jperm, PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *u, PetscScalar *d__, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64ZD(PetscInt *m, PetscInt *n, const PetscInt *irn, PetscInt *lirn, const PetscInt *ip, PetscInt *lenc, PetscInt *iperm, PetscInt *num,
                                      PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode HSLmc64XD(PetscInt *m, PetscInt *n, PetscInt *iperm, PetscInt *rw, PetscInt *cw);
PETSC_EXTERN PetscErrorCode mc64ID(PetscInt *icntl);
PETSC_EXTERN PetscErrorCode mc64AD(PetscInt *job, PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *num, PetscInt *cperm, PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw, PetscInt *icntl, PetscInt *info);
PETSC_EXTERN PetscErrorCode mc64BD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__);
PETSC_EXTERN PetscErrorCode mc64DD(PetscInt *i__, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64ED(PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64FD(PetscInt *pos0, PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64RD(PetscInt *n, PetscInt *ne, const PetscInt *ip, PetscInt *irn, PetscScalar *a);
PETSC_EXTERN PetscErrorCode mc64SD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx, PetscInt *w, PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc, PetscInt *iw, PetscInt *iw4);
PETSC_EXTERN PetscErrorCode mc64QD(const PetscInt *ip, PetscInt *lenl, PetscInt *lenh, PetscInt *w, PetscInt *wlen, PetscScalar *a, PetscInt *nval, PetscScalar *val);
PETSC_EXTERN PetscErrorCode mc64UD(PetscInt *id, PetscInt *mod, PetscInt *n, PetscInt *irn, PetscInt *lirn, PetscInt *ip, PetscInt *lenc, PetscInt *fc, PetscInt *iperm,
                                   PetscInt *num, PetscInt *numx, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode mc64WD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm, PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *u, PetscScalar *d__);
PETSC_EXTERN PetscErrorCode mc21AD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *iw);
PETSC_EXTERN PetscErrorCode mc21BD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscScalar fd15AD(char *t, int t_len);
#endif

#ifdef PETSC_USE_COMPLEX

PetscErrorCode HSLmc64AD(const PetscInt *job, PetscInt *m, PetscInt *n, 
                         PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *num, 
                         PetscInt *perm, PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw,
                         PetscInt *icntl, PetscScalar *cntl, PetscInt *info)
{
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "HSL MC64 does not support complex numbers");
}

#else

/* Table of constant values */

static PetscInt c__1 = 1;
static PetscInt c__2 = 2;
/* static PetscScalar c_b248 = 1.; */
typedef int ftnlen;
typedef struct {
  int cierr;
  int ciunit;
  int ciend;
  char *cifmt;
  int cirec;
} cilist;

/* COPYRIGHT (c) 2007 Science and Technology Facilities Council */
/*           and Jacko Koster (Trondheim, Norway) */

/* History: See ChangeLog */

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64AD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* Purpose */
/* ======= */

/* This subroutine attempts to find a permutation for an MxN, M>=N, */
/* sparse matrix A = {a_ij} that makes the permuted matrix have N */
/* entries on its diagonal. */
/* If the matrix is structurally nonsingular, the subroutine optionally */
/* returns a permutation that maximizes the smallest element on the */
/* diagonal, maximizes the sum of the diagonal entries, or maximizes */
/* the product of the diagonal entries of the permuted matrix. */
/* For the latter option, the subroutine also finds scaling factors that */
/* may be used to scale the matrix so that the nonzero diagonal entries */
/* of the permuted matrix are one in absolute value and all the */
/* off-diagonal entries are less than or equal to one in absolute value. */
/* The natural logarithms of the scaling factors u(i), i=1..M, for the */
/* rows and v(j), j=1..N, for the columns are returned so that the */
/* scaled matrix B = {b_ij} has entries b_ij = a_ij * EXP(u_i + v_j). */
/* The scaling factors are returned by this subroutine, but the actual */
/* scaling of the matrix has to be performed by the calling program. */

/* Parameters */
/* ========== */


/* JOB is an PETSCINT variable which must be set by the user to control */
/* the action. It is not altered by the subroutine. */
/* Possible values for JOB are: */
/*   1 Compute a column permutation of the matrix so that the */
/*     permuted matrix has as many entries on its diagonal as possible. */
/*     The values on the diagonal are of arbitrary size. HSL subroutine */
/*     MC21A/MC64Z is used for this. See [1]. */
/*   2 Compute a column permutation of the matrix so that the smallest */
/*     value on the diagonal of the permuted matrix is maximized. */
/*     See [3]. */
/*   3 Compute a column permutation of the matrix so that the smallest */
/*     value on the diagonal of the permuted matrix is maximized. */
/*     The algorithm differs from the one used for JOB = 2 and may */
/*     have quite a different performance. See [2]. */
/*   4 Compute a column permutation of the matrix so that the sum */
/*     of the diagonal entries of the permuted matrix is maximized. */
/*     See [3]. */
/*   5 Compute a column permutation of the matrix so that the product */
/*     of the diagonal entries of the permuted matrix is maximized */
/*     and vectors to scale the matrix so that the nonzero diagonal */
/*     entries of the permuted matrix are one in absolute value and */
/*     all the off-diagonal entries are less than or equal to one in */
/*     absolute value. See [3]. */
/*  Restriction: 1 <= JOB <= 5. */

/* M is an PETSCINT variable which must be set by the user to the */
/*   number of rows of the matrix A. It is not altered by the */
/*   subroutine. Restriction: M >= N. */

/* N is an PETSCINT variable which must be set by the user to the */
/*   number of columns of the matrix A. It is not altered by the */
/*   subroutine. Restriction: N >= 1. */

/* NE is an PETSCINT variable which must be set by the user to the */
/*   number of entries in the matrix. It is not altered by the */
/*   subroutine. Restriction: NE >= 1. */

/* IP is an PETSCINT array of length N+1. */
/*   IP(J), J=1..N, must be set by the user to the position in array IRN */
/*   of the first row index of an entry in column J. IP(N+1) must be set */
/*   to NE+1. It is not altered by the subroutine. */

/* IRN is an PETSCINT array of length NE. */
/*   IRN(K), K=1..NE, must be set by the user to hold the row indices of */
/*   the entries of the matrix. Those belonging to column J must be */
/*   stored contiguously in the positions IP(J)..IP(J+1)-1. The ordering */
/*   of the row indices within each column is unimportant. Repeated */
/*   entries are not allowed. The array IRN is not altered by the */
/*   subroutine. */

/* A is a DOUBLE PRECISION array of length NE. */
/*   The user must set A(K), K=1..NE, to the numerical value of the */
/*   entry that corresponds to IRN(K). */
/*   It is not used by the subroutine when JOB = 1. */
/*   It is not altered by the subroutine. */

/* NUM is an PETSCINT variable that need not be set by the user. */
/*   On successful exit, NUM will be the number of entries on the */
/*   diagonal of the permuted matrix. */
/*   If NUM < N, the matrix is structurally singular. */

/* PERM is an PETSCINT array of length M that need not be set by the */
/*   user. On successful exit, PERM can be interpreted in any of the */
/*   following ways: */

/*   1. If M=N, PERM contains the column permutation. */
/*      Column PERM(I) of the original matrix is column I in the */
/*      permuted matrix, I=1..N. */
/*      (This was the definition of parameter CPERM in versions of */
/*      MC64AD before version 1.2b) */

/*   2. If M>=N, PERM contains the row permutation. */
/*      Row I of the original matrix is row ABS(PERM(I)) in the */
/*      permuted matrix, I=1..M. */
/*      The rows where PERM(I) is positive constitute an N by N matrix */
/*      the scaled version of which has ones on the diagonal. */

/* LIW is an PETSCINT variable that must be set by the user to */
/*   the dimension of array IW. It is not altered by the subroutine. */
/*   Restriction: */
/*     JOB = 1 :  LIW >=  4N +  M */
/*     JOB = 2 :  LIW >=  2N + 2M */
/*     JOB = 3 :  LIW >=  8N + 2M + NE */
/*     JOB = 4 :  LIW >=  3N + 2M */
/*     JOB = 5 :  LIW >=  3N + 2M */

/* IW is an PETSCINT array of length LIW that is used for workspace. */

/* LDW is an PETSCINT variable that must be set by the user to the */
/*   dimension of array DW. It is not altered by the subroutine. */
/*   Restriction: */
/*     JOB = 1 :  LDW not used */
/*     JOB = 2 :  LDW >=      M */
/*     JOB = 3 :  LDW >=          NE */
/*     JOB = 4 :  LDW >=     2M + NE */
/*     JOB = 5 :  LDW >= N + 2M + NE */

/* DW is a REAL array of length LDW used for workspace. */
/*   If JOB = 5, on return, DW(i) contains u_i, i=1..M, and */
/*   DW(M+j) contains v_j, j=1..N. */

/* ICNTL is an PETSCINT array of length NICNTL. */
/*   Its components control the output of MC64AD and must be set by the */
/*   user before calling MC64AD. They are not altered by the subroutine. */

/*   ICNTL(1) must be set to specify the output stream for */
/*   error messages. If ICNTL(1) < 0, messages are suppressed. */

/*   ICNTL(2) must be set by the user to specify the output stream for */
/*   warning messages. If ICNTL(2) < 0, messages are suppressed. */

/*   ICNTL(3) must be set by the user to specify the output stream for */
/*   diagnostic messages. If ICNTL(3) < 0, messages are suppressed. */

/*   ICNTL(4) must be set by the user to a value other than 0 to avoid */
/*   checking of the input data.  Setting ICNTL(4) to any */
/*   other will avoid the checks but is likely to cause problems */
/*   later if out-of-range indices or duplicates are present. */
/*   The user should set ICNTL(4) nonzero, if the data is known not */
/*   to contain such problems. The code will exhibit undefined */
/*   behaviour in case data checking is not done and the */
/*   input data does not satisfy the restrictions as listed */
/*   elsewhere. */

/*   ICNTL(5) must be set by the user to control the printing of */
/*   diagnostic messages. */
/*   If ICNTL(5) <= 0, no messages are output. */
/*   If ICNTL(5) = 1, only error messages are output. */
/*   If ICNTL(5) = 2, error and warning messages output. */
/*   If ICNTL(5) = 3, as for 2 plus scalar parameters, the first */
/*   ten entries of array parameters, and the control parameters on */
/*   the first entry. */
/*   If ICNTL(5) > 3, full data will be printed on entry and exit. */

/* CNTL is a DOUBLE PRECISION array of length NCNTL. */
/*   Its components control the output of MC64AD and must be set by the */
/*   user before calling MC64AD. They are not altered by the subroutine. */

/*   CNTL(1) must be set to specify the relaxation parameter. */
/*   It is used by MC64 only if JOB = 3,4,5. */
/*   It must be set to a non-negative value (usually close to zero). */
/*   If CNTL(1) < 0.0, it is treated as 0.0. */

/*   CNTL(1) is a relaxation parameter. A positive value will lead to */
/*   matchings computed by MC64AD that are not optimal/maximal in some */
/*   sense but only nearly so. However, these non-optimal matchings are */
/*   often computed more quickly. Appropriate values for CNTL(1) are */
/*   problem dependent but usually slightly larger than 0.0. */


/* INFO is an PETSCINT array of length NINFO which need not be set by the */
/*   user. INFO(1) is set non-negative to indicate success. A negative */
/*   value is returned if an error occurred, a positive value if a */
/*   warning occurred. INFO(2) holds further information on the error. */
/*   On exit from the subroutine, INFO(1) will take one of the */
/*   following values: */
/*    0 : successful entry (for structurally nonsingular matrix). */
/*   +1 : successful entry (for structurally singular matrix). */
/*   +2 : the returned scaling factors are large and may cause */
/*        overflow when used to scale the matrix. */
/*        (For JOB = 4,5 entries only.) */
/*   +4 : CNTL(1) is negative and treated as zero. */
/*   -1 : JOB < 1 or JOB > 5.  Value of JOB held in INFO(2). */
/*   -2 : N < 1.  Value of invalid N held in INFO(2). */
/*   -3 : NE < 1.  Value of NE held in INFO(2). */
/*   -4 : M < N. Value of M held in INFO(2). */
/*   -6 : entries are found whose row indices are out of range. INFO(2) */
/*        contains the position in arrays A/IRN in which first entry is found. */
/*        (This value can be returned only if ICNTL(4) was set to zero.) */
/*   -7 : repeated entries are found. INFO(2) contains the position in arrays */
/*        A/IRN in which first entry is found. */
/*        (This value can be returned only if ICNTL(4) was set to zero.) */

/*   A return with one of the values INFO(1)=+3,+5,+6,+7 is also possible */
/*   These values are combinations of the above warnings (+1,+2,+4) and */
/*   correspond to the sum of the constituent warnings. */

/*   INFO(3) to INFO(NINFO) are not currently used and are set to zero */
/*        by the routine. */

/* References: */
/*  [1] I. S. Duff, (1981), */
/*      "Algorithm 575. Permutations for a zero-free diagonal", */
/*      ACM Trans. Math. Software 7(3), 387-390. */
/*  [2] I. S. Duff and J. Koster, (1998), */
/*      "The design and use of algorithms for permuting large */
/*      entries to the diagonal of sparse matrices", */
/*      SIAM J. Matrix Anal. Appl., vol. 20, no. 4, pp. 889-901. */
/*  [3] I. S. Duff and J. Koster, (2001), */
/*      "On algorithms for permuting large entries to the diagonal */
/*      of sparse matrices", */
/*      SIAM J. Matrix Anal. Appl., vol. 22, no. 4, pp. 973-996. */
/* Local variables and parameters */
/* External routines and functions */
/* Intrinsic functions */
/* Set RINF to largest positive real number (infinity) */
PetscErrorCode HSLmc64AD(const PetscInt *job, PetscInt *m, PetscInt *n, 
                         PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *num, 
                         PetscInt *perm, PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw,
                         PetscInt *icntl, PetscScalar *cntl, PetscInt *info)
{
  PetscErrorCode ierr;
#ifdef CHECKING
    /* Format strings */
    static char fmt_9001[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2,\002 because \002,(a),\002 = \002,i10)";
    static char fmt_9006[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        Column \002,i8,\002 contains an entry with inval"
	    "id row index \002,i8)";
    static char fmt_9007[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        Column \002,i8,\002 contains two or more entries"
	    " with row index \002,i8)";
    static char fmt_9020[] = "(\002 ****** Input parameters for MC64AD:\002"
	    "/\002 JOB =\002,i10/\002 M   =\002,i10/\002 N   =\002,i10/\002 N"
	    "E  =\002,i10)";
    static char fmt_9021[] = "(\002 IP(1:N+1)  = \002,8i8/(14x,8i8))";
    static char fmt_9022[] = "(\002 IRN(1:NE)  = \002,8i8/(14x,8i8))";
    static char fmt_9023[] = "(\002 A(1:NE)    = \002,4(1pd14.4)/(14x,4(1pd1"
	    "4.4)))";
    static char fmt_9024[] = "(\002 ICNTL(1:10)= \002,8i8/(14x,2i8))";
    static char fmt_9025[] = "(\002 CNTL(1)    = \002,1pd14.4)";
    static char fmt_9010[] = "(\002 ****** Warning from MC64A/AD. INFO(1) ="
	    " \002,i2)";
    static char fmt_9011[] = "(\002        - The matrix is structurally sing"
	    "ular.\002)";
    static char fmt_9012[] = "(\002        - Some scaling factors may be too"
	    " large.\002)";
    static char fmt_9014[] = "(\002        - CNTL(1) is negative and was tre"
	    "ated as zero.\002)";
    static char fmt_9030[] = "(\002 ****** Output parameters for MC64AD"
	    ":\002/\002 INFO(1:2)  = \002,2i8)";
    static char fmt_9031[] = "(\002 NUM        = \002,i8)";
    static char fmt_9032[] = "(\002 PERM(1:M)  = \002,8i8/(14x,8i8))";
    static char fmt_9033[] = "(\002 DW(1:M)    = \002,5(f11.3)/(14x,5(f11.3)"
	    "))";
    static char fmt_9034[] = "(\002 DW(M+1:M+N)= \002,5(f11.3)/(14x,5(f11.3)"
	    "))";
#endif

    /* System generated locals */
    PetscInt i__1, i__2, i__3;
    PetscScalar d__1, d__2;

    /* Builtin functions */
#ifdef CHECKING
    PetscInt s_wsfe(cilist *), do_fio(PetscInt *, char *, ftnlen), e_wsfe(void);
#endif

    /* Local variables */
    static PetscInt i__, j, k;
    static PetscScalar fact, rinf;
#ifdef CHECKING
    static PetscInt warn1, warn2, warn4;

    /* Fortran I/O blocks */
    static cilist io___5 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___6 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___7 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___8 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___12 = { 0, 0, 0, fmt_9006, 0 };
    static cilist io___13 = { 0, 0, 0, fmt_9007, 0 };
    static cilist io___14 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___15 = { 0, 0, 0, fmt_9021, 0 };
    static cilist io___16 = { 0, 0, 0, fmt_9022, 0 };
    static cilist io___17 = { 0, 0, 0, fmt_9023, 0 };
    static cilist io___18 = { 0, 0, 0, fmt_9021, 0 };
    static cilist io___19 = { 0, 0, 0, fmt_9022, 0 };
    static cilist io___20 = { 0, 0, 0, fmt_9023, 0 };
    static cilist io___21 = { 0, 0, 0, fmt_9024, 0 };
    static cilist io___22 = { 0, 0, 0, fmt_9025, 0 };
    static cilist io___24 = { 0, 0, 0, fmt_9010, 0 };
    static cilist io___25 = { 0, 0, 0, fmt_9011, 0 };
    static cilist io___26 = { 0, 0, 0, fmt_9012, 0 };
    static cilist io___27 = { 0, 0, 0, fmt_9014, 0 };
    static cilist io___28 = { 0, 0, 0, fmt_9030, 0 };
    static cilist io___29 = { 0, 0, 0, fmt_9031, 0 };
    static cilist io___30 = { 0, 0, 0, fmt_9032, 0 };
    static cilist io___31 = { 0, 0, 0, fmt_9033, 0 };
    static cilist io___32 = { 0, 0, 0, fmt_9034, 0 };
    static cilist io___33 = { 0, 0, 0, fmt_9032, 0 };
    static cilist io___34 = { 0, 0, 0, fmt_9033, 0 };
    static cilist io___35 = { 0, 0, 0, fmt_9034, 0 };
#endif
    extern PetscScalar huge_(PetscScalar *);

    PetscFunctionBegin;
    /* Parameter adjustments */
    --perm;
    --ip;
    --a;
    --irn;
    --iw;
    --dw;
    --icntl;
    --cntl;
    --info;

    /* Function Body */
    rinf = 1.79769313486231571E+308/*huge_(&rinf)*/;
    rinf /= *n;
#ifdef CHECKING
    warn1 = 0;
    warn2 = 0;
    warn4 = 0;
/* Check value of JOB */
    if (icntl[4] == 0) {
/* Check input data */
	if (*job < 1 || *job > 5) {
	    info[1] = -1;
	    info[2] = *job;
	    if (icntl[1] >= 0 && icntl[5] > 0) {
		io___5.ciunit = icntl[1];
		s_wsfe(&io___5);
		do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
		do_fio(&c__1, (char *)"JOB", (ftnlen)3);
		do_fio(&c__1, (char *)&(*job), (ftnlen)sizeof(PetscInt));
		e_wsfe();
	    }
	    goto L99;
	}
/* Check value of N */
	if (*n < 1) {
	    info[1] = -2;
	    info[2] = *n;
	    if (icntl[1] >= 0 && icntl[5] > 0) {
		io___6.ciunit = icntl[1];
		s_wsfe(&io___6);
		do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
		do_fio(&c__1, (char *)"N", (ftnlen)1);
		do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(PetscInt));
		e_wsfe();
	    }
	    goto L99;
	}
/* Check value of NE */
	if (*ne < 1) {
	    info[1] = -3;
	    info[2] = *ne;
	    if (icntl[1] >= 0 && icntl[5] > 0) {
		io___7.ciunit = icntl[1];
		s_wsfe(&io___7);
		do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
		do_fio(&c__1, (char *)"NE", (ftnlen)2);
		do_fio(&c__1, (char *)&(*ne), (ftnlen)sizeof(PetscInt));
		e_wsfe();
	    }
	    goto L99;
	}
/* Check value of M */
	if (*m < *n) {
	    info[1] = -4;
	    info[2] = *m;
	    if (icntl[1] >= 0 && icntl[5] > 0) {
		io___8.ciunit = icntl[1];
		s_wsfe(&io___8);
		do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
		do_fio(&c__1, (char *)"M", (ftnlen)1);
		do_fio(&c__1, (char *)&(*m), (ftnlen)sizeof(PetscInt));
		e_wsfe();
	    }
	    goto L99;
	}
/* Check LIW */
/*     IF (JOB.EQ.1) K = 4*N +   M */
/*     IF (JOB.EQ.2) K = 2*N + 2*M */
/*     IF (JOB.EQ.3) K = 8*N + 2*M + NE */
/*     IF (JOB.EQ.4) K = 3*N + 2*M */
/*     IF (JOB.EQ.5) K = 3*N + 2*M */
/*     IF (JOB.EQ.6) K = 3*N + 2*M + NE */
/*     IF (LIW.LT.K) THEN */
/*       INFO(1) = -5 */
/*       INFO(2) = K */
/*       IF (ICNTL(1).GE.0) WRITE(ICNTL(1),9004) INFO(1),K */
/*       GO TO 99 */
/*     ENDIF */
/* Check LDW; If JOB = 1, do not check */
/*     IF (JOB.GT.1) THEN */
/*       IF (JOB.EQ.2) K =       M */
/*       IF (JOB.EQ.3) K =           NE */
/*       IF (JOB.EQ.4) K =     2*M + NE */
/*       IF (JOB.EQ.5) K = N + 2*M + NE */
/*       IF (JOB.EQ.6) K = N + 3*M + NE */
/*       IF (LDW.LT.K) THEN */
/*         INFO(1) = -5 */
/*         INFO(2) = K */
/*         IF (ICNTL(1).GE.0) WRITE(ICNTL(1),9005) INFO(1),K */
/*         GO TO 99 */
/*       ENDIF */
/*     ENDIF */
    }
#endif
    if (icntl[4] == 0) {
/* Check row indices. Use IW(1:M) as workspace */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    iw[i__] = 0;
/* L3: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		i__ = irn[k];
#ifdef CHECKING
/* Check for row indices that are out of range */
		if (i__ < 1 || i__ > *m) {
		    info[1] = -6;
		    info[2] = k;
		    if (icntl[1] >= 0 && icntl[5] > 0) {
			io___12.ciunit = icntl[1];
			s_wsfe(&io___12);
			do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(
				PetscInt));
			do_fio(&c__1, (char *)&j, (ftnlen)sizeof(PetscInt));
			do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(PetscInt));
			e_wsfe();
		    }
		    goto L99;
		}
/* Check for repeated row indices within a column */
		if (iw[i__] == j) {
		    info[1] = -7;
		    info[2] = k;
		    if (icntl[1] >= 0 && icntl[5] > 0) {
			io___13.ciunit = icntl[1];
			s_wsfe(&io___13);
			do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(
				PetscInt));
			do_fio(&c__1, (char *)&j, (ftnlen)sizeof(PetscInt));
			do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(PetscInt));
			e_wsfe();
		    }
		    goto L99;
		} else {
		    iw[i__] = j;
		}
/* L4: */
#endif
	    }
/* L6: */
	}
    }
#ifdef CHECKING
/* Print diagnostics on input */
    if (icntl[5] > 2) {
	if (icntl[3] >= 0) {
	    io___14.ciunit = icntl[3];
	    s_wsfe(&io___14);
	    do_fio(&c__1, (char *)&(*job), (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)&(*m), (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)&(*ne), (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	    if (icntl[5] == 3) {
		io___15.ciunit = icntl[3];
		s_wsfe(&io___15);
/* Computing MIN */
		i__2 = 10, i__3 = *n + 1;
		i__1 = PetscMin(i__2,i__3);
		for (j = 1; j <= i__1; ++j) {
		    do_fio(&c__1, (char *)&ip[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		io___16.ciunit = icntl[3];
		s_wsfe(&io___16);
		i__2 = PetscMin(10,*ne);
		for (j = 1; j <= i__2; ++j) {
		    do_fio(&c__1, (char *)&irn[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		if (*job > 1) {
		    io___17.ciunit = icntl[3];
		    s_wsfe(&io___17);
		    i__2 = PetscMin(10,*ne);
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&a[j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		}
	    } else {
		io___18.ciunit = icntl[3];
		s_wsfe(&io___18);
		i__2 = *n + 1;
		for (j = 1; j <= i__2; ++j) {
		    do_fio(&c__1, (char *)&ip[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		io___19.ciunit = icntl[3];
		s_wsfe(&io___19);
		i__2 = *ne;
		for (j = 1; j <= i__2; ++j) {
		    do_fio(&c__1, (char *)&irn[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		if (*job > 1) {
		    io___20.ciunit = icntl[3];
		    s_wsfe(&io___20);
		    i__2 = *ne;
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&a[j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		}
	    }
	    io___21.ciunit = icntl[3];
	    s_wsfe(&io___21);
	    for (j = 1; j <= 10; ++j) {
		do_fio(&c__1, (char *)&icntl[j], (ftnlen)sizeof(PetscInt));
	    }
	    e_wsfe();
	    io___22.ciunit = icntl[3];
	    s_wsfe(&io___22);
	    do_fio(&c__1, (char *)&cntl[1], (ftnlen)sizeof(PetscScalar));
	    e_wsfe();
	}
    }
#endif
/* Set components of INFO to zero */
    for (i__ = 1; i__ <= 10; ++i__) {
	info[i__] = 0;
/* L8: */
    }
/* Compute maximum matching */
    if (*job == 1) {
/* Put length of column J in IW(J) */
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    iw[j] = ip[j + 1] - ip[j];
/* L10: */
	}
/* IW(N+1:3N+M+N) is workspace */
	ierr = HSLmc64ZD(m, n, &irn[1], ne, &ip[1], &iw[1], &perm[1], num, &iw[*n + 1], &iw[(*n << 1) + 1], &iw[*n * 3 + 1], &iw[*n * 3 + *m + 1]);CHKERRQ(ierr);
	goto L90;
    }
/* Compute bottleneck matching */
    if (*job == 2) {
/* Pass CNTL(1) to MC64B through DW(1) */
	dw[1] = PetscMax(0.,cntl[1]);
/* IW(1:2N+2M), DW(1:M) are workspaces */
	ierr = HSLmc64BD(m, n, ne, &ip[1], &irn[1], &a[1], &perm[1], num, &iw[1],
                     &iw[*n + 1], &iw[(*n << 1) + 1], &iw[(*n << 1) + *m + 1], &
                     dw[1], &rinf);CHKERRQ(ierr);
	goto L90;
    }
/* Compute bottleneck matching */
    if (*job == 3) {
/* Copy IRN(K) into IW(K), ABS(A(K)) into DW(K), K=1..NE */
	i__2 = *ne;
	for (k = 1; k <= i__2; ++k) {
	    iw[k] = irn[k];
	    dw[k] = (d__1 = a[k], fabs(d__1));
/* L20: */
	}
/* Sort entries in each column by decreasing value. */
	ierr = mc64RD(n, ne, &ip[1], &iw[1], &dw[1]);CHKERRQ(ierr);
/* Pass CNTL(1) to MC64S through FACT */
	fact = PetscMax(0.,cntl[1]);
/* IW(NE+1:NE+5N+M+(3N+M)) is workspace */
	ierr = HSLmc64SD(m, n, ne, &ip[1], &iw[1], &dw[1], &perm[1], num, &iw[*ne + 1], &iw[*ne + *n + 1], &iw[*ne + (*n << 1) + 1],
                     &iw[*ne + *n * 3 + 1], &iw[*ne + (*n << 2) + 1], &iw[*ne + *n * 5 + 1],
                     &iw[*ne + *n * 5 + *m + 1], &fact, &rinf);CHKERRQ(ierr);
	goto L90;
    }
    if (*job == 4) {
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    fact = 0.;
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		if ((d__1 = a[k], fabs(d__1)) > fact) {
		    fact = (d__2 = a[k], fabs(d__2));
		}
/* L30: */
	    }
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		dw[(*m << 1) + k] = fact - (d__1 = a[k], fabs(d__1));
/* L40: */
	    }
/* L50: */
	}
/* B = DW(2M+1:2M+NE); IW(1:3N+2M) and DW(1:2M) are workspaces */
/* Pass CNTL(1) to MC64W through DW(1) */
/* Pass JOB to MC64W through IW(1) */
	dw[1] = PetscMax(0.,cntl[1]);
	iw[1] = *job;
/* Call MC64W */
	ierr = HSLmc64WD(m, n, ne, &ip[1], &irn[1], &dw[(*m << 1) + 1], &perm[1],
                     num, &iw[1], &iw[*n + 1], &iw[(*n << 1) + 1], &iw[*n * 3 + 1],
                     &iw[*n * 3 + *m + 1], &dw[1], &dw[*m + 1], &rinf);CHKERRQ(ierr);
	goto L90;
    }
    if (*job == 5 || *job == 6) {
	if (*job == 5) {
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		fact = 0.;
		i__3 = ip[j + 1] - 1;
		for (k = ip[j]; k <= i__3; ++k) {
		    dw[(*m << 1) + *n + k] = (d__1 = a[k], fabs(d__1));
		    if (dw[(*m << 1) + *n + k] > fact) {
			fact = dw[(*m << 1) + *n + k];
		    }
/* L60: */
		}
		dw[(*m << 1) + j] = fact;
/* CC Significant change made here so that column with single */
/*   zero gets set to RINF and not 1.0 */
		if (fact != 0.) {
		    fact = log(fact);
		} else {
		    fact = rinf;
		}
		i__3 = ip[j + 1] - 1;
		for (k = ip[j]; k <= i__3; ++k) {
		    if (dw[(*m << 1) + *n + k] != 0.) {
			dw[(*m << 1) + *n + k] = fact - log(dw[(*m << 1) + *n 
				+ k]);
		    } else {
/*                  write(*,*) 'set diag to ',RINF */
			dw[(*m << 1) + *n + k] = rinf;
/* 5.0D+14 */
/* *RINF/(N+1) */
		    }
/* L70: */
		}
/*           ELSE */
/*             DO 71 K = IP(J),IP(J+1)-1 */
/*               DW(2*M+N+K) = ONE */
/*  71         CONTINUE */
/*           ENDIF */
/* L75: */
	    }
	}
/*       IF (JOB.EQ.6) THEN */
/*         DO 175 K = 1,NE */
/*           IW(3*N+2*M+K) = IRN(K) */
/*           DW(2*M+N+K) = ABS(A(K)) */
/* 175     CONTINUE */
/*         DO 61 I = 1,M */
/*           DW(2*M+N+NE+I) = ZERO */
/*  61     CONTINUE */
/*         DO 63 J = 1,N */
/*           DO 62 K = IP(J),IP(J+1)-1 */
/*             I = IRN(K) */
/*             IF (DW(2*M+N+K).GT.DW(2*M+N+NE+I)) THEN */
/*               DW(2*M+N+NE+I) = DW(2*M+N+K) */
/*             ENDIF */
/*  62       CONTINUE */
/*  63     CONTINUE */
/*         DO 64 I = 1,M */
/*           IF (DW(2*M+N+NE+I).NE.ZERO) THEN */
/*             DW(2*M+N+NE+I) = 1/DW(2*M+N+NE+I) */
/*           ENDIF */
/*  64     CONTINUE */
/*         DO 66 J = 1,N */
/*           DO 65 K = IP(J),IP(J+1)-1 */
/*             I = IRN(K) */
/*             DW(2*M+N+K) = DW(2*M+N+NE+I) * DW(2*M+N+K) */
/*  65       CONTINUE */
/*  66     CONTINUE */
/*         CALL MC64RD(N,NE,IP,IW(3*N+2*M+1),DW(2*M+N+1)) */
/*         DO 176 J = 1,N */
/*           IF (IP(J).NE.IP(J+1)) THEN */
/*             FACT = DW(2*M+N+IP(J)) */
/*           ELSE */
/*             FACT = ZERO */
/*           ENDIF */
/*           DW(2*M+J) = FACT */
/*           IF (FACT.NE.ZERO) THEN */
/*             FACT = LOG(FACT) */
/*             DO 170 K = IP(J),IP(J+1)-1 */
/*               IF (DW(2*M+N+K).NE.ZERO) THEN */
/*                 DW(2*M+N+K) = FACT - LOG(DW(2*M+N+K)) */
/*               ELSE */
/*                  write(*,*) 'set diag to ',RINF */
/*                 DW(2*M+N+K) = RINF */
/* 5.0D+14 */
/* 0.5* RINF/(N+1) */
/*               ENDIF */
/* 170         CONTINUE */
/*           ELSE */
/*             DO 171 K = IP(J),IP(J+1)-1 */
/*               DW(2*M+N+K) = ONE */
/* 171         CONTINUE */
/*           ENDIF */
/* 176     CONTINUE */
/*       ENDIF */
/* Pass CNTL(1) to MC64W through DW(1) */
/* Pass JOB to MC64W through IW(1) */
	dw[1] = PetscMax(0.,cntl[1]);
	iw[1] = *job;
/* Call MC64W */
	if (*job == 5) {
	    ierr = HSLmc64WD(m, n, ne, &ip[1], &irn[1], &dw[(*m << 1) + *n + 1], 
                         &perm[1], num, &iw[1], &iw[*n + 1], &iw[(*n << 1) + 1], &
                         iw[*n * 3 + 1], &iw[*n * 3 + *m + 1], &dw[1], &dw[*m + 1],
                         &rinf);CHKERRQ(ierr);
	}
/*       IF (JOB.EQ.6) THEN */
/*         CALL MC64_HSL_WD(M,N,NE,IP,IW(3*N+2*M+1),DW(2*M+N+1),PERM,NUM, */
/*    &         IW(1),IW(N+1),IW(2*N+1),IW(3*N+1),IW(3*N+M+1), */
/*    &         DW(1),DW(M+1),RINF) */
/*       ENDIF */
/*       IF (JOB.EQ.6) THEN */
/*         DO 79 I = 1,M */
/*           IF (DW(2*M+N+NE+I).NE.0) THEN */
/*             DW(I) = DW(I) + LOG(DW(2*M+N+NE+I)) */
/*           ENDIF */
/*  79     CONTINUE */
/*       ENDIF */
	if (*num == *n) {
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		if (dw[(*m << 1) + j] != 0.) {
		    dw[*m + j] -= log(dw[(*m << 1) + j]);
		} else {
		    dw[*m + j] = 0.;
		}
/* L80: */
	    }
	}
/* Check size of row and column scaling factors */
	fact = log(rinf) * .5f;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    if (dw[*m + j] < fact) {
		goto L86;
	    }
#ifdef CHECKING
	    warn2 = 2;
#endif
/* Scaling factor is large, return with warning */
	    goto L90;
L86:
	    ;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (dw[i__] < fact) {
		goto L87;
	    }
#ifdef CHECKING
	    warn2 = 2;
#endif
/* Scaling factor is large, return with warning */
	    goto L90;
L87:
	    ;
	}
/*       GO TO 90 */
    }
/* If matrix is structurally singular, return with warning */
L90:
#ifdef CHECKING
    if (*num < *n) {
	warn1 = 1;
    }
/* If CNTL(1) is negative and treated as zero, return with warning */
    if (*job == 4 || *job == 5 || *job == 6) {
	if (cntl[1] < 0.) {
	    warn4 = 4;
	}
    }
/* Set warning flag and print warnings (only if no errors were found) */
    if (info[1] == 0) {
	info[1] = warn1 + warn2 + warn4;
	if (info[1] > 0 && icntl[2] >= 0 && icntl[5] > 1) {
	    io___24.ciunit = icntl[2];
	    s_wsfe(&io___24);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	    if (warn1 == 1) {
		io___25.ciunit = icntl[2];
		s_wsfe(&io___25);
		e_wsfe();
	    }
	    if (warn2 == 2) {
		io___26.ciunit = icntl[2];
		s_wsfe(&io___26);
		e_wsfe();
	    }
	    if (warn4 == 4) {
		io___27.ciunit = icntl[2];
		s_wsfe(&io___27);
		e_wsfe();
	    }
	}
    }
/* Print diagnostics on output */
    if (icntl[5] > 2) {
	if (icntl[3] >= 0) {
	    io___28.ciunit = icntl[3];
	    s_wsfe(&io___28);
	    for (j = 1; j <= 2; ++j) {
		do_fio(&c__1, (char *)&info[j], (ftnlen)sizeof(PetscInt));
	    }
	    e_wsfe();
	    io___29.ciunit = icntl[3];
	    s_wsfe(&io___29);
	    do_fio(&c__1, (char *)&(*num), (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	    if (icntl[5] == 3) {
		io___30.ciunit = icntl[3];
		s_wsfe(&io___30);
		i__2 = PetscMin(10,*m);
		for (j = 1; j <= i__2; ++j) {
		    do_fio(&c__1, (char *)&perm[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		if (*job == 5 || *job == 6) {
		    io___31.ciunit = icntl[3];
		    s_wsfe(&io___31);
		    i__2 = PetscMin(10,*m);
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&dw[j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		    io___32.ciunit = icntl[3];
		    s_wsfe(&io___32);
		    i__2 = PetscMin(10,*n);
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&dw[*m + j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		}
	    } else {
		io___33.ciunit = icntl[3];
		s_wsfe(&io___33);
		i__2 = *m;
		for (j = 1; j <= i__2; ++j) {
		    do_fio(&c__1, (char *)&perm[j], (ftnlen)sizeof(PetscInt));
		}
		e_wsfe();
		if (*job == 5 || *job == 6) {
		    io___34.ciunit = icntl[3];
		    s_wsfe(&io___34);
		    i__2 = *m;
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&dw[j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		    io___35.ciunit = icntl[3];
		    s_wsfe(&io___35);
		    i__2 = *n;
		    for (j = 1; j <= i__2; ++j) {
			do_fio(&c__1, (char *)&dw[*m + j], (ftnlen)sizeof(
				PetscScalar));
		    }
		    e_wsfe();
		}
	    }
	}
    }
#endif
/* Return from subroutine. */
#ifdef CHECKING
L99:
#endif
    PetscFunctionReturn(0);
/* L9004: */
/* L9005: */
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64BD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* N, NE, IP, IRN are described in MC64AD. */
/* A is a DOUBLE PRECISION array of length NE. */
/*   A(K), K=1..NE, must be set to the value of the entry */
/*   that corresponds to IRN(K). It is not altered. */
/* IPERM is an PETSCINT array of length M. On exit, it contains the */
/*    matching: IPERM(I) = 0 or row I is matched to column IPERM(I). */
/* NUM is PETSCINT variable. On exit, it contains the cardinality of the */
/*    matching stored in IPERM. */
/* D is a DOUBLE PRECISION work array of length M. */
/*    On entry, D(1) contains the relaxation parameter RLX. */
/* RINF is the largest positive real number */
/* Local variables */
/* Local parameters */
/* External subroutines and/or functions */
PetscErrorCode HSLmc64BD(PetscInt *m, PetscInt *n, PetscInt *ne, 
                         const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *
                         num, PetscInt *jperm, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__,
                         PetscScalar *rinf)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3;
    PetscScalar d__1, d__2, d__3;

    /* Local variables */
    static PetscInt i__, j, k;
    static PetscScalar a0;
    static PetscInt i0, q0;
    static PetscScalar ai, di;
    static PetscInt ii, jj, kk;
    static PetscScalar bv;
    static PetscInt up;
    static PetscScalar dq0;
    static PetscInt kk1, kk2;
    static PetscScalar csp;
    static PetscInt isp, jsp;
    static PetscScalar tbv;
    static PetscInt low;
    static PetscScalar rlx, dnew;
    static PetscInt jord, qlen, idum, jdum, lpos;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --d__;
    --l;
    --q;
    --iperm;
    --pr;
    --jperm;
    --ip;
    --a;
    --irn;

    /* Function Body */
    rlx = d__[1];
/* Initialization */
    *num = 0;
    bv = *rinf;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	jperm[k] = 0;
	pr[k] = ip[k];
/* L10: */
    }
    i__1 = *m;
    for (k = 1; k <= i__1; ++k) {
	iperm[k] = 0;
	d__[k] = 0.;
/* L12: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a0 = -1.;
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    ai = (d__1 = a[k], fabs(d__1));
	    if (ai > d__[i__]) {
		d__[i__] = ai;
	    }
	    if (jperm[j] != 0) {
		goto L20;
	    }
	    if (ai >= bv) {
		a0 = bv;
		if (iperm[i__] != 0) {
		    goto L20;
		}
		jperm[j] = i__;
		iperm[i__] = j;
		++(*num);
	    } else {
		if (ai <= a0) {
		    goto L20;
		}
		a0 = ai;
		i0 = i__;
	    }
L20:
	    ;
	}
	if (a0 != -1. && a0 < bv) {
	    bv = a0;
	    if (iperm[i0] != 0) {
		goto L30;
	    }
	    iperm[i0] = j;
	    jperm[j] = i0;
	    ++(*num);
	}
L30:
	;
    }
    if (*m == *n) {
/* Update BV with smallest of all the largest maximum absolute values */
/* of the rows. D(I) contains the largest absolute value in row I. */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MIN */
	    d__1 = bv, d__2 = d__[i__];
	    bv = PetscMin(d__1,d__2);
/* L35: */
	}
    }
/* Shortcut if all columns are matched at this stage. */
    if (*num == *n) {
	goto L1000;
    }
/* Rescan unassigned columns; improve initial assignment */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (jperm[j] != 0) {
	    goto L95;
	}
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    ai = (d__1 = a[k], fabs(d__1));
	    if (ai < bv) {
		goto L50;
	    }
	    if (iperm[i__] == 0) {
		goto L90;
	    }
	    jj = iperm[i__];
	    kk1 = pr[jj];
	    kk2 = ip[jj + 1] - 1;
	    if (kk1 > kk2) {
		goto L50;
	    }
	    i__3 = kk2;
	    for (kk = kk1; kk <= i__3; ++kk) {
		ii = irn[kk];
		if (iperm[ii] != 0) {
		    goto L70;
		}
		if ((d__1 = a[kk], fabs(d__1)) >= bv) {
		    goto L80;
		}
L70:
		;
	    }
	    pr[jj] = kk2 + 1;
L50:
	    ;
	}
	goto L95;
L80:
	jperm[jj] = ii;
	iperm[ii] = jj;
	pr[jj] = kk + 1;
L90:
	++(*num);
	jperm[j] = i__;
	iperm[i__] = j;
	pr[j] = k + 1;
L95:
	;
    }
/* Shortcut if all columns are matched at this stage. */
    if (*num == *n) {
	goto L1000;
    }
/* Prepare for main loop */
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = -1.;
	l[i__] = 0;
/* L99: */
    }
/* TBV is a relaxed value of BV (ie TBV is slightly smaller than BV). */
    tbv = bv * (1 - rlx);
/* Main loop ... each pass round this loop is similar to Dijkstra's */
/* algorithm for solving the single source shortest path problem */
    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {
	if (jperm[jord] != 0) {
	    goto L100;
	}
	qlen = 0;
	low = *m + 1;
	up = *m + 1;
/* CSP is cost of shortest path to any unassigned row */
/* ISP is matrix position of unassigned row element in shortest path */
/* JSP is column index of unassigned row element in shortest path */
	csp = -1.;
/* Build shortest path tree starting from unassigned column JORD */
	j = jord;
	pr[j] = -1;
/* Scan column J */
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    dnew = (d__1 = a[k], fabs(d__1));
	    if (csp >= dnew) {
		goto L115;
	    }
	    if (iperm[i__] == 0) {
/* Row I is unassigned; update shortest path info */
		csp = dnew;
		isp = i__;
		jsp = j;
		if (csp >= tbv) {
		    goto L160;
		}
	    } else {
		d__[i__] = dnew;
		if (dnew >= tbv) {
/* Add row I to Q2 */
		    --low;
		    q[low] = i__;
		} else {
/* Add row I to Q, and push it */
		    ++qlen;
		    l[i__] = qlen;
		    ierr = mc64DD(&i__, m, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		}
		jj = iperm[i__];
		pr[jj] = j;
	    }
L115:
	    ;
	}
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
/* If Q2 is empty, extract new rows from Q */
	    if (low == up) {
		if (qlen == 0) {
		    goto L160;
		}
		i__ = q[1];
		if (csp >= d__[i__]) {
		    goto L160;
		}
		bv = d__[i__];
		tbv = bv * (1 - rlx);
		i__3 = *m;
		for (idum = 1; idum <= i__3; ++idum) {
          ierr = mc64ED(&qlen, m, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		    l[i__] = 0;
		    --low;
		    q[low] = i__;
		    if (qlen == 0) {
			goto L153;
		    }
		    i__ = q[1];
		    if (d__[i__] < tbv) {
			goto L153;
		    }
/* L152: */
		}
/* End of dummy loop; this point is never reached */
	    }
/* Move row Q0 */
L153:
	    --up;
	    q0 = q[up];
	    dq0 = d__[q0];
	    l[q0] = up;
/* Scan column that matches with row Q0 */
	    j = iperm[q0];
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		i__ = irn[k];
/* Update D(I); only if row I is not marked */
		if (l[i__] >= up) {
		    goto L155;
		}
/* Computing MIN */
		d__2 = dq0, d__3 = (d__1 = a[k], fabs(d__1));
		dnew = PetscMin(d__2,d__3);
		if (csp >= dnew) {
		    goto L155;
		}
		if (iperm[i__] == 0) {
/* Row I is unassigned; update shortest path info */
		    csp = dnew;
		    isp = i__;
		    jsp = j;
		    if (csp >= tbv) {
			goto L160;
		    }
		} else {
		    di = d__[i__];
		    if (di >= tbv || di >= dnew) {
			goto L155;
		    }
		    d__[i__] = dnew;
		    if (dnew >= tbv) {
/* Delete row I from Q (if necessary); add row I to Q2 */
			if (di != -1.) {
			    lpos = l[i__];
			    ierr = mc64FD(&lpos, &qlen, m, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
			}
			l[i__] = 0;
			--low;
			q[low] = i__;
		    } else {
/* Add row I to Q (if necessary); push row I up Q */
			if (di == -1.) {
			    ++qlen;
			    l[i__] = qlen;
			}
			ierr = mc64DD(&i__, m, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		    }
/* Update tree */
		    jj = iperm[i__];
		    pr[jj] = j;
		}
L155:
		;
	    }
/* L150: */
	}
/* If CSP = MINONE, no augmenting path is found */
L160:
	if (csp == -1.) {
	    goto L190;
	}
/* Update bottleneck value */
	bv = PetscMin(bv,csp);
	tbv = bv * (1 - rlx);
/* Find augmenting path by tracing backward in PR; update IPERM,JPERM */
	++(*num);
	i__ = isp;
	j = jsp;
	i__2 = *num + 1;
	for (jdum = 1; jdum <= i__2; ++jdum) {
	    i0 = jperm[j];
	    jperm[j] = i__;
	    iperm[i__] = j;
	    j = pr[j];
	    if (j == -1) {
		goto L190;
	    }
	    i__ = i0;
/* L170: */
	}
/* End of dummy loop; this point is never reached */
L190:
	i__2 = *m;
	for (kk = up; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
	    l[i__] = 0;
/* L191: */
	}
	i__2 = up - 1;
	for (kk = low; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
/* L192: */
	}
	i__2 = qlen;
	for (kk = 1; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
	    l[i__] = 0;
/* L193: */
	}
L100:
	;
    }
/* End of main loop */
/* BV is now bottleneck value of final matching */
/* IPERM is complete if M = N and NUM = N */
L1000:
    if (*m == *n && *num == *n) {
	goto L2000;
    }
/* Complete IPERM; L, JPERM are work arrays */
    ierr = HSLmc64XD(m, n, &iperm[1], &l[1], &jperm[1]);CHKERRQ(ierr);
L2000:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64SD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* M, N, NE, IP, IRN, are described in MC64AD. */
/* A is a DOUBLE PRECISION array of length NE. */
/*   A(K), K=1..NE, must be set to the value of the entry that */
/*   corresponds to IRN(k). The entries in each column must be */
/*   non-negative and ordered by decreasing value. */
/* IPERM is an PETSCINT array of length M. On exit, it contains the */
/*   bottleneck matching: IPERM(I) - 0 or row I is matched to column */
/*   IPERM(I). */
/* NUMX is an PETSCINT variable. On exit, it contains the cardinality */
/*   of the matching stored in IPERM. */
/* FC is an PetscInt array of length N that contains the list of */
/*   unmatched columns. */
/* LEN(J), LENL(J), LENH(J) are PetscInt arrays of length N that point */
/*   to entries in matrix column J. */
/*   In the matrix defined by the column parts IP(J)+LENL(J) we know */
/*   a matching does not exist; in the matrix defined by the column */
/*   parts IP(J)+LENH(J) we know one exists. */
/*   LEN(J) lies between LENL(J) and LENH(J) and determines the matrix */
/*   that is tested for a maximum matching. */
/* W is an PetscInt array of length N and contains the indices of the */
/*   columns for which LENL /= LENH. */
/* WLEN is number of indices stored in array W. */
/* IW is PetscInt work array of length M. */
/* IW4 is PetscInt work array of length 3N+M used by MC64U. */

/* RLX is a DOUBLE PRECISION variable. It is a relaxation */
/*   parameter for finding the optimal matching. */

/* RINF is the largest positive real number */
/* External subroutines and/or functions */
/* Intrinsic functions */
/* BMIN and BMAX are such that a maximum matching exists for the input */
/*   matrix in which all entries smaller than BMIN are dropped. */
/*   For BMAX, a maximum matching does not exist. */
/* BVAL is a value between BMIN and BMAX. */
/* CNT is the number of calls made to MC64U so far. */
/* NUM is the cardinality of last matching found. */
/* Compute a first maximum matching from scratch on whole matrix. */
PetscErrorCode HSLmc64SD(PetscInt *m, PetscInt *n, PetscInt *ne,
                         const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx,
                         PetscInt *w, PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc,
                         PetscInt *iw, PetscInt *iw4, PetscScalar *rlx, PetscScalar *rinf)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, l, ii;
    static PetscInt mod, cnt, num;
    static PetscScalar bval, bmin, bmax;
    static PetscInt nval, wlen, idum1, idum2, idum3;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iw;
    --iperm;
    --iw4;
    --fc;
    --lenh;
    --lenl;
    --len;
    --w;
    --ip;
    --a;
    --irn;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	fc[j] = j;
	len[j] = ip[j + 1] - ip[j];
/* L20: */
    }
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iw[i__] = 0;
/* L21: */
    }
/* The first call to MC64U */
    cnt = 1;
    mod = 1;
    *numx = 0;
    ierr = HSLmc64UD(&cnt, &mod, m, n, &irn[1], ne, &ip[1], &len[1], &fc[1], &iw[1], numx, n, &iw4[1], &iw4[*n + 1], &iw4[(*n << 1) + 1], &iw4[(*n << 1) + *m + 1]);CHKERRQ(ierr);
/* IW contains a maximum matching of length NUMX. */
    num = *numx;
    if (num != *n) {
/* Matrix is structurally singular */
	bmax = *rinf;
    } else {
/* Matrix is structurally nonsingular, NUM=NUMX=N; */
/* Set BMAX just above the smallest of all the maximum absolute */
/* values of the columns */
	bmax = *rinf;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    bval = 0.f;
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		if (a[k] > bval) {
		    bval = a[k];
		}
/* L25: */
	    }
	    if (bval < bmax) {
		bmax = bval;
	    }
/* L30: */
	}
/* ... should print warning if BMAX == RINF */
	bmax *= 1.001f;
    }
/* Initialize BVAL,BMIN */
    bval = 0.f;
    bmin = 0.f;
/* Initialize LENL,LEN,LENH,W,WLEN according to BMAX. */
/* Set LEN(J), LENH(J) just after last entry in column J. */
/* Set LENL(J) just after last entry in column J with value >= BMAX. */
    wlen = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	l = ip[j + 1] - ip[j];
	lenh[j] = l;
	len[j] = l;
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    if (a[k] < bmax) {
		goto L46;
	    }
/* L45: */
	}
/* Column J is empty or all entries are >= BMAX */
	k = ip[j + 1];
L46:
	lenl[j] = k - ip[j];
/* Add J to W if LENL(J) /= LENH(J) */
	if (lenl[j] == l) {
	    goto L48;
	}
	++wlen;
	w[wlen] = j;
L48:
	;
    }
/* Main loop */
    i__1 = *ne;
    for (idum1 = 1; idum1 <= i__1; ++idum1) {
	if (num == *numx) {
/* We have a maximum matching in IW; store IW in IPERM */
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		iperm[i__] = iw[i__];
/* L50: */
	    }
/* Keep going round this loop until matching IW is no longer maximum. */
	    i__2 = *ne;
	    for (idum2 = 1; idum2 <= i__2; ++idum2) {
		bmin = bval;
		if (bmax - bmin <= *rlx) {
		    goto L1000;
		}
/* Find splitting value BVAL */
		ierr = mc64QD(&ip[1], &lenl[1], &len[1], &w[1], &wlen, &a[1], &nval, &bval);CHKERRQ(ierr);
		if (nval <= 1) {
		    goto L1000;
		}
/* Set LEN such that all matrix entries with value < BVAL are */
/* discarded. Store old LEN in LENH. Do this for all columns W(K). */
/* Each step, either K is incremented or WLEN is decremented. */
		k = 1;
		i__3 = *n;
		for (idum3 = 1; idum3 <= i__3; ++idum3) {
		    if (k > wlen) {
			goto L71;
		    }
		    j = w[k];
		    i__4 = ip[j] + lenl[j];
		    for (ii = ip[j] + len[j] - 1; ii >= i__4; --ii) {
			if (a[ii] >= bval) {
			    goto L60;
			}
			i__ = irn[ii];
			if (iw[i__] != j) {
			    goto L55;
			}
/* Remove entry from matching */
			iw[i__] = 0;
			--num;
			fc[*n - num] = j;
L55:
			;
		    }
L60:
		    lenh[j] = len[j];
/* IP(J)+LEN(J)-1 is last entry in column >= BVAL */
		    len[j] = ii - ip[j] + 1;
/* If LENH(J) = LENL(J), remove J from W */
		    if (lenl[j] == lenh[j]) {
			w[k] = w[wlen];
			--wlen;
		    } else {
			++k;
		    }
/* L70: */
		}
L71:
		if (num < *numx) {
		    goto L81;
		}
/* L80: */
	    }
/* End of dummy loop; this point is never reached */
/* Set mode for next call to MC64U */
L81:
	    mod = 1;
	} else {
/* We do not have a maximum matching in IW. */
	    bmax = bval;
/* BMIN is the bottleneck value of a maximum matching; */
/* for BMAX the matching is not maximum, so BMAX>BMIN */
/* and following condition is always false if RLX = 0.0 */
	    if (bmax - bmin <= *rlx) {
		goto L1000;
	    }
/* Find splitting value BVAL */
	    ierr = mc64QD(&ip[1], &len[1], &lenh[1], &w[1], &wlen, &a[1], &nval, &bval);CHKERRQ(ierr);
	    if (nval == 0 || bval == bmin) {
		goto L1000;
	    }
/* Set LEN such that all matrix entries with value >= BVAL are */
/* inside matrix. Store old LEN in LENL. Do this for all columns W(K). */
/* Each step, either K is incremented or WLEN is decremented. */
	    k = 1;
	    i__2 = *n;
	    for (idum3 = 1; idum3 <= i__2; ++idum3) {
		if (k > wlen) {
		    goto L88;
		}
		j = w[k];
		i__3 = ip[j] + lenh[j] - 1;
		for (ii = ip[j] + len[j]; ii <= i__3; ++ii) {
		    if (a[ii] < bval) {
			goto L86;
		    }
/* L85: */
		}
L86:
		lenl[j] = len[j];
		len[j] = ii - ip[j];
		if (lenl[j] == lenh[j]) {
		    w[k] = w[wlen];
		    --wlen;
		} else {
		    ++k;
		}
/* L87: */
	    }
/* End of dummy loop; this point is never reached */
/* Set mode for next call to MC64U */
L88:
	    mod = 0;
	}
	++cnt;
	ierr = HSLmc64UD(&cnt, &mod, m, n, &irn[1], ne, &ip[1], &len[1], &fc[1], 
                     &iw[1], &num, numx, &iw4[1], &iw4[*n + 1], &iw4[(*n << 1) + 1],
                     &iw4[(*n << 1) + *m + 1]);CHKERRQ(ierr);
/* IW contains maximum matching of length NUM */
/* L90: */
    }
/* End of dummy loop; this point is never reached */
/* BMIN is now bottleneck value of final matching */
/* IPERM is complete if M = N and NUMX = N */
L1000:
    if (*m == *n && *numx == *n) {
	goto L2000;
    }
/* Complete IPERM; IW, W are work arrays */
    ierr = HSLmc64XD(m, n, &iperm[1], &iw[1], &w[1]);CHKERRQ(ierr);
L2000:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64UD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* PR(J) is the previous column to J in the depth first search. */
/*   Array PR is used as workspace in the sorting algorithm. */
/* Elements (I,IPERM(I)) I=1,..,M are entries at the end of the */
/*   algorithm unless N assignments have not been made in which case */
/*   N-NUM pairs (I,IPERM(I)) will not be entries in the matrix. */
/* CV(I) is the most recent loop number (ID+JORD) at which row I */
/*   was visited. */
/* ARP(J) is the number of entries in column J which have been scanned */
/*   when looking for a cheap assignment. */
/* OUT(J) is one less than the number of entries in column J which have */
/*   not been scanned during one pass through the main loop. */
/* NUMX is maximum possible size of matching. */
PetscErrorCode HSLmc64UD(PetscInt *id, PetscInt *mod, PetscInt *m, 
                         PetscInt *n, const PetscInt *irn, PetscInt *lirn, const PetscInt *ip, PetscInt *lenc, 
                         PetscInt *fc, PetscInt *iperm, PetscInt *num, PetscInt *numx, PetscInt *pr,
                         PetscInt *arp, PetscInt *cv, PetscInt *out)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, j1, ii, kk, id0, id1, in1, in2, nfc, num0, num1,
	     num2, jord, last;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --cv;
    --iperm;
    --out;
    --arp;
    --pr;
    --fc;
    --lenc;
    --ip;
    --irn;

    /* Function Body */
    if (*id == 1) {
/* The first call to MC64U. */
/* Initialize CV and ARP; parameters MOD, NUMX are not accessed */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cv[i__] = 0;
/* L5: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    arp[j] = 0;
/* L6: */
	}
	num1 = *n;
	num2 = *n;
    } else {
/* Not the first call to MC64U. */
/* Re-initialize ARP if entries were deleted since last call to MC64U */
	if (*mod == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		arp[j] = 0;
/* L8: */
	    }
	}
	num1 = *numx;
	num2 = *n - *numx;
    }
    num0 = *num;
/* NUM0 is size of input matching */
/* NUM1 is maximum possible size of matching */
/* NUM2 is maximum allowed number of unassigned rows/columns */
/* NUM is size of current matching */
/* Quick return if possible */
/*      IF (NUM.EQ.N) GO TO 199 */
/* NFC is number of rows/columns that could not be assigned */
    nfc = 0;
/* PetscInts ID0+1 to ID0+N are unique numbers for call ID to MC64U, */
/* so 1st call uses 1..N, 2nd call uses N+1..2N, etc */
    id0 = (*id - 1) * *n;
/* Main loop. Each pass round this loop either results in a new */
/* assignment or gives a column with no assignment */
    i__1 = *n;
    for (jord = num0 + 1; jord <= i__1; ++jord) {
/* Each pass uses unique number ID1 */
	id1 = id0 + jord;
/* J is unmatched column */
	j = fc[jord - num0];
	pr[j] = -1;
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
/* Look for a cheap assignment */
	    if (arp[j] >= lenc[j]) {
		goto L30;
	    }
	    in1 = ip[j] + arp[j];
	    in2 = ip[j] + lenc[j] - 1;
	    i__3 = in2;
	    for (ii = in1; ii <= i__3; ++ii) {
		i__ = irn[ii];
		if (iperm[i__] == 0) {
		    goto L80;
		}
/* L20: */
	    }
/* No cheap assignment in row */
	    arp[j] = lenc[j];
/* Begin looking for assignment chain starting with row J */
L30:
	    out[j] = lenc[j] - 1;
/* Inner loop.  Extends chain by one or backtracks */
	    i__3 = jord;
	    for (kk = 1; kk <= i__3; ++kk) {
		in1 = out[j];
		if (in1 < 0) {
		    goto L50;
		}
		in2 = ip[j] + lenc[j] - 1;
		in1 = in2 - in1;
/* Forward scan */
		i__4 = in2;
		for (ii = in1; ii <= i__4; ++ii) {
		    i__ = irn[ii];
		    if (cv[i__] == id1) {
			goto L40;
		    }
/* Column J has not yet been accessed during this pass */
		    j1 = j;
		    j = iperm[i__];
		    cv[i__] = id1;
		    pr[j] = j1;
		    out[j1] = in2 - ii - 1;
		    goto L70;
L40:
		    ;
		}
/* Backtracking step. */
L50:
		j1 = pr[j];
		if (j1 == -1) {
/* No augmenting path exists for column J. */
		    ++nfc;
		    fc[nfc] = j;
		    if (nfc > num2) {
/* A matching of maximum size NUM1 is not possible */
			last = jord;
			goto L101;
		    }
		    goto L100;
		}
		j = j1;
/* L60: */
	    }
/* End of dummy loop; this point is never reached */
L70:
	    ;
	}
/* End of dummy loop; this point is never reached */
/* New assignment is made. */
L80:
	iperm[i__] = j;
	arp[j] = ii - ip[j] + 1;
	++(*num);
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
	    j = pr[j];
	    if (j == -1) {
		goto L95;
	    }
	    ii = ip[j] + lenc[j] - out[j] - 2;
	    i__ = irn[ii];
	    iperm[i__] = j;
/* L90: */
	}
/* End of dummy loop; this point is never reached */
L95:
	if (*num == num1) {
/* A matching of maximum size NUM1 is found */
	    last = jord;
	    goto L101;
	}

L100:
	;
    }
/* All unassigned columns have been considered */
    last = *n;
/* Now, a transversal is computed or is not possible. */
/* Complete FC before returning. */
L101:
    i__1 = *n;
    for (jord = last + 1; jord <= i__1; ++jord) {
	++nfc;
	fc[nfc] = fc[jord - num0];
/* L110: */
    }
/*  199 RETURN */
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64WD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* M, N, NE, IP, IRN are described in MC64AD. */
/* A is a DOUBLE PRECISION array of length NE. */
/*   A(K), K=1..NE, must be set to the value of the entry that */
/*   corresponds to IRN(K). It is not altered. */
/*   All values A(K) must be non-negative. */
/* IPERM is an PETSCINT array of length M. On exit, it contains the */
/*   weighted matching: IPERM(I) = 0 or row I is matched to column */
/*   IPERM(I). */
/* NUM is an PETSCINT variable. On exit, it contains the cardinality of */
/*   the matching stored in IPERM. */
/* D is a DOUBLE PRECISION array of length M. */
/*   On exit, V = D(1:N) contains the dual column variable. */
/*   If U(1:M) denotes the dual row variable and if the matrix */
/*   is structurally nonsingular (NUM = N), the following holds: */
/*      U(I)+V(J) <= A(I,J)  if IPERM(I) /= J */
/*      U(I)+V(J)  = A(I,J)  if IPERM(I)  = J */
/*      U(I) = 0  if IPERM(I) = 0 */
/*      V(J) = 0  if there is no I for which IPERM(I) = J */
/*   On entry, U(1) contains the relaxation parameter RLX. */
/* RINF is the largest positive real number */
/* Local variables */
/* Local parameters */
/* External subroutines and/or functions */
/* Set RINF to largest positive real number */
PetscErrorCode HSLmc64WD(PetscInt *m, PetscInt *n, PetscInt *ne,
                         const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num,
                         PetscInt *jperm, PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l,
                         PetscScalar *u, PetscScalar *d__, PetscScalar *rinf)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3;

    /* Local variables */
    static PetscInt i__, j, k, i0, k0, k1, k2, q0;
    static PetscScalar di;
    static PetscInt ii, jj, kk;
    static PetscScalar vj;
    static PetscInt up;
    static PetscScalar dq0;
    static PetscInt kk1, kk2;
    static PetscScalar csp;
    static PetscInt isp, jsp, low;
    static PetscScalar dmin__, dnew;
    static PetscInt jord, qlen, jdum, lpos;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --d__;
    --u;
    --l;
    --q;
    --iperm;
    --pr;
    --out;
    --jperm;
    --ip;
    --a;
    --irn;

    /* Function Body */
    *rinf = fd15AD((char *)"H", (ftnlen)1);
/* Initialization */
    *num = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	d__[k] = 0.;
	jperm[k] = 0;
	pr[k] = ip[k];
/* L10: */
    }
    i__1 = *m;
    for (k = 1; k <= i__1; ++k) {
	u[k] = *rinf;
	iperm[k] = 0;
	l[k] = 0;
/* L15: */
    }
/* Initialize U(I) */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    if (a[k] > u[i__]) {
		goto L20;
	    }
	    u[i__] = a[k];
	    iperm[i__] = j;
	    l[i__] = k;
L20:
	    ;
	}
/* L30: */
    }
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = iperm[i__];
	if (j == 0) {
	    goto L40;
	}
/* Row I is not empty */
	iperm[i__] = 0;
	if (jperm[j] != 0) {
	    goto L40;
	}
/* Don't choose cheap assignment from dense columns */
	if (ip[j + 1] - ip[j] > *n / 10 && *n > 50) {
	    goto L40;
	}
/* Assignment of column J to row I */
	++(*num);
	iperm[i__] = j;
	jperm[j] = l[i__];
L40:
	;
    }
/*      write(14,*) 'Number of cheap assignments ',NUM */
    if (*num == *n) {
	goto L1000;
    }
/* Scan unassigned columns; improve assignment */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* JPERM(J) ne 0 iff column J is already assigned */
	if (jperm[j] != 0) {
	    goto L95;
	}
	k1 = ip[j];
	k2 = ip[j + 1] - 1;
/* Continue only if column J is not empty */
	if (k1 > k2) {
	    goto L95;
	}
/*       VJ = RINF */
/* Changes made to allow for NaNs */
	i0 = irn[k1];
	vj = a[k1] - u[i0];
	k0 = k1;
	i__2 = k2;
	for (k = k1 + 1; k <= i__2; ++k) {
	    i__ = irn[k];
	    di = a[k] - u[i__];
	    if (di > vj) {
		goto L50;
	    }
	    if (di < vj || di == *rinf) {
		goto L55;
	    }
	    if (iperm[i__] != 0 || iperm[i0] == 0) {
		goto L50;
	    }
L55:
	    vj = di;
	    i0 = i__;
	    k0 = k;
L50:
	    ;
	}
	d__[j] = vj;
	k = k0;
	i__ = i0;
	if (iperm[i__] == 0) {
	    goto L90;
	}
	i__2 = k2;
	for (k = k0; k <= i__2; ++k) {
	    i__ = irn[k];
	    if (a[k] - u[i__] > vj) {
		goto L60;
	    }
	    jj = iperm[i__];
/* Scan remaining part of assigned column JJ */
	    kk1 = pr[jj];
	    kk2 = ip[jj + 1] - 1;
	    if (kk1 > kk2) {
		goto L60;
	    }
	    i__3 = kk2;
	    for (kk = kk1; kk <= i__3; ++kk) {
		ii = irn[kk];
		if (iperm[ii] > 0) {
		    goto L70;
		}
		if (a[kk] - u[ii] <= d__[jj]) {
		    goto L80;
		}
L70:
		;
	    }
	    pr[jj] = kk2 + 1;
L60:
	    ;
	}
	goto L95;
L80:
	jperm[jj] = kk;
	iperm[ii] = jj;
	pr[jj] = kk + 1;
L90:
	++(*num);
	jperm[j] = k;
	iperm[i__] = j;
	pr[j] = k + 1;
L95:
	;
    }
/*     write(14,*) 'Number of improved  assignments ',NUM */
    if (*num == *n) {
	goto L1000;
    }
/* Prepare for main loop */
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = *rinf;
	l[i__] = 0;
/* L99: */
    }
/* Main loop ... each pass round this loop is similar to Dijkstra's */
/* algorithm for solving the single source shortest path problem */
    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {
	if (jperm[jord] != 0) {
	    goto L100;
	}
/* JORD is next unmatched column */
/* DMIN is the length of shortest path in the tree */
	dmin__ = *rinf;
	qlen = 0;
	low = *n + 1;
	up = *n + 1;
/* CSP is the cost of the shortest augmenting path to unassigned row */
/* IRN(ISP). The corresponding column index is JSP. */
	csp = *rinf;
/* Build shortest path tree starting from unassigned column (root) JORD */
	j = jord;
	pr[j] = -1;
/* Scan column J */
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
/*         IF (N.EQ.3) THEN */
/*           write(14,*) 'Scanning column ',J */
/*           write(14,*) 'IP  ',IP(1:4) */
/*           write(14,*) 'IRN ',IRN(1:6) */
/*           write(14,*) 'A   ',A(1:6) */
/*           write(14,*) 'U ',U(1:3) */
/*           write(14,*) 'IPERM ',IPERM(1:3) */
/*         ENDIF */
	    i__ = irn[k];
	    dnew = a[k] - u[i__];
	    if (dnew >= csp) {
		goto L115;
	    }
	    if (iperm[i__] == 0) {
		csp = dnew;
		isp = k;
		jsp = j;
	    } else {
		if (dnew < dmin__) {
		    dmin__ = dnew;
		}
		d__[i__] = dnew;
		++qlen;
		q[qlen] = k;
	    }
L115:
	    ;
	}
/* Initialize heap Q and Q2 with rows held in Q(1:QLEN) */
	q0 = qlen;
	qlen = 0;
	i__2 = q0;
	for (kk = 1; kk <= i__2; ++kk) {
	    k = q[kk];
	    i__ = irn[k];
	    if (csp <= d__[i__]) {
		d__[i__] = *rinf;
		goto L120;
	    }
	    if (d__[i__] <= dmin__) {
		--low;
		q[low] = i__;
		l[i__] = low;
	    } else {
		++qlen;
		l[i__] = qlen;
		ierr = mc64DD(&i__, m, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
	    }
/* Update tree */
	    jj = iperm[i__];
	    out[jj] = k;
	    pr[jj] = j;
L120:
	    ;
	}
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
/* If Q2 is empty, extract rows from Q */
	    if (low == up) {
		if (qlen == 0) {
		    goto L160;
		}
		i__ = q[1];
		if (d__[i__] >= csp) {
		    goto L160;
		}
		dmin__ = d__[i__];
L152:
		ierr = mc64ED(&qlen, m, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
		--low;
		q[low] = i__;
		l[i__] = low;
		if (qlen == 0) {
		    goto L153;
		}
		i__ = q[1];
		if (d__[i__] > dmin__) {
		    goto L153;
		}
		goto L152;
	    }
/* Q0 is row whose distance D(Q0) to the root is smallest */
L153:
	    q0 = q[up - 1];
	    dq0 = d__[q0];
/* Exit loop if path to Q0 is longer than the shortest augmenting path */
	    if (dq0 >= csp) {
		goto L160;
	    }
	    --up;
/* Scan column that matches with row Q0 */
	    j = iperm[q0];
	    vj = dq0 - a[jperm[j]] + u[q0];
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		i__ = irn[k];
		if (l[i__] >= up) {
		    goto L155;
		}
/* DNEW is new cost */
		dnew = vj + a[k] - u[i__];
/* Do not update D(I) if DNEW ge cost of shortest path */
		if (dnew >= csp) {
		    goto L155;
		}
		if (iperm[i__] == 0) {
/* Row I is unmatched; update shortest path info */
		    csp = dnew;
		    isp = k;
		    jsp = j;
		} else {
/* Row I is matched; do not update D(I) if DNEW is larger */
		    di = d__[i__];
		    if (di <= dnew) {
			goto L155;
		    }
		    if (l[i__] >= low) {
			goto L155;
		    }
		    d__[i__] = dnew;
		    if (dnew <= dmin__) {
			lpos = l[i__];
			if (lpos != 0) {
              ierr = mc64FD(&lpos, &qlen, m, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
			}
			--low;
			q[low] = i__;
			l[i__] = low;
		    } else {
			if (l[i__] == 0) {
			    ++qlen;
			    l[i__] = qlen;
			}
			ierr = mc64DD(&i__, m, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
		    }
/* Update tree */
		    jj = iperm[i__];
		    out[jj] = k;
		    pr[jj] = j;
		}
L155:
		;
	    }
/* L150: */
	}
/* If CSP = RINF, no augmenting path is found */
L160:
	if (csp == *rinf) {
	    goto L190;
	}
/* Find augmenting path by tracing backward in PR; update IPERM,JPERM */
	++(*num);
/*       write(14,*) 'NUM = ',NUM */
/*       write(14,*) 'Augmenting path found from unmatched col ',JORD */
	i__ = irn[isp];
	iperm[i__] = jsp;
	jperm[jsp] = isp;
	j = jsp;
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
	    jj = pr[j];
	    if (jj == -1) {
		goto L180;
	    }
	    k = out[j];
	    i__ = irn[k];
	    iperm[i__] = jj;
	    jperm[jj] = k;
	    j = jj;
/* L170: */
	}
/* End of dummy loop; this point is never reached */
/* Update U for rows in Q(UP:N) */
L180:
	i__2 = *n;
	for (kk = up; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    u[i__] = u[i__] + d__[i__] - csp;
/* L185: */
	}
L190:
	i__2 = *n;
	for (kk = low; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = *rinf;
	    l[i__] = 0;
/* L191: */
	}
	i__2 = qlen;
	for (kk = 1; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = *rinf;
	    l[i__] = 0;
/* L193: */
	}
L100:
	;
    }
/* End of main loop */
/* Set dual column variable in D(1:N) */
L1000:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	k = jperm[j];
	if (k != 0) {
	    d__[j] = a[k] - u[irn[k]];
	} else {
	    d__[j] = 0.;
	}
/* L200: */
    }
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] == 0) {
	    u[i__] = 0.;
	}
/* L1201: */
    }
    if (*num == *n && *m == *n) {
	goto L1100;
    }
/* Complete IPERM; L, JPERM are work arrays */
    ierr = HSLmc64XD(m, n, &iperm[1], &l[1], &jperm[1]);CHKERRQ(ierr);
/* The matrix is structurally singular, complete IPERM. */
/* JPERM, OUT are work arrays */
/*     DO 300 J = 1,N */
/*       JPERM(J) = 0 */
/* 300 CONTINUE */
/*     K = 0 */
/*     DO 310 I = 1,N */
/*       IF (IPERM(I).EQ.0) THEN */
/*         K = K + 1 */
/*         OUT(K) = I */
/*       ELSE */
/*         J = IPERM(I) */
/*         JPERM(J) = I */
/*       ENDIF */
/* 310 CONTINUE */
/*     K = 0 */
/*     DO 320 J = 1,N */
/*       IF (JPERM(J).NE.0) GO TO 320 */
/*       K = K + 1 */
/*       JDUM = OUT(K) */
/*       IPERM(JDUM) = - J */
/* 320 CONTINUE */
L1100:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64ZD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* PR(I) is the previous row to I in the depth first search. */
/*   It is used as a work array in the sorting algorithm. */
/* Elements (IPERM(I),I) I=1,...M  are non-zero at the end of the */
/*   algorithm unless N assignments have not been made.  In which case */
/*   (IPERM(I),I) will be zero for N-NUM entries. */
/* CV(I) is the most recent row extension at which column I was visited. */
/* ARP(I) is one less than the number of non-zeros in row I */
/*   which have not been scanned when looking for a cheap assignment. */
/* OUT(I) is one less than the number of non-zeros in row I */
/*   which have not been scanned during one pass through the main loop. */
PetscErrorCode HSLmc64ZD(PetscInt *m, PetscInt *n, const PetscInt *irn,
                         PetscInt *lirn, const PetscInt *ip, PetscInt *lenc, PetscInt *iperm, PetscInt *num,
                         PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, j1, ii, kk;
    static PetscInt in1, in2, jord;

    PetscFunctionBegin;
/* External subroutines and/or functions */
    /* Parameter adjustments */
    --cv;
    --iperm;
    --out;
    --arp;
    --pr;
    --lenc;
    --ip;
    --irn;

    /* Function Body */
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	cv[i__] = 0;
	iperm[i__] = 0;
/* L10: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	arp[j] = lenc[j] - 1;
/* L12: */
    }
    *num = 0;

/* Main loop. Each pass round this loop either results in a new */
/* assignment or gives a row with no assignment. */

    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {

	j = jord;
	pr[j] = -1;
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
/* Look for a cheap assignment */
	    in1 = arp[j];
	    if (in1 < 0) {
		goto L30;
	    }
	    in2 = ip[j] + lenc[j] - 1;
	    in1 = in2 - in1;
	    i__3 = in2;
	    for (ii = in1; ii <= i__3; ++ii) {
		i__ = irn[ii];
		if (iperm[i__] == 0) {
		    goto L80;
		}
/* L20: */
	    }
/* No cheap assignment in row. */
	    arp[j] = -1;
/* Begin looking for assignment chain starting with row J. */
L30:
	    out[j] = lenc[j] - 1;
/* Inner loop.  Extends chain by one or backtracks. */
	    i__3 = jord;
	    for (kk = 1; kk <= i__3; ++kk) {
		in1 = out[j];
		if (in1 < 0) {
		    goto L50;
		}
		in2 = ip[j] + lenc[j] - 1;
		in1 = in2 - in1;
/* Forward scan. */
		i__4 = in2;
		for (ii = in1; ii <= i__4; ++ii) {
		    i__ = irn[ii];
		    if (cv[i__] == jord) {
			goto L40;
		    }
/* Column I has not yet been accessed during this pass. */
		    j1 = j;
		    j = iperm[i__];
		    cv[i__] = jord;
		    pr[j] = j1;
		    out[j1] = in2 - ii - 1;
		    goto L70;
L40:
		    ;
		}
/* Backtracking step. */
L50:
		j = pr[j];
		if (j == -1) {
		    goto L1000;
		}
/* L60: */
	    }
L70:
	    ;
	}

/* New assignment is made. */
L80:
	iperm[i__] = j;
	arp[j] = in2 - ii - 1;
	++(*num);
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
	    j = pr[j];
	    if (j == -1) {
		goto L1000;
	    }
	    ii = ip[j] + lenc[j] - out[j] - 2;
	    i__ = irn[ii];
	    iperm[i__] = j;
/* L90: */
	}

L1000:
	;
    }
/* IPERM is complete if M = N and NUM = N */
    if (*m == *n && *num == *n) {
	goto L2000;
    }
/* Complete IPERM; CV, ARP are work arrays */
    ierr = HSLmc64XD(m, n, &iperm[1], &cv[1], &arp[1]);CHKERRQ(ierr);
L2000:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
/* CCCC LAST UPDATE Tue Nov 26 03:20:26 MET 2002 */
#undef __FUNCT__
#define __FUNCT__ "HSLmc64XD"
/* *** Copyright (c) 2002  I.S. Duff and J. Koster                   *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */

/* Complete the (incomplete) row permutation in IPERM. */

/* If M=N, the matrix is structurally singular, complete IPERM */
/* If M>N, the matrix is rectangular, complete IPERM */
/* RW, CW are work arrays; */
/* Store indices of unmatched rows in RW */
/* Mark matched columns in CW */
PetscErrorCode HSLmc64XD(PetscInt *m, PetscInt *n, PetscInt *iperm, PetscInt *rw, PetscInt *cw)
{
    /* System generated locals */
    PetscInt i__1;

    /* Local variables */
    static PetscInt i__, j, k;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --rw;
    --iperm;
    --cw;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	cw[j] = 0;
/* L10: */
    }
    k = 0;
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] == 0) {
	    ++k;
	    rw[k] = i__;
	} else {
	    j = iperm[i__];
	    cw[j] = i__;
	}
/* L20: */
    }
    k = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (cw[j] != 0) {
	    goto L30;
	}
	++k;
	i__ = rw[k];
	iperm[i__] = -j;
L30:
	;
    }
    i__1 = *m;
    for (j = *n + 1; j <= i__1; ++j) {
	++k;
	i__ = rw[k];
	iperm[i__] = -j;
/* L40: */
    }
    PetscFunctionReturn(0);
}

/* ddeps.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

/* COPYRIGHT (c) 1987 AEA Technology */
/* Original date 10 Feb 1993 */
/*       Toolpack tool decs employed. */
/* 20/2/02 Cosmetic changes applied to reduce single/double differences */
/* 12th July 2004 Version 1.0.0. Version numbering added. */
#undef __FUNCT__
#define __FUNCT__ "mc34AD"
/* THIS SUBROUTINE ACCEPTS AS INPUT THE STANDARD DATA STRUCTURE FOR */
/*     A SYMMETRIC MATRIX STORED AS A LOWER TRIANGLE AND PRODUCES */
/*     AS OUTPUT THE SYMMETRIC MATRIX HELD IN THE SAME DATA */
/*     STRUCTURE AS A GENERAL MATRIX. */
/* N IS AN PETSCINT VARIABLE THAT MUST BE SET BY THE USER TO THE */
/*     ORDER OF THE MATRIX. NOT ALTERED BY THE ROUTINE */
/*     RESTRICTION (IBM VERSION ONLY): N LE 32767. */
/* IRN IS AN PETSCINT (PETSCINT*2 IN IBM VERSION) ARRAY THAT */
/*     MUST BE SET BY THE USER TO HOLD THE ROW INDICES OF THE LOWER */
/*     TRIANGULAR PART OF THE SYMMETRIC MATRIX.  THE ENTRIES OF A */
/*     SINGLE COLUMN ARE CONTIGUOUS. THE ENTRIES OF COLUMN J */
/*     PRECEDE THOSE OF COLUMN J+1 (J_=_1, ..., N-1), AND THERE IS */
/*     NO WASTED SPACE BETWEEN COLUMNS. ROW INDICES WITHIN A COLUMN */
/*     MAY BE IN ANY ORDER.  ON EXIT IT WILL HAVE THE SAME MEANING */
/*     BUT WILL BE CHANGED TO HOLD THE ROW INDICES OF ENTRIES IN */
/*     THE EXPANDED STRUCTURE.  DIAGONAL ENTRIES NEED NOT BE */
/*     PRESENT. THE NEW ROW INDICES ADDED IN THE UPPER TRIANGULAR */
/*     PART WILL BE IN ORDER FOR EACH COLUMN AND WILL PRECEDE THE */
/*     ROW INDICES FOR THE LOWER TRIANGULAR PART WHICH WILL REMAIN */
/*     IN THE INPUT ORDER. */
/* JCOLST IS AN PETSCINT ARRAY OF LENGTH N+1 THAT MUST BE SET BY */
/*     THE USER SO THAT JCOLST(J) IS THE POSITION IN ARRAYS IRN AND */
/*     A OF THE FIRST ENTRY IN COLUMN J (J_=_1, ..., N). */
/*     JCOLST(N+1) MUST BE SET TO ONE MORE THAN THE TOTAL NUMBER OF */
/*     ENTRIES.  ON EXIT, JCOLST(J) WILL HAVE THE SAME MEANING BUT */
/*     WILL BE CHANGED TO POINT TO THE POSITION OF THE FIRST ENTRY */
/*     OF COLUMN J IN THE EXPANDED STRUCTURE. THE NEW VALUE OF */
/*     JCOLST(N+1) WILL BE ONE GREATER THAN THE NUMBER OF ENTRIES */
/*     IN THE EXPANDED STRUCTURE. */
/* YESA IS A PETSCBOOL VARIABLE THAT MUST BE SET TO .TRUE. IF THE */
/*     USER DESIRES TO GENERATE THE EXPANDED FORM FOR THE VALUES ALSO. */
/*     IF YESA IS .FALSE., THE ARRAY A WILL NOT BE REFERENCED.  IT IS */
/*     NOT ALTERED BY THE ROUTINE. */
/* A IS A REAL (DOUBLE PRECISION IN THE D VERSION) ARRAY THAT */
/*     CAN BE SET BY THE USER SO THAT A(K) HOLDS THE VALUE OF THE */
/*     ENTRY IN POSITION K OF IRN, {K = 1, _..._ JCOLST(N+1)-1}. */
/*     ON EXIT, IF YESA IS .TRUE., THE ARRAY WILL HOLD THE VALUES */
/*     OF THE ENTRIES IN THE EXPANDED STRUCTURE CORRESPONDING TO */
/*     THE OUTPUT VALUES OF IRN.   IF YESA IS .FALSE., THE ARRAY IS */
/*     NOT ACCESSED BY THE SUBROUTINE. */
/* IW IS AN PETSCINT (PETSCINT*2 IN IBM VERSION) ARRAY OF LENGTH */
/*     N THAT WILL BE USED AS WORKSPACE. */

/* CKP1 IS A LOCAL VARIABLE USED AS A RUNNING POINTER. */
/* OLDTAU IS NUMBER OF ENTRIES IN SYMMETRIC STORAGE. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Executable Statements .. */
PetscErrorCode mc34AD(PetscInt *n, PetscInt *irn, PetscInt *jcolst, PetscBool *yesa, PetscScalar *a, PetscInt *iw)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    static PetscInt i__, j, i1, i2, ii, ckp1, lenk, ipos, ipkp1, ndiag, oldtau,
	     newtau, jstart;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iw;
    --a;
    --jcolst;
    --irn;

    /* Function Body */
    oldtau = jcolst[*n + 1] - 1;
/* INITIALIZE WORK ARRAY */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iw[i__] = 0;
/* L5: */
    }

/* IW(J) IS SET EQUAL TO THE TOTAL NUMBER OF ENTRIES IN COLUMN J */
/*     OF THE EXPANDED SYMMETRIC MATRIX. */
/* NDIAG COUNTS NUMBER OF DIAGONAL ENTRIES PRESENT */
    ndiag = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i1 = jcolst[j];
	i2 = jcolst[j + 1] - 1;
	iw[j] = iw[j] + i2 - i1 + 1;
	i__2 = i2;
	for (ii = i1; ii <= i__2; ++ii) {
	    i__ = irn[ii];
	    if (i__ != j) {
		++iw[i__];
	    } else {
		++ndiag;
	    }
/* L10: */
	}
/* L20: */
    }

/* NEWTAU IS NUMBER OF ENTRIES IN EXPANDED STORAGE. */
    newtau = (oldtau << 1) - ndiag;
/* IPKP1 POINTS TO POSITION AFTER END OF COLUMN BEING CURRENTLY */
/*     PROCESSED */
    ipkp1 = oldtau + 1;
/* CKP1 POINTS TO POSITION AFTER END OF SAME COLUMN IN EXPANDED */
/*     STRUCTURE */
    ckp1 = newtau + 1;
/* GO THROUGH THE ARRAY IN THE REVERSE ORDER PLACING LOWER TRIANGULAR */
/*     ELEMENTS IN THE APPROPRIATE SLOTS. */
    for (j = *n; j >= 1; --j) {
	i1 = jcolst[j];
	i2 = ipkp1;
/* LENK IS NUMBER OF ENTRIES IN COLUMN J OF ORIGINAL STRUCTURE */
	lenk = i2 - i1;
/* JSTART IS RUNNING POINTER TO POSITION IN NEW STRUCTURE */
	jstart = ckp1;
/* SET IKP1 FOR NEXT COLUMN */
	ipkp1 = i1;
	--i2;
/* RUN THROUGH COLUMNS IN REVERSE ORDER */
/* LOWER TRIANGULAR PART OF COLUMN MOVED TO END OF SAME COLUMN IN */
/*     EXPANDED FORM */
	i__1 = i1;
	for (ii = i2; ii >= i__1; --ii) {
	    --jstart;
	    if (*yesa) {
		a[jstart] = a[ii];
	    }
	    irn[jstart] = irn[ii];
/* L30: */
	}
/* JCOLST IS SET TO POSITION OF FIRST ENTRY IN LOWER TRIANGULAR PART OF */
/*     COLUMN J IN EXPANDED FORM */
	jcolst[j] = jstart;
/* SET CKP1 FOR NEXT COLUMN */
	ckp1 -= iw[j];
/* RESET IW(J) TO NUMBER OF ENTRIES IN LOWER TRIANGLE OF COLUMN. */
	iw[j] = lenk;
/* L40: */
    }

/* AGAIN SWEEP THROUGH THE COLUMNS IN THE REVERSE ORDER, THIS */
/*     TIME WHEN ONE IS HANDLING COLUMN J THE UPPER TRIANGULAR */
/*     ELEMENTS A(J,I) ARE PUT IN POSITION. */
    for (j = *n; j >= 1; --j) {
	i1 = jcolst[j];
	i2 = jcolst[j] + iw[j] - 1;
/* RUN DOWN COLUMN IN ORDER */
/* NOTE THAT I IS ALWAYS GREATER THAN OR EQUAL TO J */
	i__1 = i2;
	for (ii = i1; ii <= i__1; ++ii) {
	    i__ = irn[ii];
	    if (i__ == j) {
		goto L60;
	    }
	    --jcolst[i__];
	    ipos = jcolst[i__];
	    if (*yesa) {
		a[ipos] = a[ii];
	    }
	    irn[ipos] = j;
L60:
	    ;
	}
/* L80: */
    }
    jcolst[*n + 1] = newtau + 1;
    PetscFunctionReturn(0);
}

/* COPYRIGHT (c) 1999 Council for the Central Laboratory */
/*                    of the Research Councils */
/* Original date July 1999 */
/* CCCC PACKAGE MC64A/AD */
/* CCCC AUTHORS Iain Duff (i.duff@rl.ac.uk) and */
/* CCCC         Jacko Koster (jacko.koster@uninett.no) */
/* 12th July 2004 Version 1.0.0. Version numbering added. */
/* 30/07/04  Version 1.1.0. Permutation array flagged negative to */
/*           indicate dependent columns in singular case.  Calls to */
/*           MC64F/FD changed to avoid unsafe reference to array L. */
/* 21st February 2005 Version 1.2.0. FD05 dependence changed to FD15. */
/*  7th March 2005 Version 1.3.0. Scan of dense columns avoided in */
/*           the cheap assignment phase in MC64W/WD. */
/* 15th May 2007 Version 1.4.0. Minor change made in MC64W/WD to avoid */
/*           crash when the input array contains NaNs. */
/* 28th June 2007  Version 1.5.0. Permutation array flagged negative to */
/*           indicate dependent columns in singular case for all values */
/*           of the JOB parameter (previously was only for JOB 4 and 5). */
/*           Also some very minor changes concerning tests in DO loops */
/*           being moved from beginning to end of DO loop. */
#undef __FUNCT__
#define __FUNCT__ "mc64ID"
/*  Purpose */
/*  ======= */

/*  The components of the array ICNTL control the action of MC64A/AD. */
/*  Default values for these are set in this subroutine. */

/*  Parameters */
/*  ========== */


/*  Local variables */

/*    ICNTL(1) has default value 6. */
/*     It is the output stream for error messages. If it */
/*     is negative, these messages will be suppressed. */

/*    ICNTL(2) has default value 6. */
/*     It is the output stream for warning messages. */
/*     If it is negative, these messages are suppressed. */

/*    ICNTL(3) has default value -1. */
/*     It is the output stream for monitoring printing. */
/*     If it is negative, these messages are suppressed. */

/*    ICNTL(4) has default value 0. */
/*     If left at the defaut value, the incoming data is checked for */
/*     out-of-range indices and duplicates.  Setting ICNTL(4) to any */
/*     other will avoid the checks but is likely to cause problems */
/*     later if out-of-range indices or duplicates are present. */
/*     The user should only set ICNTL(4) non-zero, if the data is */
/*     known to avoid these problems. */

/*    ICNTL(5) to ICNTL(10) are not used by MC64A/AD but are set to */
/*     zero in this routine. */
/* Initialization of the ICNTL array. */
PetscErrorCode mc64ID(PetscInt *icntl)
{
    static PetscInt i__;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --icntl;

    /* Function Body */
    icntl[1] = 6;
    icntl[2] = 6;
    icntl[3] = -1;
    for (i__ = 4; i__ <= 10; ++i__) {
	icntl[i__] = 0;
/* L10: */
    }
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64AD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/*  Purpose */
/*  ======= */

/* This subroutine attempts to find a column permutation for an NxN */
/* sparse matrix A = {a_ij} that makes the permuted matrix have N */
/* entries on its diagonal. */
/* If the matrix is structurally nonsingular, the subroutine optionally */
/* returns a column permutation that maximizes the smallest element */
/* on the diagonal, maximizes the sum of the diagonal entries, or */
/* maximizes the product of the diagonal entries of the permuted matrix. */
/* For the latter option, the subroutine also finds scaling factors */
/* that may be used to scale the matrix so that the nonzero diagonal */
/* entries of the permuted matrix are one in absolute value and all the */
/* off-diagonal entries are less than or equal to one in absolute value. */
/* The natural logarithms of the scaling factors u(i), i=1..N, for the */
/* rows and v(j), j=1..N, for the columns are returned so that the */
/* scaled matrix B = {b_ij} has entries b_ij = a_ij * EXP(u_i + v_j). */

/*  Parameters */
/*  ========== */


/* JOB is an PETSCINT variable which must be set by the user to */
/* control the action. It is not altered by the subroutine. */
/* Possible values for JOB are: */
/*   1 Compute a column permutation of the matrix so that the */
/*     permuted matrix has as many entries on its diagonal as possible. */
/*     The values on the diagonal are of arbitrary size. HSL subroutine */
/*     MC21A/AD is used for this. See [1]. */
/*   2 Compute a column permutation of the matrix so that the smallest */
/*     value on the diagonal of the permuted matrix is maximized. */
/*     See [3]. */
/*   3 Compute a column permutation of the matrix so that the smallest */
/*     value on the diagonal of the permuted matrix is maximized. */
/*     The algorithm differs from the one used for JOB = 2 and may */
/*     have quite a different performance. See [2]. */
/*   4 Compute a column permutation of the matrix so that the sum */
/*     of the diagonal entries of the permuted matrix is maximized. */
/*     See [3]. */
/*   5 Compute a column permutation of the matrix so that the product */
/*     of the diagonal entries of the permuted matrix is maximized */
/*     and vectors to scale the matrix so that the nonzero diagonal */
/*     entries of the permuted matrix are one in absolute value and */
/*     all the off-diagonal entries are less than or equal to one in */
/*     absolute value. See [3]. */
/*  Restriction: 1 <= JOB <= 5. */

/* N is an PETSCINT variable which must be set by the user to the */
/*   order of the matrix A. It is not altered by the subroutine. */
/*   Restriction: N >= 1. */

/* NE is an PETSCINT variable which must be set by the user to the */
/*   number of entries in the matrix. It is not altered by the */
/*   subroutine. */
/*   Restriction: NE >= 1. */

/* IP is an PETSCINT array of length N+1. */
/*   IP(J), J=1..N, must be set by the user to the position in array IRN */
/*   of the first row index of an entry in column J. IP(N+1) must be set */
/*   to NE+1. It is not altered by the subroutine. */

/* IRN is an PETSCINT array of length NE. */
/*   IRN(K), K=1..NE, must be set by the user to hold the row indices of */
/*   the entries of the matrix. Those belonging to column J must be */
/*   stored contiguously in the positions IP(J)..IP(J+1)-1. The ordering */
/*   of the row indices within each column is unimportant. Repeated */
/*   entries are not allowed. The array IRN is not altered by the */
/*   subroutine. */

/* A is a DOUBLE PRECISION array of length NE. */
/*   The user must set A(K), K=1..NE, to the numerical value of the */
/*   entry that corresponds to IRN(K). */
/*   It is not used by the subroutine when JOB = 1. */
/*   It is not altered by the subroutine. */

/* NUM is an PETSCINT variable that need not be set by the user. */
/*   On successful exit, NUM will be the number of entries on the */
/*   diagonal of the permuted matrix. */
/*   If NUM < N, the matrix is structurally singular. */

/* CPERM is an PETSCINT array of length N that need not be set by the */
/*   user. On successful exit, CPERM contains the column permutation. */
/*   Column ABS(CPERM(J)) of the original matrix is column J in the */
/*   permuted matrix, J=1..N. For the N-NUM  entries of CPERM that are */
/*   not matched the permutation array is set negative so that a full */
/*   permutation of the matrix is obtained even in the structurally */
/*   singular case. */

/* LIW is an PETSCINT variable that must be set by the user to */
/*   the dimension of array IW. It is not altered by the subroutine. */
/*   Restriction: */
/*     JOB = 1 :  LIW >= 5N */
/*     JOB = 2 :  LIW >= 4N */
/*     JOB = 3 :  LIW >= 10N + NE */
/*     JOB = 4 :  LIW >= 5N */
/*     JOB = 5 :  LIW >= 5N */

/* IW is an PETSCINT array of length LIW that is used for workspace. */

/* LDW is an PETSCINT variable that must be set by the user to the */
/*   dimension of array DW. It is not altered by the subroutine. */
/*   Restriction: */
/*     JOB = 1 :  LDW is not used */
/*     JOB = 2 :  LDW >= N */
/*     JOB = 3 :  LDW >= NE */
/*     JOB = 4 :  LDW >= 2N + NE */
/*     JOB = 5 :  LDW >= 3N + NE */

/* DW is a DOUBLE PRECISION array of length LDW */
/*   that is used for workspace. If JOB = 5, on return, */
/*   DW(i) contains u_i, i=1..N, and DW(N+j) contains v_j, j=1..N. */

/* ICNTL is an PETSCINT array of length 10. Its components control the */
/*   output of MC64A/AD and must be set by the user before calling */
/*   MC64A/AD. They are not altered by the subroutine. */

/*   ICNTL(1) must be set to specify the output stream for */
/*   error messages. If ICNTL(1) < 0, messages are suppressed. */
/*   The default value set by MC46I/ID is 6. */

/*   ICNTL(2) must be set by the user to specify the output stream for */
/*   warning messages. If ICNTL(2) < 0, messages are suppressed. */
/*   The default value set by MC46I/ID is 6. */

/*   ICNTL(3) must be set by the user to specify the output stream for */
/*   diagnostic messages. If ICNTL(3) < 0, messages are suppressed. */
/*   The default value set by MC46I/ID is -1. */

/*   ICNTL(4) must be set by the user to a value other than 0 to avoid */
/*   checking of the input data. */
/*   The default value set by MC46I/ID is 0. */

/* INFO is an PETSCINT array of length 10 which need not be set by the */
/*   user. INFO(1) is set non-negative to indicate success. A negative */
/*   value is returned if an error occurred, a positive value if a */
/*   warning occurred. INFO(2) holds further information on the error. */
/*   On exit from the subroutine, INFO(1) will take one of the */
/*   following values: */
/*    0 : successful entry (for structurally nonsingular matrix). */
/*   +1 : successful entry (for structurally singular matrix). */
/*   +2 : the returned scaling factors are large and may cause */
/*        overflow when used to scale the matrix. */
/*        (For JOB = 5 entry only.) */
/*   -1 : JOB < 1 or JOB > 5.  Value of JOB held in INFO(2). */
/*   -2 : N < 1.  Value of N held in INFO(2). */
/*   -3 : NE < 1. Value of NE held in INFO(2). */
/*   -4 : the defined length LIW violates the restriction on LIW. */
/*        Value of LIW required given by INFO(2). */
/*   -5 : the defined length LDW violates the restriction on LDW. */
/*        Value of LDW required given by INFO(2). */
/*   -6 : entries are found whose row indices are out of range. INFO(2) */
/*        contains the index of a column in which such an entry is found. */
/*   -7 : repeated entries are found. INFO(2) contains the index of a */
/*        column in which such entries are found. */
/*  INFO(3) to INFO(10) are not currently used and are set to zero by */
/*        the routine. */

/* References: */
/*  [1]  I. S. Duff, (1981), */
/*       "Algorithm 575. Permutations for a zero-free diagonal", */
/*       ACM Trans. Math. Software 7(3), 387-390. */
/*  [2]  I. S. Duff and J. Koster, (1998), */
/*       "The design and use of algorithms for permuting large */
/*       entries to the diagonal of sparse matrices", */
/*       SIAM J. Matrix Anal. Appl., vol. 20, no. 4, pp. 889-901. */
/*  [3]  I. S. Duff and J. Koster, (1999), */
/*       "On algorithms for permuting large entries to the diagonal */
/*       of sparse matrices", */
/*       Technical Report RAL-TR-1999-030, RAL, Oxfordshire, England. */
/* Local variables and parameters */
/* External routines and functions */
/* Intrinsic functions */
/* Set RINF to largest positive real number (infinity) */
PetscErrorCode mc64AD(PetscInt *job, PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *num, PetscInt *cperm, 
                      PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw, PetscInt *icntl, PetscInt *info)
{
  PetscErrorCode ierr;
    /* Format strings */
#ifdef CHECKING
    static char fmt_9001[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2,\002 because \002,(a),\002 = \002,i10)";
    static char fmt_9004[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        LIW too small, must be at least \002,i8)";
    static char fmt_9005[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        LDW too small, must be at least \002,i8)";
    static char fmt_9006[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        Column \002,i8,\002 contains an entry with inval"
	    "id row index \002,i8)";
    static char fmt_9007[] = "(\002 ****** Error in MC64A/AD. INFO(1) = \002"
	    ",i2/\002        Column \002,i8,\002 contains two or more entries"
	    " with row index \002,i8)";
    static char fmt_9020[] = "(\002 ****** Input parameters for MC64A/AD:"
	    "\002/\002 JOB = \002,i8/\002 N   = \002,i8/\002 NE  = \002,i8)";
    static char fmt_9021[] = "(\002 IP(1:N+1)  = \002,8i8/(14x,8i8))";
    static char fmt_9022[] = "(\002 IRN(1:NE)  = \002,8i8/(14x,8i8))";
    static char fmt_9023[] = "(\002 A(1:NE)    = \002,4(1pd14.4)/(14x,4(1pd1"
	    "4.4)))";
    static char fmt_9011[] = "(\002 ****** Warning from MC64A/AD. INFO(1) ="
	    " \002,i2/\002        The matrix is structurally singular.\002)";
    static char fmt_9012[] = "(\002 ****** Warning from MC64A/AD. INFO(1) ="
	    " \002,i2/\002        Some scaling factors may be too large.\002)";
    static char fmt_9030[] = "(\002 ****** Output parameters for MC64A/AD"
	    ":\002/\002 INFO(1:2)  = \002,2i8)";
    static char fmt_9031[] = "(\002 NUM        = \002,i8)";
    static char fmt_9032[] = "(\002 CPERM(1:N) = \002,8i8/(14x,8i8))";
    static char fmt_9033[] = "(\002 DW(1:N)    = \002,5(f11.3)/(14x,5(f11.3)"
	    "))";
    static char fmt_9034[] = "(\002 DW(N+1:2N) = \002,5(f11.3)/(14x,5(f11.3)"
	    "))";
#endif

    /* System generated locals */
    PetscInt i__1, i__2;
    PetscScalar d__1, d__2;

    /* Builtin functions */
#ifdef CHECKING
    PetscInt s_wsfe(cilist *), do_fio(PetscInt *, char *, ftnlen), e_wsfe(void);
#endif

    /* Local variables */
    static PetscInt i__, j, k;
    static PetscScalar fact, rinf;

#ifdef CHECKING
    /* Fortran I/O blocks */
    static cilist io___16 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___17 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___18 = { 0, 0, 0, fmt_9001, 0 };
    static cilist io___20 = { 0, 0, 0, fmt_9004, 0 };
    static cilist io___21 = { 0, 0, 0, fmt_9005, 0 };
    static cilist io___24 = { 0, 0, 0, fmt_9006, 0 };
    static cilist io___25 = { 0, 0, 0, fmt_9007, 0 };
    static cilist io___26 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___27 = { 0, 0, 0, fmt_9021, 0 };
    static cilist io___28 = { 0, 0, 0, fmt_9022, 0 };
    static cilist io___29 = { 0, 0, 0, fmt_9023, 0 };
    static cilist io___31 = { 0, 0, 0, fmt_9011, 0 };
    static cilist io___32 = { 0, 0, 0, fmt_9012, 0 };
    static cilist io___33 = { 0, 0, 0, fmt_9030, 0 };
    static cilist io___34 = { 0, 0, 0, fmt_9031, 0 };
    static cilist io___35 = { 0, 0, 0, fmt_9032, 0 };
    static cilist io___36 = { 0, 0, 0, fmt_9033, 0 };
    static cilist io___37 = { 0, 0, 0, fmt_9034, 0 };
#endif

    PetscFunctionBegin;
    /* Parameter adjustments */
    --cperm;
    --ip;
    --a;
    --irn;
    --iw;
    --dw;
    --icntl;
    --info;

    /* Function Body */
    rinf = fd15AD((char *)"H", (ftnlen)1);
#ifdef CHECKING
/* Check value of JOB */
    if (*job < 1 || *job > 5) {
	info[1] = -1;
	info[2] = *job;
	if (icntl[1] >= 0) {
	    io___16.ciunit = icntl[1];
	    s_wsfe(&io___16);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)"JOB", (ftnlen)3);
	    do_fio(&c__1, (char *)&(*job), (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
	goto L99;
    }
/* Check value of N */
    if (*n < 1) {
	info[1] = -2;
	info[2] = *n;
	if (icntl[1] >= 0) {
	    io___17.ciunit = icntl[1];
	    s_wsfe(&io___17);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)"N", (ftnlen)1);
	    do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
	goto L99;
    }
/* Check value of NE */
    if (*ne < 1) {
	info[1] = -3;
	info[2] = *ne;
	if (icntl[1] >= 0) {
	    io___18.ciunit = icntl[1];
	    s_wsfe(&io___18);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)"NE", (ftnlen)2);
	    do_fio(&c__1, (char *)&(*ne), (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
	goto L99;
    }
#endif
/* Check LIW */
    if (*job == 1) {
	k = *n * 5;
    }
    if (*job == 2) {
	k = *n << 2;
    }
    if (*job == 3) {
	k = *n * 10 + *ne;
    }
    if (*job == 4) {
	k = *n * 5;
    }
    if (*job == 5) {
	k = *n * 5;
    }
#ifdef CHECKING
    if (*liw < k) {
	info[1] = -4;
	info[2] = k;
	if (icntl[1] >= 0) {
	    io___20.ciunit = icntl[1];
	    s_wsfe(&io___20);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
	goto L99;
    }
#endif
/* Check LDW */
/* If JOB = 1, do not check */
    if (*job > 1) {
	if (*job == 2) {
	    k = *n;
	}
	if (*job == 3) {
	    k = *ne;
	}
	if (*job == 4) {
	    k = (*n << 1) + *ne;
	}
	if (*job == 5) {
	    k = *n * 3 + *ne;
	}
#ifdef CHECKING
	if (*ldw < k) {
	    info[1] = -5;
	    info[2] = k;
	    if (icntl[1] >= 0) {
		io___21.ciunit = icntl[1];
		s_wsfe(&io___21);
		do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
		do_fio(&c__1, (char *)&k, (ftnlen)sizeof(PetscInt));
		e_wsfe();
	    }
	    goto L99;
	}
#endif
    }
    if (icntl[4] == 0) {
/* Check row indices. Use IW(1:N) as workspace */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    iw[i__] = 0;
/* L3: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		i__ = irn[k];
#ifdef CHECKING
/* Check for row indices that are out of range */
		if (i__ < 1 || i__ > *n) {
		    info[1] = -6;
		    info[2] = j;
		    if (icntl[1] >= 0) {
			io___24.ciunit = icntl[1];
			s_wsfe(&io___24);
			do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(
				PetscInt));
			do_fio(&c__1, (char *)&j, (ftnlen)sizeof(PetscInt));
			do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(PetscInt));
			e_wsfe();
		    }
		    goto L99;
		}
/* Check for repeated row indices within a column */
		if (iw[i__] == j) {
		    info[1] = -7;
		    info[2] = j;
		    if (icntl[1] >= 0) {
			io___25.ciunit = icntl[1];
			s_wsfe(&io___25);
			do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(
				PetscInt));
			do_fio(&c__1, (char *)&j, (ftnlen)sizeof(PetscInt));
			do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(PetscInt));
			e_wsfe();
		    }
		    goto L99;
		} else {
		    iw[i__] = j;
		}
/* L4: */
#endif
	    }
/* L6: */
	}
    }
#ifdef CHECKING
/* Print diagnostics on input */
    if (icntl[3] >= 0) {
	io___26.ciunit = icntl[3];
	s_wsfe(&io___26);
	do_fio(&c__1, (char *)&(*job), (ftnlen)sizeof(PetscInt));
	do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(PetscInt));
	do_fio(&c__1, (char *)&(*ne), (ftnlen)sizeof(PetscInt));
	e_wsfe();
	io___27.ciunit = icntl[3];
	s_wsfe(&io___27);
	i__1 = *n + 1;
	for (j = 1; j <= i__1; ++j) {
	    do_fio(&c__1, (char *)&ip[j], (ftnlen)sizeof(PetscInt));
	}
	e_wsfe();
	io___28.ciunit = icntl[3];
	s_wsfe(&io___28);
	i__1 = *ne;
	for (j = 1; j <= i__1; ++j) {
	    do_fio(&c__1, (char *)&irn[j], (ftnlen)sizeof(PetscInt));
	}
	e_wsfe();
	if (*job > 1) {
	    io___29.ciunit = icntl[3];
	    s_wsfe(&io___29);
	    i__1 = *ne;
	    for (j = 1; j <= i__1; ++j) {
		do_fio(&c__1, (char *)&a[j], (ftnlen)sizeof(PetscScalar));
	    }
	    e_wsfe();
	}
    }
#endif
/* Set components of INFO to zero */
    for (i__ = 1; i__ <= 10; ++i__) {
	info[i__] = 0;
/* L8: */
    }
/* Compute maximum matching with MC21A/AD */
    if (*job == 1) {
/* Put length of column J in IW(J) */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    iw[j] = ip[j + 1] - ip[j];
/* L10: */
	}
/* IW(N+1:5N) is workspace */
	ierr = mc21AD(n, &irn[1], ne, &ip[1], &iw[1], &cperm[1], num, &iw[*n + 1]);CHKERRQ(ierr);
	goto L90;
    }
/* Compute bottleneck matching */
    if (*job == 2) {
/* IW(1:5N), DW(1:N) are workspaces */
      ierr = mc64BD(n, ne, &ip[1], &irn[1], &a[1], &cperm[1], num, &iw[1], &iw[*n + 1], &iw[(*n << 1) + 1], &iw[*n * 3 + 1], &dw[1]);CHKERRQ(ierr);
	goto L90;
    }
/* Compute bottleneck matching */
    if (*job == 3) {
/* Copy IRN(K) into IW(K), ABS(A(K)) into DW(K), K=1..NE */
	i__1 = *ne;
	for (k = 1; k <= i__1; ++k) {
	    iw[k] = irn[k];
	    dw[k] = (d__1 = a[k], fabs(d__1));
/* L20: */
	}
/* Sort entries in each column by decreasing value. */
	ierr = mc64RD(n, ne, &ip[1], &iw[1], &dw[1]);CHKERRQ(ierr);
/* IW(NE+1:NE+10N) is workspace */
	ierr = mc64SD(n, ne, &ip[1], &iw[1], &dw[1], &cperm[1], num, &iw[*ne + 1], &iw[*ne + *n + 1], &iw[*ne + (*n << 1) + 1], &iw[*ne + *n * 3 + 1], &iw[*ne + (*n << 2) + 1], &iw[*ne + *n * 5 + 1], &iw[*ne + *n * 6 + 1]);CHKERRQ(ierr);
	goto L90;
    }
    if (*job == 4) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    fact = 0.;
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		if ((d__1 = a[k], fabs(d__1)) > fact) {
		    fact = (d__2 = a[k], fabs(d__2));
		}
/* L30: */
	    }
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		dw[(*n << 1) + k] = fact - (d__1 = a[k], fabs(d__1));
/* L40: */
	    }
/* L50: */
	}
/* B = DW(2N+1:2N+NE); IW(1:5N) and DW(1:2N) are workspaces */
	ierr = mc64WD(n, ne, &ip[1], &irn[1], &dw[(*n << 1) + 1], &cperm[1], num, &
                  iw[1], &iw[*n + 1], &iw[(*n << 1) + 1], &iw[*n * 3 + 1], &iw[(*n << 2) + 1], &dw[1], &dw[*n + 1]);CHKERRQ(ierr);
	goto L90;
    }
    if (*job == 5) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    fact = 0.;
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		dw[*n * 3 + k] = (d__1 = a[k], fabs(d__1));
		if (dw[*n * 3 + k] > fact) {
		    fact = dw[*n * 3 + k];
		}
/* L60: */
	    }
	    dw[(*n << 1) + j] = fact;
	    if (fact != 0.) {
		fact = log(fact);
	    } else {
		fact = rinf / *n;
	    }
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		if (dw[*n * 3 + k] != 0.) {
		    dw[*n * 3 + k] = fact - log(dw[*n * 3 + k]);
		} else {
		    dw[*n * 3 + k] = rinf / *n;
		}
/* L70: */
	    }
/* L75: */
	}
/* B = DW(3N+1:3N+NE); IW(1:5N) and DW(1:2N) are workspaces */
	ierr = mc64WD(n, ne, &ip[1], &irn[1], &dw[*n * 3 + 1], &cperm[1], num, &iw[1], &iw[*n + 1], &iw[(*n << 1) + 1], &iw[*n * 3 + 1], &iw[(*n << 2) + 1], &dw[1], &dw[*n + 1]);CHKERRQ(ierr);
	if (*num == *n) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (dw[(*n << 1) + j] != 0.) {
		    dw[*n + j] -= log(dw[(*n << 1) + j]);
		} else {
		    dw[*n + j] = 0.;
		}
/* L80: */
	    }
	}
/* Check size of scaling factors */
	fact = log(rinf) * .5f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (dw[j] < fact && dw[*n + j] < fact) {
		goto L86;
	    }
	    info[1] = 2;
	    goto L90;
L86:
	    ;
	}
/*       GO TO 90 */
    }
L90:
#ifdef CHECKING
    if (info[1] == 0 && *num < *n) {
/* Matrix is structurally singular, return with warning */
	info[1] = 1;
	if (icntl[2] >= 0) {
	    io___31.ciunit = icntl[2];
	    s_wsfe(&io___31);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
    }
    if (info[1] == 2) {
/* Scaling factors are large, return with warning */
	if (icntl[2] >= 0) {
	    io___32.ciunit = icntl[2];
	    s_wsfe(&io___32);
	    do_fio(&c__1, (char *)&info[1], (ftnlen)sizeof(PetscInt));
	    e_wsfe();
	}
    }
/* Print diagnostics on output */
    if (icntl[3] >= 0) {
	io___33.ciunit = icntl[3];
	s_wsfe(&io___33);
	for (j = 1; j <= 2; ++j) {
	    do_fio(&c__1, (char *)&info[j], (ftnlen)sizeof(PetscInt));
	}
	e_wsfe();
	io___34.ciunit = icntl[3];
	s_wsfe(&io___34);
	do_fio(&c__1, (char *)&(*num), (ftnlen)sizeof(PetscInt));
	e_wsfe();
	io___35.ciunit = icntl[3];
	s_wsfe(&io___35);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    do_fio(&c__1, (char *)&cperm[j], (ftnlen)sizeof(PetscInt));
	}
	e_wsfe();
	if (*job == 5) {
	    io___36.ciunit = icntl[3];
	    s_wsfe(&io___36);
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		do_fio(&c__1, (char *)&dw[j], (ftnlen)sizeof(PetscScalar));
	    }
	    e_wsfe();
	    io___37.ciunit = icntl[3];
	    s_wsfe(&io___37);
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		do_fio(&c__1, (char *)&dw[*n + j], (ftnlen)sizeof(PetscScalar))
			;
	    }
	    e_wsfe();
	}
    }
#endif
/* Return from subroutine. */
#ifdef CHECKING
L99:
#endif
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64BD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* N, NE, IP, IRN are described in MC64A/AD. */
/* A is a DOUBLE PRECISION array of length */
/*   NE. A(K), K=1..NE, must be set to the value of the entry */
/*   that corresponds to IRN(K). It is not altered. */
/* IPERM is an PETSCINT array of length N. On exit, it contains the */
/*    matching: IPERM(I) = 0 or row I is matched to column IPERM(I). */
/* NUM is PETSCINT variable. On exit, it contains the cardinality of the */
/*    matching stored in IPERM. */
/* IW is an PETSCINT work array of length 4N. */
/* DW is a DOUBLE PRECISION array of length N. */
/* Local variables */
/* Local parameters */
/* Intrinsic functions */
/* External subroutines and/or functions */
/* Set RINF to largest positive real number */
PetscErrorCode mc64BD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn,
                      PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm,
                      PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3;
    PetscScalar d__1, d__2, d__3;

    /* Local variables */
    static PetscInt i__, j, k;
    static PetscScalar a0;
    static PetscInt i0, q0;
    static PetscScalar ai, di;
    static PetscInt ii, jj, kk;
    static PetscScalar bv;
    static PetscInt up;
    static PetscScalar dq0;
    static PetscInt kk1, kk2;
    static PetscScalar csp;
    static PetscInt isp, jsp, low;
    static PetscScalar dnew;
    static PetscInt jord, qlen, idum, jdum;
    static PetscScalar rinf;
    static PetscInt lpos;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --d__;
    --l;
    --q;
    --pr;
    --jperm;
    --iperm;
    --ip;
    --a;
    --irn;

    /* Function Body */
    rinf = fd15AD((char *)"H", (ftnlen)1);
/* Initialization */
    *num = 0;
    bv = rinf;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	iperm[k] = 0;
	jperm[k] = 0;
	pr[k] = ip[k];
	d__[k] = 0.;
/* L10: */
    }
/* Scan columns of matrix; */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a0 = -1.;
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    ai = (d__1 = a[k], fabs(d__1));
	    if (ai > d__[i__]) {
		d__[i__] = ai;
	    }
	    if (jperm[j] != 0) {
		goto L30;
	    }
	    if (ai >= bv) {
		a0 = bv;
		if (iperm[i__] != 0) {
		    goto L30;
		}
		jperm[j] = i__;
		iperm[i__] = j;
		++(*num);
	    } else {
		if (ai <= a0) {
		    goto L30;
		}
		a0 = ai;
		i0 = i__;
	    }
L30:
	    ;
	}
	if (a0 != -1. && a0 < bv) {
	    bv = a0;
	    if (iperm[i0] != 0) {
		goto L20;
	    }
	    iperm[i0] = j;
	    jperm[j] = i0;
	    ++(*num);
	}
L20:
	;
    }
/* Update BV with smallest of all the largest maximum absolute values */
/* of the rows. */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MIN */
	d__1 = bv, d__2 = d__[i__];
	bv = PetscMin(d__1,d__2);
/* L25: */
    }
    if (*num == *n) {
	goto L1000;
    }
/* Rescan unassigned columns; improve initial assignment */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (jperm[j] != 0) {
	    goto L95;
	}
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    ai = (d__1 = a[k], fabs(d__1));
	    if (ai < bv) {
		goto L50;
	    }
	    if (iperm[i__] == 0) {
		goto L90;
	    }
	    jj = iperm[i__];
	    kk1 = pr[jj];
	    kk2 = ip[jj + 1] - 1;
	    if (kk1 > kk2) {
		goto L50;
	    }
	    i__3 = kk2;
	    for (kk = kk1; kk <= i__3; ++kk) {
		ii = irn[kk];
		if (iperm[ii] != 0) {
		    goto L70;
		}
		if ((d__1 = a[kk], fabs(d__1)) >= bv) {
		    goto L80;
		}
L70:
		;
	    }
	    pr[jj] = kk2 + 1;
L50:
	    ;
	}
	goto L95;
L80:
	jperm[jj] = ii;
	iperm[ii] = jj;
	pr[jj] = kk + 1;
L90:
	++(*num);
	jperm[j] = i__;
	iperm[i__] = j;
	pr[j] = k + 1;
L95:
	;
    }
    if (*num == *n) {
	goto L1000;
    }
/* Prepare for main loop */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = -1.;
	l[i__] = 0;
/* L99: */
    }
/* Main loop ... each pass round this loop is similar to Dijkstra's */
/* algorithm for solving the single source shortest path problem */
    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {
	if (jperm[jord] != 0) {
	    goto L100;
	}
	qlen = 0;
	low = *n + 1;
	up = *n + 1;
/* CSP is cost of shortest path to any unassigned row */
/* ISP is matrix position of unassigned row element in shortest path */
/* JSP is column index of unassigned row element in shortest path */
	csp = -1.;
/* Build shortest path tree starting from unassigned column JORD */
	j = jord;
	pr[j] = -1;
/* Scan column J */
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    dnew = (d__1 = a[k], fabs(d__1));
	    if (csp >= dnew) {
		goto L115;
	    }
	    if (iperm[i__] == 0) {
/* Row I is unassigned; update shortest path info */
		csp = dnew;
		isp = i__;
		jsp = j;
		if (csp >= bv) {
		    goto L160;
		}
	    } else {
		d__[i__] = dnew;
		if (dnew >= bv) {
/* Add row I to Q2 */
		    --low;
		    q[low] = i__;
		} else {
/* Add row I to Q, and push it */
		    ++qlen;
		    l[i__] = qlen;
		    ierr = mc64DD(&i__, n, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		}
		jj = iperm[i__];
		pr[jj] = j;
	    }
L115:
	    ;
	}
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
/* If Q2 is empty, extract new rows from Q */
	    if (low == up) {
		if (qlen == 0) {
		    goto L160;
		}
		i__ = q[1];
		if (csp >= d__[i__]) {
		    goto L160;
		}
		bv = d__[i__];
		i__3 = *n;
		for (idum = 1; idum <= i__3; ++idum) {
          ierr = mc64ED(&qlen, n, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		    l[i__] = 0;
		    --low;
		    q[low] = i__;
		    if (qlen == 0) {
			goto L153;
		    }
		    i__ = q[1];
		    if (d__[i__] != bv) {
			goto L153;
		    }
/* L152: */
		}
/* End of dummy loop; this point is never reached */
	    }
/* Move row Q0 */
L153:
	    --up;
	    q0 = q[up];
	    dq0 = d__[q0];
	    l[q0] = up;
/* Scan column that matches with row Q0 */
	    j = iperm[q0];
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		i__ = irn[k];
/* Update D(I) */
		if (l[i__] >= up) {
		    goto L155;
		}
/* Computing MIN */
		d__2 = dq0, d__3 = (d__1 = a[k], fabs(d__1));
		dnew = PetscMin(d__2,d__3);
		if (csp >= dnew) {
		    goto L155;
		}
		if (iperm[i__] == 0) {
/* Row I is unassigned; update shortest path info */
		    csp = dnew;
		    isp = i__;
		    jsp = j;
		    if (csp >= bv) {
			goto L160;
		    }
		} else {
		    di = d__[i__];
		    if (di >= bv || di >= dnew) {
			goto L155;
		    }
		    d__[i__] = dnew;
		    if (dnew >= bv) {
/* Delete row I from Q (if necessary); add row I to Q2 */
			if (di != -1.) {
			    lpos = l[i__];
			    ierr = mc64FD(&lpos, &qlen, n, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
			}
			l[i__] = 0;
			--low;
			q[low] = i__;
		    } else {
/* Add row I to Q (if necessary); push row I up Q */
			if (di == -1.) {
			    ++qlen;
			    l[i__] = qlen;
			}
			ierr = mc64DD(&i__, n, &q[1], &d__[1], &l[1], &c__1);CHKERRQ(ierr);
		    }
/* Update tree */
		    jj = iperm[i__];
		    pr[jj] = j;
		}
L155:
		;
	    }
/* L150: */
	}
/* If CSP = MINONE, no augmenting path is found */
L160:
	if (csp == -1.) {
	    goto L190;
	}
/* Update bottleneck value */
	bv = PetscMin(bv,csp);
/* Find augmenting path by tracing backward in PR; update IPERM,JPERM */
	++(*num);
	i__ = isp;
	j = jsp;
	i__2 = *num + 1;
	for (jdum = 1; jdum <= i__2; ++jdum) {
	    i0 = jperm[j];
	    jperm[j] = i__;
	    iperm[i__] = j;
	    j = pr[j];
	    if (j == -1) {
		goto L190;
	    }
	    i__ = i0;
/* L170: */
	}
/* End of dummy loop; this point is never reached */
L190:
	i__2 = *n;
	for (kk = up; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
	    l[i__] = 0;
/* L191: */
	}
	i__2 = up - 1;
	for (kk = low; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
/* L192: */
	}
	i__2 = qlen;
	for (kk = 1; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = -1.;
	    l[i__] = 0;
/* L193: */
	}
L100:
	;
    }
/* End of main loop */
/* BV is bottleneck value of final matching */
    if (*num == *n) {
	goto L1000;
    }
/* Matrix is structurally singular, complete IPERM. */
/* JPERM, PR are work arrays */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jperm[j] = 0;
/* L300: */
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] == 0) {
	    ++k;
	    pr[k] = i__;
	} else {
	    j = iperm[i__];
	    jperm[j] = i__;
	}
/* L310: */
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (jperm[i__] != 0) {
	    goto L320;
	}
	++k;
	jdum = pr[k];
	iperm[jdum] = -i__;
L320:
	;
    }
L1000:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64DD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* Variables N,Q,D,L are described in MC64B/BD */
/* IF IWAY is equal to 1, then */
/* node I is pushed from its current position upwards */
/* IF IWAY is not equal to 1, then */
/* node I is pushed from its current position downwards */
/* Local variables and parameters */
PetscErrorCode mc64DD(PetscInt *i__, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway)
{
    /* System generated locals */
    PetscInt i__1;

    /* Local variables */
    static PetscScalar di;
    static PetscInt qk, pos, idum, posk;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --l;
    --d__;
    --q;

    /* Function Body */
    pos = l[*i__];
    if (pos <= 1) {
	goto L20;
    }
    di = d__[*i__];
/* POS is index of current position of I in the tree */
    if (*iway == 1) {
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos / 2;
	    qk = q[posk];
	    if (di <= d__[qk]) {
		goto L20;
	    }
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
	    if (pos <= 1) {
		goto L20;
	    }
/* L10: */
	}
/* End of dummy loop; this point is never reached */
    } else {
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos / 2;
	    qk = q[posk];
	    if (di >= d__[qk]) {
		goto L20;
	    }
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
	    if (pos <= 1) {
		goto L20;
	    }
/* L15: */
	}
/* End of dummy loop; this point is never reached */
    }
/* End of dummy if; this point is never reached */
L20:
    q[pos] = *i__;
    l[*i__] = pos;
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64ED"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* Variables QLEN,N,Q,D,L are described in MC64B/BD (IWAY = 1) or */
/*     MC64W/WD (IWAY = 2) */
/* The root node is deleted from the binary heap. */
/* Local variables and parameters */
/* Move last element to begin of Q */
PetscErrorCode mc64ED(PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway)
{
    /* System generated locals */
    PetscInt i__1;

    /* Local variables */
    static PetscInt i__;
    static PetscScalar di, dk, dr;
    static PetscInt pos, idum, posk;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --l;
    --d__;
    --q;

    /* Function Body */
    i__ = q[*qlen];
    di = d__[i__];
    --(*qlen);
    pos = 1;
    if (*iway == 1) {
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos << 1;
	    if (posk > *qlen) {
		goto L20;
	    }
	    dk = d__[q[posk]];
	    if (posk < *qlen) {
		dr = d__[q[posk + 1]];
		if (dk < dr) {
		    ++posk;
		    dk = dr;
		}
	    }
	    if (di >= dk) {
		goto L20;
	    }
/* Exchange old last element with larger priority child */
	    q[pos] = q[posk];
	    l[q[pos]] = pos;
	    pos = posk;
/* L10: */
	}
/* End of dummy loop; this point is never reached */
    } else {
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos << 1;
	    if (posk > *qlen) {
		goto L20;
	    }
	    dk = d__[q[posk]];
	    if (posk < *qlen) {
		dr = d__[q[posk + 1]];
		if (dk > dr) {
		    ++posk;
		    dk = dr;
		}
	    }
	    if (di <= dk) {
		goto L20;
	    }
/* Exchange old last element with smaller child */
	    q[pos] = q[posk];
	    l[q[pos]] = pos;
	    pos = posk;
/* L15: */
	}
/* End of dummy loop; this point is never reached */
    }
/* End of dummy if; this point is never reached */
L20:
    q[pos] = i__;
    l[i__] = pos;
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64FD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* Variables QLEN,N,Q,D,L are described in MC64B/BD (IWAY = 1) or */
/*     MC64WD (IWAY = 2). */
/* Move last element in the heap */
/* Quick return, if possible */
PetscErrorCode mc64FD(PetscInt *pos0, PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway)
{
    /* System generated locals */
    PetscInt i__1;

    /* Local variables */
    static PetscInt i__;
    static PetscScalar di, dk, dr;
    static PetscInt qk, pos, idum, posk;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --l;
    --d__;
    --q;

    /* Function Body */
    if (*qlen == *pos0) {
	--(*qlen);
	return 0;
    }
/* Move last element from queue Q to position POS0 */
/* POS is current position of node I in the tree */
    i__ = q[*qlen];
    di = d__[i__];
    --(*qlen);
    pos = *pos0;
    if (*iway == 1) {
	if (pos <= 1) {
	    goto L20;
	}
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos / 2;
	    qk = q[posk];
	    if (di <= d__[qk]) {
		goto L20;
	    }
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
	    if (pos <= 1) {
		goto L20;
	    }
/* L10: */
	}
/* End of dummy loop; this point is never reached */
L20:
	q[pos] = i__;
	l[i__] = pos;
	if (pos != *pos0) {
	    return 0;
	}
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos << 1;
	    if (posk > *qlen) {
		goto L40;
	    }
	    dk = d__[q[posk]];
	    if (posk < *qlen) {
		dr = d__[q[posk + 1]];
		if (dk < dr) {
		    ++posk;
		    dk = dr;
		}
	    }
	    if (di >= dk) {
		goto L40;
	    }
	    qk = q[posk];
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
/* L30: */
	}
/* End of dummy loop; this point is never reached */
    } else {
	if (pos <= 1) {
	    goto L34;
	}
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos / 2;
	    qk = q[posk];
	    if (di >= d__[qk]) {
		goto L34;
	    }
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
	    if (pos <= 1) {
		goto L34;
	    }
/* L32: */
	}
/* End of dummy loop; this point is never reached */
L34:
	q[pos] = i__;
	l[i__] = pos;
	if (pos != *pos0) {
	    return 0;
	}
	i__1 = *n;
	for (idum = 1; idum <= i__1; ++idum) {
	    posk = pos << 1;
	    if (posk > *qlen) {
		goto L40;
	    }
	    dk = d__[q[posk]];
	    if (posk < *qlen) {
		dr = d__[q[posk + 1]];
		if (dk > dr) {
		    ++posk;
		    dk = dr;
		}
	    }
	    if (di <= dk) {
		goto L40;
	    }
	    qk = q[posk];
	    q[pos] = qk;
	    l[qk] = pos;
	    pos = posk;
/* L36: */
	}
/* End of dummy loop; this point is never reached */
    }
/* End of dummy if; this point is never reached */
L40:
    q[pos] = i__;
    l[i__] = pos;
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64RD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* This subroutine sorts the entries in each column of the */
/* sparse matrix (defined by N,NE,IP,IRN,A) by decreasing */
/* numerical value. */
/* Local constants */
/* Local variables */
/* Local arrays */
PetscErrorCode mc64RD(PetscInt *n, PetscInt *ne, const PetscInt *ip, PetscInt *irn, PetscScalar *a)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3;

    /* Local variables */
    static PetscInt j, k, r__, s;
    static PetscScalar ha;
    static PetscInt hi, td, mid, len, ipj;
    static PetscScalar key;
    static PetscInt last, todo[50], first;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ip;
    --a;
    --irn;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	len = ip[j + 1] - ip[j];
	if (len <= 1) {
	    goto L100;
	}
	ipj = ip[j];
/* Sort array roughly with partial quicksort */
	if (len < 15) {
	    goto L400;
	}
	todo[0] = ipj;
	todo[1] = ipj + len;
	td = 2;
L500:
	first = todo[td - 2];
	last = todo[td - 1];
/* KEY is the smallest of two values present in interval [FIRST,LAST) */
	key = a[(first + last) / 2];
	i__2 = last - 1;
	for (k = first; k <= i__2; ++k) {
	    ha = a[k];
	    if (ha == key) {
		goto L475;
	    }
	    if (ha > key) {
		goto L470;
	    }
	    key = ha;
	    goto L470;
L475:
	    ;
	}
/* Only one value found in interval, so it is already sorted */
	td += -2;
	goto L425;
/* Reorder interval [FIRST,LAST) such that entries before MID are gt KEY */
L470:
	mid = first;
	i__2 = last - 1;
	for (k = first; k <= i__2; ++k) {
	    if (a[k] <= key) {
		goto L450;
	    }
	    ha = a[mid];
	    a[mid] = a[k];
	    a[k] = ha;
	    hi = irn[mid];
	    irn[mid] = irn[k];
	    irn[k] = hi;
	    ++mid;
L450:
	    ;
	}
/* Both subintervals [FIRST,MID), [MID,LAST) are nonempty */
/* Stack the longest of the two subintervals first */
	if (mid - first >= last - mid) {
	    todo[td + 1] = last;
	    todo[td] = mid;
	    todo[td - 1] = mid;
/*          TODO(TD-1) = FIRST */
	} else {
	    todo[td + 1] = mid;
	    todo[td] = first;
	    todo[td - 1] = last;
	    todo[td - 2] = mid;
	}
	td += 2;
L425:
	if (td == 0) {
	    goto L400;
	}
/* There is still work to be done */
	if (todo[td - 1] - todo[td - 2] >= 15) {
	    goto L500;
	}
/* Next interval is already short enough for straightforward insertion */
	td += -2;
	goto L425;
/* Complete sorting with straightforward insertion */
L400:
	i__2 = ipj + len - 1;
	for (r__ = ipj + 1; r__ <= i__2; ++r__) {
	    if (a[r__ - 1] < a[r__]) {
		ha = a[r__];
		hi = irn[r__];
		a[r__] = a[r__ - 1];
		irn[r__] = irn[r__ - 1];
		i__3 = ipj + 1;
		for (s = r__ - 1; s >= i__3; --s) {
		    if (a[s - 1] < ha) {
			a[s] = a[s - 1];
			irn[s] = irn[s - 1];
		    } else {
			a[s] = ha;
			irn[s] = hi;
			goto L200;
		    }
/* L300: */
		}
		a[ipj] = ha;
		irn[ipj] = hi;
	    }
L200:
	    ;
	}
L100:
	;
    }
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64SD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* N, NE, IP, IRN, are described in MC64A/AD. */
/* A is a DOUBLE PRECISION array of length NE. */
/*   A(K), K=1..NE, must be set to the value of the entry that */
/*   corresponds to IRN(k). The entries in each column must be */
/*   non-negative and ordered by decreasing value. */
/* IPERM is an PETSCINT array of length N. On exit, it contains the */
/*   bottleneck matching: IPERM(I) - 0 or row I is matched to column */
/*   IPERM(I). */
/* NUMX is an PETSCINT variable. On exit, it contains the cardinality */
/*   of the matching stored in IPERM. */
/* IW is an PETSCINT work array of length 10N. */
/* FC is an PetscInt array of length N that contains the list of */
/*   unmatched columns. */
/* LEN(J), LENL(J), LENH(J) are PetscInt arrays of length N that point */
/*   to entries in matrix column J. */
/*   In the matrix defined by the column parts IP(J)+LENL(J) we know */
/*   a matching does not exist; in the matrix defined by the column */
/*   parts IP(J)+LENH(J) we know one exists. */
/*   LEN(J) lies between LENL(J) and LENH(J) and determines the matrix */
/*   that is tested for a maximum matching. */
/* W is an PetscInt array of length N and contains the indices of the */
/*   columns for which LENL ne LENH. */
/* WLEN is number of indices stored in array W. */
/* IW is PetscInt work array of length N. */
/* IW4 is PetscInt work array of length 4N used by MC64U/UD. */
/* BMIN and BMAX are such that a maximum matching exists for the input */
/*   matrix in which all entries smaller than BMIN are dropped. */
/*   For BMAX, a maximum matching does not exist. */
/* BVAL is a value between BMIN and BMAX. */
/* CNT is the number of calls made to MC64U/UD so far. */
/* NUM is the cardinality of last matching found. */
/* Set RINF to largest positive real number */
PetscErrorCode mc64SD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx, PetscInt *w, 
                      PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc, PetscInt *iw, PetscInt *iw4)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, l, ii, mod, cnt, num;
    static PetscScalar bval, bmin, bmax, rinf;
    static PetscInt nval, wlen, idum1, idum2, idum3;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iw4;
    --iw;
    --fc;
    --lenh;
    --lenl;
    --len;
    --w;
    --iperm;
    --ip;
    --a;
    --irn;

    /* Function Body */
    rinf = fd15AD((char *)"H", (ftnlen)1);
/* Compute a first maximum matching from scratch on whole matrix. */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	fc[j] = j;
	iw[j] = 0;
	len[j] = ip[j + 1] - ip[j];
/* L20: */
    }
/* The first call to MC64U/UD */
    cnt = 1;
    mod = 1;
    *numx = 0;
    ierr = mc64UD(&cnt, &mod, n, &irn[1], ne, &ip[1], &len[1], &fc[1], &iw[1], numx,
                  n, &iw4[1], &iw4[*n + 1], &iw4[(*n << 1) + 1], &iw4[*n * 3 + 1]);CHKERRQ(ierr);
/* IW contains a maximum matching of length NUMX. */
    num = *numx;
    if (num != *n) {
/* Matrix is structurally singular */
	bmax = rinf;
    } else {
/* Matrix is structurally nonsingular, NUM=NUMX=N; */
/* Set BMAX just above the smallest of all the maximum absolute */
/* values of the columns */
	bmax = rinf;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    bval = 0.f;
	    i__2 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__2; ++k) {
		if (a[k] > bval) {
		    bval = a[k];
		}
/* L25: */
	    }
	    if (bval < bmax) {
		bmax = bval;
	    }
/* L30: */
	}
	bmax *= 1.001f;
    }
/* Initialize BVAL,BMIN */
    bval = 0.f;
    bmin = 0.f;
/* Initialize LENL,LEN,LENH,W,WLEN according to BMAX. */
/* Set LEN(J), LENH(J) just after last entry in column J. */
/* Set LENL(J) just after last entry in column J with value ge BMAX. */
    wlen = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	l = ip[j + 1] - ip[j];
	lenh[j] = l;
	len[j] = l;
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    if (a[k] < bmax) {
		goto L46;
	    }
/* L45: */
	}
/* Column J is empty or all entries are ge BMAX */
	k = ip[j + 1];
L46:
	lenl[j] = k - ip[j];
/* Add J to W if LENL(J) ne LENH(J) */
	if (lenl[j] == l) {
	    goto L48;
	}
	++wlen;
	w[wlen] = j;
L48:
	;
    }
/* Main loop */
    i__1 = *ne;
    for (idum1 = 1; idum1 <= i__1; ++idum1) {
	if (num == *numx) {
/* We have a maximum matching in IW; store IW in IPERM */
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		iperm[i__] = iw[i__];
/* L50: */
	    }
/* Keep going round this loop until matching IW is no longer maximum. */
	    i__2 = *ne;
	    for (idum2 = 1; idum2 <= i__2; ++idum2) {
		bmin = bval;
		if (bmax == bmin) {
		    goto L99;
		}
/* Find splitting value BVAL */
		ierr = mc64QD(&ip[1], &lenl[1], &len[1], &w[1], &wlen, &a[1], &nval, &bval);CHKERRQ(ierr);
		if (nval <= 1) {
		    goto L99;
		}
/* Set LEN such that all matrix entries with value lt BVAL are */
/* discarded. Store old LEN in LENH. Do this for all columns W(K). */
/* Each step, either K is incremented or WLEN is decremented. */
		k = 1;
		i__3 = *n;
		for (idum3 = 1; idum3 <= i__3; ++idum3) {
		    if (k > wlen) {
			goto L71;
		    }
		    j = w[k];
		    i__4 = ip[j] + lenl[j];
		    for (ii = ip[j] + len[j] - 1; ii >= i__4; --ii) {
			if (a[ii] >= bval) {
			    goto L60;
			}
			i__ = irn[ii];
			if (iw[i__] != j) {
			    goto L55;
			}
/* Remove entry from matching */
			iw[i__] = 0;
			--num;
			fc[*n - num] = j;
L55:
			;
		    }
L60:
		    lenh[j] = len[j];
/* IP(J)+LEN(J)-1 is last entry in column ge BVAL */
		    len[j] = ii - ip[j] + 1;
/* If LENH(J) = LENL(J), remove J from W */
		    if (lenl[j] == lenh[j]) {
			w[k] = w[wlen];
			--wlen;
		    } else {
			++k;
		    }
/* L70: */
		}
L71:
		if (num < *numx) {
		    goto L81;
		}
/* L80: */
	    }
/* End of dummy loop; this point is never reached */
/* Set mode for next call to MC64U/UD */
L81:
	    mod = 1;
	} else {
/* We do not have a maximum matching in IW. */
	    bmax = bval;
/* BMIN is the bottleneck value of a maximum matching; */
/* for BMAX the matching is not maximum, so BMAX>BMIN */
/*          IF (BMAX .EQ. BMIN) GO TO 99 */
/* Find splitting value BVAL */
	    ierr = mc64QD(&ip[1], &len[1], &lenh[1], &w[1], &wlen, &a[1], &nval, &bval);CHKERRQ(ierr);
	    if (nval == 0 || bval == bmin) {
		goto L99;
	    }
/* Set LEN such that all matrix entries with value ge BVAL are */
/* inside matrix. Store old LEN in LENL. Do this for all columns W(K). */
/* Each step, either K is incremented or WLEN is decremented. */
	    k = 1;
	    i__2 = *n;
	    for (idum3 = 1; idum3 <= i__2; ++idum3) {
		if (k > wlen) {
		    goto L88;
		}
		j = w[k];
		i__3 = ip[j] + lenh[j] - 1;
		for (ii = ip[j] + len[j]; ii <= i__3; ++ii) {
		    if (a[ii] < bval) {
			goto L86;
		    }
/* L85: */
		}
L86:
		lenl[j] = len[j];
		len[j] = ii - ip[j];
		if (lenl[j] == lenh[j]) {
		    w[k] = w[wlen];
		    --wlen;
		} else {
		    ++k;
		}
/* L87: */
	    }
/* End of dummy loop; this point is never reached */
/* Set mode for next call to MC64U/UD */
L88:
	    mod = 0;
	}
	++cnt;
	ierr = mc64UD(&cnt, &mod, n, &irn[1], ne, &ip[1], &len[1], &fc[1], &iw[1], &
                  num, numx, &iw4[1], &iw4[*n + 1], &iw4[(*n << 1) + 1], &iw4[*n * 3 + 1]);CHKERRQ(ierr);
/* IW contains maximum matching of length NUM */
/* L90: */
    }
/* End of dummy loop; this point is never reached */
/* BMIN is bottleneck value of final matching */
L99:
    if (*numx == *n) {
	goto L1000;
    }
/* The matrix is structurally singular, complete IPERM */
/* W, IW are work arrays */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	w[j] = 0;
/* L300: */
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] == 0) {
	    ++k;
	    iw[k] = i__;
	} else {
	    j = iperm[i__];
	    w[j] = i__;
	}
/* L310: */
    }
    k = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (w[j] != 0) {
	    goto L320;
	}
	++k;
	idum1 = iw[k];
	iperm[idum1] = -j;
L320:
	;
    }
L1000:
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64QD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* This routine searches for at most XX different numerical values */
/* in the columns W(1:WLEN). XX>=2. */
/* Each column J is scanned between IP(J)+LENL(J) and IP(J)+LENH(J)-1 */
/* until XX values are found or all columns have been considered. */
/* On output, NVAL is the number of different values that is found */
/* and SPLIT(1:NVAL) contains the values in decreasing order. */
/* If NVAL > 0, the routine returns VAL = SPLIT((NVAL+1)/2). */

/* Scan columns in W(1:WLEN). For each encountered value, if value not */
/* already present in SPLIT(1:NVAL), insert value such that SPLIT */
/* remains sorted by decreasing value. */
/* The sorting is done by straightforward insertion; therefore the use */
/* of this routine should be avoided for large XX (XX < 20). */
PetscErrorCode mc64QD(const PetscInt *ip, PetscInt *lenl, PetscInt *lenh, PetscInt *w, PetscInt *wlen, PetscScalar *a, PetscInt *nval, PetscScalar *val)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3;

    /* Local variables */
    static PetscInt j, k, s;
    static PetscScalar ha;
    static PetscInt ii, pos;
    static PetscScalar split[10];

    PetscFunctionBegin;
    /* Parameter adjustments */
    --a;
    --w;
    --lenh;
    --lenl;
    --ip;

    /* Function Body */
    *nval = 0;
    i__1 = *wlen;
    for (k = 1; k <= i__1; ++k) {
	j = w[k];
	i__2 = ip[j] + lenh[j] - 1;
	for (ii = ip[j] + lenl[j]; ii <= i__2; ++ii) {
	    ha = a[ii];
	    if (*nval == 0) {
		split[0] = ha;
		*nval = 1;
	    } else {
/* Check presence of HA in SPLIT */
		for (s = *nval; s >= 1; --s) {
		    if (split[s - 1] == ha) {
			goto L15;
		    }
		    if (split[s - 1] > ha) {
			pos = s + 1;
			goto L21;
		    }
/* L20: */
		}
		pos = 1;
/* The insertion */
L21:
		i__3 = pos;
		for (s = *nval; s >= i__3; --s) {
		    split[s] = split[s - 1];
/* L22: */
		}
		split[pos - 1] = ha;
		++(*nval);
	    }
/* Exit loop if XX values are found */
	    if (*nval == 10) {
		goto L11;
	    }
L15:
	    ;
	}
/* L10: */
    }
/* Determine VAL */
L11:
    if (*nval > 0) {
	*val = split[(*nval + 1) / 2 - 1];
    }
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64UD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* PR(J) is the previous column to J in the depth first search. */
/*   Array PR is used as workspace in the sorting algorithm. */
/* Elements (I,IPERM(I)) I=1,..,N are entries at the end of the */
/*   algorithm unless N assignments have not been made in which case */
/*   N-NUM pairs (I,IPERM(I)) will not be entries in the matrix. */
/* CV(I) is the most recent loop number (ID+JORD) at which row I */
/*   was visited. */
/* ARP(J) is the number of entries in column J which have been scanned */
/*   when looking for a cheap assignment. */
/* OUT(J) is one less than the number of entries in column J which have */
/*   not been scanned during one pass through the main loop. */
/* NUMX is maximum possible size of matching. */
PetscErrorCode mc64UD(PetscInt *id, PetscInt *mod, PetscInt *n, PetscInt *irn, PetscInt *lirn, PetscInt *ip, PetscInt *lenc, PetscInt *fc, PetscInt *iperm,
                      PetscInt *num, PetscInt *numx, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, j1, ii, kk, id0, id1, in1, in2, nfc, num0, num1,
	     num2, jord, last;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --out;
    --cv;
    --arp;
    --pr;
    --iperm;
    --fc;
    --lenc;
    --ip;
    --irn;

    /* Function Body */
    if (*id == 1) {
/* The first call to MC64U/UD. */
/* Initialize CV and ARP; parameters MOD, NUMX are not accessed */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cv[i__] = 0;
	    arp[i__] = 0;
/* L5: */
	}
	num1 = *n;
	num2 = *n;
    } else {
/* Not the first call to MC64U/UD. */
/* Re-initialize ARP if entries were deleted since last call to MC64U/UD */
	if (*mod == 1) {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		arp[i__] = 0;
/* L8: */
	    }
	}
	num1 = *numx;
	num2 = *n - *numx;
    }
    num0 = *num;
/* NUM0 is size of input matching */
/* NUM1 is maximum possible size of matching */
/* NUM2 is maximum allowed number of unassigned rows/columns */
/* NUM is size of current matching */
/* Quick return if possible */
/*      IF (NUM.EQ.N) GO TO 199 */
/* NFC is number of rows/columns that could not be assigned */
    nfc = 0;
/* PetscInts ID0+1 to ID0+N are unique numbers for call ID to MC64U/UD, */
/* so 1st call uses 1..N, 2nd call uses N+1..2N, etc */
    id0 = (*id - 1) * *n;
/* Main loop. Each pass round this loop either results in a new */
/* assignment or gives a column with no assignment */
    i__1 = *n;
    for (jord = num0 + 1; jord <= i__1; ++jord) {
/* Each pass uses unique number ID1 */
	id1 = id0 + jord;
/* J is unmatched column */
	j = fc[jord - num0];
	pr[j] = -1;
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
/* Look for a cheap assignment */
	    if (arp[j] >= lenc[j]) {
		goto L30;
	    }
	    in1 = ip[j] + arp[j];
	    in2 = ip[j] + lenc[j] - 1;
	    i__3 = in2;
	    for (ii = in1; ii <= i__3; ++ii) {
		i__ = irn[ii];
		if (iperm[i__] == 0) {
		    goto L80;
		}
/* L20: */
	    }
/* No cheap assignment in row */
	    arp[j] = lenc[j];
/* Begin looking for assignment chain starting with row J */
L30:
	    out[j] = lenc[j] - 1;
/* Inner loop.  Extends chain by one or backtracks */
	    i__3 = jord;
	    for (kk = 1; kk <= i__3; ++kk) {
		in1 = out[j];
		if (in1 < 0) {
		    goto L50;
		}
		in2 = ip[j] + lenc[j] - 1;
		in1 = in2 - in1;
/* Forward scan */
		i__4 = in2;
		for (ii = in1; ii <= i__4; ++ii) {
		    i__ = irn[ii];
		    if (cv[i__] == id1) {
			goto L40;
		    }
/* Column J has not yet been accessed during this pass */
		    j1 = j;
		    j = iperm[i__];
		    cv[i__] = id1;
		    pr[j] = j1;
		    out[j1] = in2 - ii - 1;
		    goto L70;
L40:
		    ;
		}
/* Backtracking step. */
L50:
		j1 = pr[j];
		if (j1 == -1) {
/* No augmenting path exists for column J. */
		    ++nfc;
		    fc[nfc] = j;
		    if (nfc > num2) {
/* A matching of maximum size NUM1 is not possible */
			last = jord;
			goto L101;
		    }
		    goto L100;
		}
		j = j1;
/* L60: */
	    }
/* End of dummy loop; this point is never reached */
L70:
	    ;
	}
/* End of dummy loop; this point is never reached */
/* New assignment is made. */
L80:
	iperm[i__] = j;
	arp[j] = ii - ip[j] + 1;
	++(*num);
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
	    j = pr[j];
	    if (j == -1) {
		goto L95;
	    }
	    ii = ip[j] + lenc[j] - out[j] - 2;
	    i__ = irn[ii];
	    iperm[i__] = j;
/* L90: */
	}
/* End of dummy loop; this point is never reached */
L95:
	if (*num == num1) {
/* A matching of maximum size NUM1 is found */
	    last = jord;
	    goto L101;
	}

L100:
	;
    }
/* All unassigned columns have been considered */
    last = *n;
/* Now, a transversal is computed or is not possible. */
/* Complete FC before returning. */
L101:
    i__1 = *n;
    for (jord = last + 1; jord <= i__1; ++jord) {
	++nfc;
	fc[nfc] = fc[jord - num0];
/* L110: */
    }
/*  199 RETURN */
    PetscFunctionReturn(0);
}

/* ********************************************************************** */
#undef __FUNCT__
#define __FUNCT__ "mc64WD"
/* *** Copyright (c) 1999  Council for the Central Laboratory of the */
/*     Research Councils                                             *** */
/* *** Although every effort has been made to ensure robustness and  *** */
/* *** reliability of the subroutines in this MC64 suite, we         *** */
/* *** disclaim any liability arising through the use or misuse of   *** */
/* *** any of the subroutines.                                       *** */
/* *** Any problems?   Contact ...                                   *** */
/*     Iain Duff (I.Duff@rl.ac.uk) or                                *** */
/*     Jacko Koster (jacko.koster@uninett.no)                        *** */

/* N, NE, IP, IRN are described in MC64A/AD. */
/* A is a DOUBLE PRECISION array of length NE. */
/*   A(K), K=1..NE, must be set to the value of the entry that */
/*   corresponds to IRN(K). It is not altered. */
/*   All values A(K) must be non-negative. */
/* IPERM is an PETSCINT array of length N. On exit, it contains the */
/*   weighted matching: IPERM(I) = 0 or row I is matched to column */
/*   IPERM(I). */
/* NUM is an PETSCINT variable. On exit, it contains the cardinality of */
/*   the matching stored in IPERM. */
/* IW is an PETSCINT work array of length 5N. */
/* DW is a DOUBLE PRECISION array of length 2N. */
/*   On exit, U = D(1:N) contains the dual row variable and */
/*   V = D(N+1:2N) contains the dual column variable. If the matrix */
/*   is structurally nonsingular (NUM = N), the following holds: */
/*      U(I)+V(J) <= A(I,J)  if IPERM(I) |= J */
/*      U(I)+V(J)  = A(I,J)  if IPERM(I)  = J */
/*      U(I) = 0  if IPERM(I) = 0 */
/*      V(J) = 0  if there is no I for which IPERM(I) = J */
/* Local variables */
/* Local parameters */
/* External subroutines and/or functions */
/* Set RINF to largest positive real number */
PetscErrorCode mc64WD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *
	irn, PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm, 
	PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *u, 
	PetscScalar *d__)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt i__1, i__2, i__3;

    /* Local variables */
    static PetscInt i__, j, k, i0, k0, k1, k2, q0;
    static PetscScalar di;
    static PetscInt ii, jj, kk;
    static PetscScalar vj;
    static PetscInt up;
    static PetscScalar dq0;
    static PetscInt kk1, kk2;
    static PetscScalar csp;
    static PetscInt isp, jsp, low;
    static PetscScalar dmin__, dnew;
    static PetscInt jord, qlen, jdum;
    static PetscScalar rinf;
    static PetscInt lpos;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --d__;
    --u;
    --l;
    --q;
    --pr;
    --out;
    --jperm;
    --iperm;
    --ip;
    --a;
    --irn;

    /* Function Body */
    rinf = fd15AD((char *)"H", (ftnlen)1);
/* Initialization */
    *num = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	u[k] = rinf;
	d__[k] = 0.;
	iperm[k] = 0;
	jperm[k] = 0;
	pr[k] = ip[k];
	l[k] = 0;
/* L10: */
    }
/* Initialize U(I) */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    if (a[k] > u[i__]) {
		goto L20;
	    }
	    u[i__] = a[k];
	    iperm[i__] = j;
	    l[i__] = k;
L20:
	    ;
	}
/* L30: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = iperm[i__];
	if (j == 0) {
	    goto L40;
	}
/* Row I is not empty */
	iperm[i__] = 0;
	if (jperm[j] != 0) {
	    goto L40;
	}
/* Don't choose cheap assignment from dense columns */
	if (ip[j + 1] - ip[j] > *n / 10 && *n > 50) {
	    goto L40;
	}
/* Assignment of column J to row I */
	++(*num);
	iperm[i__] = j;
	jperm[j] = l[i__];
L40:
	;
    }
    if (*num == *n) {
	goto L1000;
    }
/* Scan unassigned columns; improve assignment */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* JPERM(J) ne 0 iff column J is already assigned */
	if (jperm[j] != 0) {
	    goto L95;
	}
	k1 = ip[j];
	k2 = ip[j + 1] - 1;
/* Continue only if column J is not empty */
	if (k1 > k2) {
	    goto L95;
	}
/*       VJ = RINF */
/* Changes made to allow for NaNs */
	i0 = irn[k1];
	vj = a[k1] - u[i0];
	k0 = k1;
	i__2 = k2;
	for (k = k1 + 1; k <= i__2; ++k) {
	    i__ = irn[k];
	    di = a[k] - u[i__];
	    if (di > vj) {
		goto L50;
	    }
	    if (di < vj || di == rinf) {
		goto L55;
	    }
	    if (iperm[i__] != 0 || iperm[i0] == 0) {
		goto L50;
	    }
L55:
	    vj = di;
	    i0 = i__;
	    k0 = k;
L50:
	    ;
	}
	d__[j] = vj;
	k = k0;
	i__ = i0;
	if (iperm[i__] == 0) {
	    goto L90;
	}
	i__2 = k2;
	for (k = k0; k <= i__2; ++k) {
	    i__ = irn[k];
	    if (a[k] - u[i__] > vj) {
		goto L60;
	    }
	    jj = iperm[i__];
/* Scan remaining part of assigned column JJ */
	    kk1 = pr[jj];
	    kk2 = ip[jj + 1] - 1;
	    if (kk1 > kk2) {
		goto L60;
	    }
	    i__3 = kk2;
	    for (kk = kk1; kk <= i__3; ++kk) {
		ii = irn[kk];
		if (iperm[ii] > 0) {
		    goto L70;
		}
		if (a[kk] - u[ii] <= d__[jj]) {
		    goto L80;
		}
L70:
		;
	    }
	    pr[jj] = kk2 + 1;
L60:
	    ;
	}
	goto L95;
L80:
	jperm[jj] = kk;
	iperm[ii] = jj;
	pr[jj] = kk + 1;
L90:
	++(*num);
	jperm[j] = k;
	iperm[i__] = j;
	pr[j] = k + 1;
L95:
	;
    }
    if (*num == *n) {
	goto L1000;
    }
/* Prepare for main loop */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = rinf;
	l[i__] = 0;
/* L99: */
    }
/* Main loop ... each pass round this loop is similar to Dijkstra's */
/* algorithm for solving the single source shortest path problem */
    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {
	if (jperm[jord] != 0) {
	    goto L100;
	}
/* JORD is next unmatched column */
/* DMIN is the length of shortest path in the tree */
	dmin__ = rinf;
	qlen = 0;
	low = *n + 1;
	up = *n + 1;
/* CSP is the cost of the shortest augmenting path to unassigned row */
/* IRN(ISP). The corresponding column index is JSP. */
	csp = rinf;
/* Build shortest path tree starting from unassigned column (root) JORD */
	j = jord;
	pr[j] = -1;
/* Scan column J */
	i__2 = ip[j + 1] - 1;
	for (k = ip[j]; k <= i__2; ++k) {
	    i__ = irn[k];
	    dnew = a[k] - u[i__];
	    if (dnew >= csp) {
		goto L115;
	    }
	    if (iperm[i__] == 0) {
		csp = dnew;
		isp = k;
		jsp = j;
	    } else {
		if (dnew < dmin__) {
		    dmin__ = dnew;
		}
		d__[i__] = dnew;
		++qlen;
		q[qlen] = k;
	    }
L115:
	    ;
	}
/* Initialize heap Q and Q2 with rows held in Q(1:QLEN) */
	q0 = qlen;
	qlen = 0;
	i__2 = q0;
	for (kk = 1; kk <= i__2; ++kk) {
	    k = q[kk];
	    i__ = irn[k];
	    if (csp <= d__[i__]) {
		d__[i__] = rinf;
		goto L120;
	    }
	    if (d__[i__] <= dmin__) {
		--low;
		q[low] = i__;
		l[i__] = low;
	    } else {
		++qlen;
		l[i__] = qlen;
		ierr = mc64DD(&i__, n, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
	    }
/* Update tree */
	    jj = iperm[i__];
	    out[jj] = k;
	    pr[jj] = j;
L120:
	    ;
	}
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
/* If Q2 is empty, extract rows from Q */
	    if (low == up) {
		if (qlen == 0) {
		    goto L160;
		}
		i__ = q[1];
		if (d__[i__] >= csp) {
		    goto L160;
		}
		dmin__ = d__[i__];
L152:
		ierr = mc64ED(&qlen, n, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
		--low;
		q[low] = i__;
		l[i__] = low;
		if (qlen == 0) {
		    goto L153;
		}
		i__ = q[1];
		if (d__[i__] > dmin__) {
		    goto L153;
		}
		goto L152;
	    }
/* Q0 is row whose distance D(Q0) to the root is smallest */
L153:
	    q0 = q[up - 1];
	    dq0 = d__[q0];
/* Exit loop if path to Q0 is longer than the shortest augmenting path */
	    if (dq0 >= csp) {
		goto L160;
	    }
	    --up;
/* Scan column that matches with row Q0 */
	    j = iperm[q0];
	    vj = dq0 - a[jperm[j]] + u[q0];
	    i__3 = ip[j + 1] - 1;
	    for (k = ip[j]; k <= i__3; ++k) {
		i__ = irn[k];
		if (l[i__] >= up) {
		    goto L155;
		}
/* DNEW is new cost */
		dnew = vj + a[k] - u[i__];
/* Do not update D(I) if DNEW ge cost of shortest path */
		if (dnew >= csp) {
		    goto L155;
		}
		if (iperm[i__] == 0) {
/* Row I is unmatched; update shortest path info */
		    csp = dnew;
		    isp = k;
		    jsp = j;
		} else {
/* Row I is matched; do not update D(I) if DNEW is larger */
		    di = d__[i__];
		    if (di <= dnew) {
			goto L155;
		    }
		    if (l[i__] >= low) {
			goto L155;
		    }
		    d__[i__] = dnew;
		    if (dnew <= dmin__) {
			lpos = l[i__];
			if (lpos != 0) {
              ierr = mc64FD(&lpos, &qlen, n, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
			}
			--low;
			q[low] = i__;
			l[i__] = low;
		    } else {
			if (l[i__] == 0) {
			    ++qlen;
			    l[i__] = qlen;
			}
			ierr = mc64DD(&i__, n, &q[1], &d__[1], &l[1], &c__2);CHKERRQ(ierr);
		    }
/* Update tree */
		    jj = iperm[i__];
		    out[jj] = k;
		    pr[jj] = j;
		}
L155:
		;
	    }
/* L150: */
	}
/* If CSP = RINF, no augmenting path is found */
L160:
	if (csp == rinf) {
	    goto L190;
	}
/* Find augmenting path by tracing backward in PR; update IPERM,JPERM */
	++(*num);
	i__ = irn[isp];
	iperm[i__] = jsp;
	jperm[jsp] = isp;
	j = jsp;
	i__2 = *num;
	for (jdum = 1; jdum <= i__2; ++jdum) {
	    jj = pr[j];
	    if (jj == -1) {
		goto L180;
	    }
	    k = out[j];
	    i__ = irn[k];
	    iperm[i__] = jj;
	    jperm[jj] = k;
	    j = jj;
/* L170: */
	}
/* End of dummy loop; this point is never reached */
/* Update U for rows in Q(UP:N) */
L180:
	i__2 = *n;
	for (kk = up; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    u[i__] = u[i__] + d__[i__] - csp;
/* L185: */
	}
L190:
	i__2 = *n;
	for (kk = low; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = rinf;
	    l[i__] = 0;
/* L191: */
	}
	i__2 = qlen;
	for (kk = 1; kk <= i__2; ++kk) {
	    i__ = q[kk];
	    d__[i__] = rinf;
	    l[i__] = 0;
/* L193: */
	}
L100:
	;
    }
/* End of main loop */
/* Set dual column variable in D(1:N) */
L1000:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	k = jperm[j];
	if (k != 0) {
	    d__[j] = a[k] - u[irn[k]];
	} else {
	    d__[j] = 0.;
	}
	if (iperm[j] == 0) {
	    u[j] = 0.;
	}
/* L200: */
    }
    if (*num == *n) {
	goto L1100;
    }
/* The matrix is structurally singular, complete IPERM. */
/* JPERM, OUT are work arrays */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jperm[j] = 0;
/* L300: */
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] == 0) {
	    ++k;
	    out[k] = i__;
	} else {
	    j = iperm[i__];
	    jperm[j] = i__;
	}
/* L310: */
    }
    k = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (jperm[j] != 0) {
	    goto L320;
	}
	++k;
	jdum = out[k];
	iperm[jdum] = -j;
L320:
	;
    }
L1100:
    PetscFunctionReturn(0);
}

/* COPYRIGHT (c) 1988 AEA Technology */
/* Original date 17 Feb 2005 */
/* 17th February 2005 Version 1.0.0. Replacement for FD05. */
PetscScalar fd15AD(char *t, ftnlen t_len)
{
    /* System generated locals */
    PetscScalar ret_val;

    /* Local variables */
    extern PetscScalar huge_(PetscScalar *), tiny_(PetscScalar *), radix_(
	    PetscScalar *), epsilon_(PetscScalar *);

/* ---------------------------------------------------------------- */
/*  Fortran 77 implementation of the Fortran 90 intrinsic */
/*    functions: EPSILON, TINY, HUGE and RADIX.  Note that */
/*    the RADIX result is returned as DOUBLE PRECISION. */

/*  The CHARACTER argument specifies the type of result: */

/*   'E'  smallest positive real number: 1.0 + DC(1) > 1.0, i.e. */
/*          EPSILON(DOUBLE PRECISION) */
/*   'T'  smallest full precision positive real number, i.e. */
/*          TINY(DOUBLE PRECISION) */
/*   'H'  largest finite positive real number, i.e. */
/*          HUGE(DOUBLE PRECISION) */
/*   'R'  the base of the floating point arithematic, i.e. */
/*          RADIX(DOUBLE PRECISION) */

/*    any other value gives a result of zero. */
/* ---------------------------------------------------------------- */
    if (*(unsigned char *)t == 'E') {
      ret_val = 2.22044604925031308E-016/*epsilon_(&c_b248)*/;
    } else if (*(unsigned char *)t == 'T') {
      ret_val = 2.22507385850720138E-308/*tiny_(&c_b248)*/;
    } else if (*(unsigned char *)t == 'H') {
      ret_val = 1.79769313486231571E+308/*huge_(&c_b248)*/;
    } else if (*(unsigned char *)t == 'R') {
      ret_val = (PetscScalar) 2/* radix_(&c_b248)*/;
    } else {
	ret_val = 0.;
    }
    return ret_val;
}

/* COPYRIGHT (c) 1977 AEA Technology */
/* Original date 8 Oct 1992 */
/* ######8/10/92 Toolpack tool decs employed. */
/* ######8/10/92 D version created by name change only. */
/* 13/3/02 Cosmetic changes applied to reduce single/double differences */

/* 12th July 2004 Version 1.0.0. Version numbering added. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */
PetscErrorCode mc21AD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *iw)
{
  PetscErrorCode ierr;
    /* System generated locals */
    PetscInt iw_dim1, iw_offset;


    /* Local variables */

    PetscFunctionBegin;
    /* Parameter adjustments */
    iw_dim1 = *n;
    iw_offset = 1 + iw_dim1;
    iw -= iw_offset;
    --iperm;
    --lenr;
    --ip;
    --icn;

    /* Function Body */
    ierr = mc21BD(n, &icn[1], licn, &ip[1], &lenr[1], &iperm[1], numnz, &iw[iw_dim1 + 1], &iw[(iw_dim1 << 1) + 1], &iw[iw_dim1 * 3 + 1], &iw[(iw_dim1 << 2) + 1]);CHKERRQ(ierr);
    PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "mc21BD"
/*   PR(I) IS THE PREVIOUS ROW TO I IN THE DEPTH FIRST SEARCH. */
/* IT IS USED AS A WORK ARRAY IN THE SORTING ALGORITHM. */
/*   ELEMENTS (IPERM(I),I) I=1, ... N  ARE NON-ZERO AT THE END OF THE */
/* ALGORITHM UNLESS N ASSIGNMENTS HAVE NOT BEEN MADE.  IN WHICH CASE */
/* (IPERM(I),I) WILL BE ZERO FOR N-NUMNZ ENTRIES. */
/*   CV(I) IS THE MOST RECENT ROW EXTENSION AT WHICH COLUMN I */
/* WAS VISITED. */
/*   ARP(I) IS ONE LESS THAN THE NUMBER OF NON-ZEROS IN ROW I */
/* WHICH HAVE NOT BEEN SCANNED WHEN LOOKING FOR A CHEAP ASSIGNMENT. */
/*   OUT(I) IS ONE LESS THAN THE NUMBER OF NON-ZEROS IN ROW I */
/* WHICH HAVE NOT BEEN SCANNED DURING ONE PASS THROUGH THE MAIN LOOP. */

/*   INITIALIZATION OF ARRAYS. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Executable Statements .. */
PetscErrorCode mc21BD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    static PetscInt i__, j, k, j1, ii, kk, in1, in2, jord, ioutk;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --out;
    --cv;
    --arp;
    --pr;
    --iperm;
    --lenr;
    --ip;
    --icn;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	arp[i__] = lenr[i__] - 1;
	cv[i__] = 0;
	iperm[i__] = 0;
/* L10: */
    }
    *numnz = 0;


/*   MAIN LOOP. */
/*   EACH PASS ROUND THIS LOOP EITHER RESULTS IN A NEW ASSIGNMENT */
/* OR GIVES A ROW WITH NO ASSIGNMENT. */
    i__1 = *n;
    for (jord = 1; jord <= i__1; ++jord) {
	j = jord;
	pr[j] = -1;
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
/* LOOK FOR A CHEAP ASSIGNMENT */
	    in1 = arp[j];
	    if (in1 < 0) {
		goto L30;
	    }
	    in2 = ip[j] + lenr[j] - 1;
	    in1 = in2 - in1;
	    i__3 = in2;
	    for (ii = in1; ii <= i__3; ++ii) {
		i__ = icn[ii];
		if (iperm[i__] == 0) {
		    goto L80;
		}
/* L20: */
	    }
/*   NO CHEAP ASSIGNMENT IN ROW. */
	    arp[j] = -1;
/*   BEGIN LOOKING FOR ASSIGNMENT CHAIN STARTING WITH ROW J. */
L30:
	    out[j] = lenr[j] - 1;
/* INNER LOOP.  EXTENDS CHAIN BY ONE OR BACKTRACKS. */
	    i__3 = jord;
	    for (kk = 1; kk <= i__3; ++kk) {
		in1 = out[j];
		if (in1 < 0) {
		    goto L50;
		}
		in2 = ip[j] + lenr[j] - 1;
		in1 = in2 - in1;
/* FORWARD SCAN. */
		i__4 = in2;
		for (ii = in1; ii <= i__4; ++ii) {
		    i__ = icn[ii];
		    if (cv[i__] == jord) {
			goto L40;
		    }
/*   COLUMN I HAS NOT YET BEEN ACCESSED DURING THIS PASS. */
		    j1 = j;
		    j = iperm[i__];
		    cv[i__] = jord;
		    pr[j] = j1;
		    out[j1] = in2 - ii - 1;
		    goto L70;

L40:
		    ;
		}

/*   BACKTRACKING STEP. */
L50:
		j = pr[j];
		if (j == -1) {
		    goto L100;
		}
/* L60: */
	    }

L70:
	    ;
	}

/*   NEW ASSIGNMENT IS MADE. */
L80:
	iperm[i__] = j;
	arp[j] = in2 - ii - 1;
	++(*numnz);
	i__2 = jord;
	for (k = 1; k <= i__2; ++k) {
	    j = pr[j];
	    if (j == -1) {
		goto L100;
	    }
	    ii = ip[j] + lenr[j] - out[j] - 2;
	    i__ = icn[ii];
	    iperm[i__] = j;
/* L90: */
	}

L100:
	;
    }

/*   IF MATRIX IS STRUCTURALLY SINGULAR, WE NOW COMPLETE THE */
/* PERMUTATION IPERM. */
    if (*numnz == *n) {
	PetscFunctionReturn(0);
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	arp[i__] = 0;
/* L110: */
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (iperm[i__] != 0) {
	    goto L120;
	}
	++k;
	out[k] = i__;
	goto L130;

L120:
	j = iperm[i__];
	arp[j] = i__;
L130:
	;
    }
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (arp[i__] != 0) {
	    goto L140;
	}
	++k;
	ioutk = out[k];
	iperm[ioutk] = i__;
L140:
	;
    }
    PetscFunctionReturn(0);

}

#endif /* PETSC_USE_COMPLEX */
