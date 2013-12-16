testbed: testbed.o chkopts
	-${CLINKER} -o testbed testbed.o ${PETSC_LIB}
	#${RM} testbed.o

spectralPartition: spectralPartition.o chkopts
	-${CLINKER} -o spectralPartition spectralPartition.o ${PETSC_LIB}
	#${RM} spectralPartition.o

wbm: wbm.o chkopts
	-${CLINKER} -o wbm wbm.o ${PETSC_LIB}
	#${RM} wbm.o

run:
	-@${MPIEXEC} -n 1 ./testbed -mat_no_inode -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_monitor_true_residual -ksp_converged_reason -ksp_max_it 500 -pc_type lu -pc_factor_shift_type nonzero -draw_pause 1 ${ARGS}

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
