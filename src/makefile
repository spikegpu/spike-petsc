testbed: testbed.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o
	-${CLINKER} -o testbed testbed.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o ${PETSC_LIB}
	#${RM} testbed.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o

testbed2: testbed2.o matbanded.o kspreorder.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o petsc_mat_fiedler.o hslmc73f.o
	-${CLINKER} -o testbed2 testbed2.o matbanded.o kspreorder.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o petsc_mat_fiedler.o hslmc73f.o -L. -lhsl_mc73 ${PETSC_LIB}
	#${RM} testbed2.o matbanded.o kspreorder.o petsc_mat_awbm.o petsc_mat_wbm.o hslmc64.o

spectralPartition: spectralPartition.o
	-${CLINKER} -o spectralPartition spectralPartition.o ${PETSC_LIB}
	#${RM} spectralPartition.o

wbm: wbm.o
	-${CLINKER} -o wbm wbm.o ${PETSC_LIB}
	#${RM} wbm.o

run:
	-@${MPIEXEC} -n 1 ./testbed -mat_no_inode -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_monitor_true_residual -ksp_converged_reason -ksp_max_it 500 -pc_type lu -pc_factor_shift_type nonzero -draw_pause 1 ${ARGS}

clean::
	${RM} *.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
