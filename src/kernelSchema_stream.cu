


void EG_gpu_solver_OutdegreeSplit() {

	...
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	...

	EG_gpu_traspose_graph();
	...
	kernel_EG_initialize_split<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, counter_nodi0_2, num_nodi, MG_pesi); // nel default streem
	...
	remove_nulls(hdev_nodeFlags1, configuration.shuffleSplit_index, &numAttivi1); // nel default streem
	remove_nulls(hdev_nodeFlags1+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2); // nel default streem
	...

	while ((extloop>0) && (numAttivi>0)) {
		if (numAttivi > configuration.shuffleSplit_soglia) { // in due strean concorrenti
			if (numAttivi1>0) {
				...
				kernel_EG_all_global_NEW1to2_none<<<nbs, tpb, 0, stream1>>>(0, counter_nodi0_1, numAttivi1, MG_pesi);
				...
			}
			if (numAttivi2>0) {
				...
				kernel_EG_all_global_NEW1to2_none<<<nbs, tpb, 0, stream2>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi2, MG_pesi);
				...
			}
			cudaDeviceSynchronize();
		} else { // nel default streem
			...
			kernel_EG_all_global_NEW1to2_none_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1,numAttivi, MG_pesi);
			...
		}

		...
		remove_nulls(hdev_nodeFlags2, configuration.shuffleSplit_index, &numAttivi1); // nel default streem
		remove_nulls(hdev_nodeFlags2+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2); // nel default streem
		...


		if (numAttivi > configuration.shuffleSplit_soglia) { // in due strean concorrenti
			if (numAttivi1>0) {
				...
				kernel_EG_all_global_NEW2to1_none<<<nbs, tpb, 0, stream1>>>(0, counter_nodi0_1, numAttivi1, MG_pesi);
				...
			}
			if (numAttivi2>0) {
				...
				kernel_EG_all_global_NEW2to1_none<<<nbs, tpb, 0, stream2>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi2, MG_pesi);
				...
			}
			cudaDeviceSynchronize();
		} else { // nel default streem
			...
			kernel_EG_all_global_NEW2to1_none_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1,numAttivi, MG_pesi);
			...
		}

		...
		remove_nulls(hdev_nodeFlags1, configuration.shuffleSplit_index, &numAttivi1); // nel default streem
		remove_nulls(hdev_nodeFlags1+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2); // nel default streem
		...
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}




