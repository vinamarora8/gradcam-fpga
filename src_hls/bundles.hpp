

#pragma HLS INTERFACE m_axi depth=1 bundle=fm_in port=input
#pragma HLS INTERFACE m_axi depth=1 bundle=fm_out port=output
#pragma HLS INTERFACE m_axi depth=1 bundle=fm_in port=fm_dram
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=conv1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=conv1_bias
// layer 1
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l10_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l10_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l10_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l10_c2_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l11_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l11_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l11_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l11_c2_bias
// layer 2
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l2_ds_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l2_ds_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l20_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l20_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l20_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l20_c2_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l21_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l21_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l21_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l21_c2_bias
// layer 3
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l3_ds_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l3_ds_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l30_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l30_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l30_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l30_c2_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l31_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l31_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l31_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l31_c2_bias
// layer 4
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l4_ds_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l4_ds_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l40_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l40_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l40_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l40_c2_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l41_c1_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l41_c1_bias
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l41_c2_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=l41_c2_bias
// fc
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=fc_weight
#pragma HLS INTERFACE m_axi depth=1 bundle=wt port=fc_bias
