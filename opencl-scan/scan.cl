#define SWAP(a,b) {__local double * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global double * input, __global double * output, __global double* block_sums, __local double * a, __local double * b) 
{
	uint lid = get_local_id(0);
	uint grid = get_group_id(0);
	uint block_size = get_local_size(0);
	uint idx = lid + grid * block_size;

	a[lid] = b[lid] = input[idx];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (uint s = 1; s < block_size; s <<= 1) 
	{
		if (lid > (s - 1)) 
		{
			b[lid] = a[lid] + a[lid - s];
		} 
		else
		 {
			b[lid] = a[lid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		SWAP(a, b);
	}

	output[idx] = a[lid];
	if (lid == block_size - 1) 
	{
		block_sums[grid] = a[lid];
	}
}

__kernel void add_block_sums(__global double* input, __global double* block_sums) 
{
	uint lid = get_local_id(0);
	uint grid = get_group_id(0);
	uint block_size = get_local_size(0);

	if (grid > 0) 
	{
		input[lid + grid * block_size] += block_sums[grid - 1];
	}
}