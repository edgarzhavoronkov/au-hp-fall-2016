__kernel void matrix_conv(__global float * input, __global float * mask, __global float * result, int m, int n) 
{
	int row = get_global_id(0);
	int col = get_global_id(1);

	int HM = (m - 1) / 2;

	if (row >= n || col >= n)
		return;

	float conv = 0;

	for (int k = -HM; k <= HM; ++k) 
	{
		for (int l = -HM; l <= HM; ++l) 
		{
			int i = row + k;
			int j = col + l;

			float a;

			if (i < 0 || j < 0 || i >= n || j >= n) 
			{
				a = 0;
			}
			else
			{
				a = input[i * n + j];
			}

			conv += a * mask[(k + HM) * m + (l + HM)];
		}
	}

	result[row * n + col] = conv;
}