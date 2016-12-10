#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/cl.h>

#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	try
	{
		cl::Platform::get(&platforms);

		for (cl_uint i = 0; i < platforms.size(); ++i)
		{
			std::string version;
			std::string name;
			platforms[i].getInfo(CL_PLATFORM_NAME, &name);
			platforms[i].getInfo(CL_PLATFORM_VERSION, &version);
			if (name.find("NVIDIA") != std::string::npos)
			{
				platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
			}
		}

		cl::Context context(devices);

		// create command queue
		cl::CommandQueue queue(context, devices[0]);

		// load opencl source
		std::ifstream cl_file("2d_convolution.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

		// create program
		cl::Program program(context, source);

		// compile opencl source
		cl_uint err = program.build(devices, "-D BLOCK_SIZE=16");
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			std::string log;
			program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
			std::cout << log << std::endl;
		}

		// create a message to send to kernel
		std::ifstream in("input.txt");
		size_t N, M;
		in >> N >> M;

		std::vector<float> a(N * N, 1.);
		std::vector<float> b(M * M, 1.);
		std::vector<float> c(N * N, 1.);

		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = 0; j < N; ++j)
			{
				float value;
				in >> value;
				a[i * N + j] = value;
			}
		}

		for (size_t i = 0; i < M; ++i)
		{
			for (size_t j = 0; j < M; ++j)
			{
				float value;
				in >> value;
				b[i * M + j] = value;
			}
		}

		// allocate device buffer to hold message
		cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(int) * N * N);
		cl::Buffer dev_mask(context, CL_MEM_READ_ONLY, sizeof(int) * M * M);
		cl::Buffer dev_result(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N);

		// copy from cpu to gpu
		queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * N * N, &a[0]);
		queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(int) * M * M, &b[0]);
		queue.flush();

		// load named kernel from opencl source
		cl::Kernel kernel(program, "matrix_conv");
		cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, int, int> matrix_conv(kernel);
		int size = N % 16 == 0 ? N : N + (16 - N % 16);
		cl::EnqueueArgs args(queue, cl::NullRange, cl::NDRange(size, size), cl::NDRange(16, 16));
		matrix_conv(args, dev_input, dev_mask, dev_result, (int)M, (int)N);

		queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, sizeof(int) * N * N, &c[0]);

		std::ofstream out("output.txt");

		out << std::fixed << std::setprecision(3);

		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = 0; j < N; ++j)
			{
				size_t idx = i * N + j;
				out << c[idx] << " ";
			}
			out << '\n';
		}
		std::cout << "finished" << std::endl;
	}
	catch (cl::Error e)
	{
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
	}

	return 0;
}
