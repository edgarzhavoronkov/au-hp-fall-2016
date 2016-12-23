#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/cl.h>

#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <memory>
#include <algorithm>

typedef cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg, cl::LocalSpaceArg> ScanKernel;
typedef cl::make_kernel<cl::Buffer &, cl::Buffer &> AddKernel;

const size_t BLOCK_SIZE = 256;

struct OpenCL
{
	std::unique_ptr<cl::Context> context;
	std::unique_ptr<cl::CommandQueue> queue;
	std::unique_ptr<cl::Program> program;
};

std::unique_ptr<OpenCL> init()
{
	std::unique_ptr<OpenCL> res = std::make_unique<OpenCL>();

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	cl::Platform::get(&platforms);

	for (cl_uint i = 0; i < platforms.size(); ++i) {
		std::string version;
		std::string name;
		platforms[i].getInfo(CL_PLATFORM_NAME, &name);
		platforms[i].getInfo(CL_PLATFORM_VERSION, &version);
		if (name.find("NVIDIA") != std::string::npos) {
			platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		}
	}

	cl::Context context(devices);

	// create command queue
	cl::CommandQueue queue(context, devices[0]);

	// load opencl source
	std::ifstream cl_file("scan.cl");
	std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

	// create program
	cl::Program program(context, source);

	// compile opencl source
	cl_uint err = program.build(devices, "-D BLOCK_SIZE=256");
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		std::string log;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
		std::cout << log << std::endl;
	}

	res->context = std::make_unique<cl::Context>(context);
	res->queue = std::make_unique<cl::CommandQueue>(queue);
	res->program = std::make_unique<cl::Program>(program);
	return res;
}

std::vector<double> scan(std::vector<double>& input, ScanKernel& scanner, AddKernel& adder, std::unique_ptr<OpenCL>& opencl)
{
	int blocks_count = input.size() / BLOCK_SIZE;
	if (blocks_count == 0)
	{
		blocks_count = 1;
	}
	if (blocks_count > BLOCK_SIZE && blocks_count % BLOCK_SIZE != 0)
	{
		blocks_count += BLOCK_SIZE - (blocks_count % BLOCK_SIZE);
	}

	std::vector<double> block_sums(blocks_count, 0.0);

	cl::Buffer dev_input(*(opencl->context), CL_MEM_READ_ONLY, sizeof(double) * input.size());
	cl::Buffer dev_output(*(opencl->context), CL_MEM_READ_ONLY, sizeof(double) * input.size());
	cl::Buffer dev_block_sums(*(opencl->context), CL_MEM_READ_ONLY, sizeof(double) * blocks_count);

	opencl->queue->enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);

	cl::EnqueueArgs scanner_args(*(opencl->queue), cl::NullRange, cl::NDRange(input.size()), cl::NDRange(std::min(BLOCK_SIZE, input.size())));
	scanner(scanner_args, dev_input, dev_output, dev_block_sums, cl::Local(sizeof(double) * std::min(BLOCK_SIZE, input.size())), cl::Local(sizeof(double) * std::min(BLOCK_SIZE, input.size())));

	if (input.size() > BLOCK_SIZE)
	{
		opencl->queue->enqueueReadBuffer(dev_block_sums, CL_TRUE, 0, sizeof(double) * block_sums.size(), &block_sums[0]);
		block_sums = scan(block_sums, scanner, adder, opencl);
		opencl->queue->enqueueWriteBuffer(dev_block_sums, CL_TRUE, 0, sizeof(double) * block_sums.size(), &block_sums[0]);

		cl::EnqueueArgs add_args(*(opencl->queue), cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
		adder(add_args, dev_output, dev_block_sums);
	}

	std::vector<double> result(input.size(), 0.0);
	opencl->queue->enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * result.size(), &result[0]);
	return result;
}

int main() 
{

	try 
	{
		std::unique_ptr<OpenCL> opencl = init();

		// create a message to send to kernel
		std::ifstream in("input.txt");
		size_t N;
		
		in >> N;

		size_t input_size = N;

		if (N % BLOCK_SIZE != 0)
		{
			input_size += BLOCK_SIZE - (N % BLOCK_SIZE);
		}

		std::vector<double> input(input_size, 0.0);

		for (size_t i = 0; i < N; ++i) 
		{
			in >> input[i];
		}

		// load named kernel from opencl source
		cl::Kernel scan_kernel(*(opencl->program), "scan_hillis_steele");
		ScanKernel scanner(scan_kernel);

		cl::Kernel add_kernel(*(opencl->program), "add_block_sums");
		AddKernel adder(add_kernel);

		std::vector<double> res = scan(input, scanner, adder, opencl);

		std::ofstream out("output.txt");

		out << std::fixed << std::setprecision(3);

		for (size_t i = 0; i < N; ++i) 
		{
			out << res[i] << " ";
		}
		out << std::endl;
		std::cout << "finished" << std::endl;
	}
	catch (cl::Error e) 
	{
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
	}

	return 0;
}
