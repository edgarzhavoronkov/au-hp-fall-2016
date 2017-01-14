#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "image.h"

#include <boost/program_options.hpp>
#include <tbb/flow_graph.h>

using namespace std;
using namespace tbb::flow;
namespace po = boost::program_options;


int main(int argc, char **argv) 
{
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "describe arguments")
		("file,f", po::value<string>(), "file with intensity log")
		("limit,l", po::value<size_t>(), "limit of matrices, handled in parallel")
		("brightness,b", po::value<uint16_t>(), "intensity value to look for. Must be from 0 to 255");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) 
	{
		cout << desc << "\n";
		return -1;
	}

	if (vm.count("brightness") && vm.count("limit"))
	{
		//TODO: for debug purposes. Remove when commit
		uint16_t target_intensity = vm["brightness"].as<uint16_t>();
		size_t limit = vm["limit"].as<size_t>();

		if (target_intensity > image::max_intensity)
		{
			cout << desc << "\n";
			return -1;
		}

		cout << "Target value of intensity is: " << target_intensity << "\n";
		cout << "Limit of matrices handled simultaniously: " << limit << "\n";
		
		//handle all this shit

		size_t images_count = 100;
		size_t generated_count = 0;
		size_t width = 128;
		size_t height = 128;

		graph g;

		auto image_generator = [&images_count, &width, &height, &generated_count] (image& out) -> bool {
			if (generated_count < images_count)
			{
				image ret(width, height);
				ret.create_random();
				++generated_count;
				out = ret;
				return true;
			} 
			else
			{
				return false;
			}
		};

		source_node<image> source(g, image_generator);
		limiter_node<image> limiter(g, limit);
		broadcast_node<image> broadcast(g);

		auto get_max_intensity = [] (const image& img) -> points {
			return img.max_intensity_points();
		};

		auto get_min_intensity = [](const image& img) -> points {
			return img.min_intensity_points();
		};

		auto get_target_intensity = [&target_intensity](const image& img) -> points {
			return img.target_intensity_points(target_intensity);
		};

		function_node<image, points, queueing> maximizer(g, serial, get_max_intensity);
		function_node<image, points, queueing> minimizer(g, serial, get_min_intensity);
		function_node<image, points, queueing> matcher(g, serial, get_target_intensity);

		join_node<tuple<image, points, points, points>> join(g);

		auto mark_found_points = [](tuple<image, points, points, points> p) -> image {
			image img = get<0>(p);
			points maxs = get<1>(p);
			points mins = get<2>(p);
			points trgs = get<3>(p);
			img.mark_points(maxs);
			img.mark_points(mins);
			img.mark_points(trgs);
			return img;
		};

		function_node<tuple<image, points, points, points>, image, queueing> marker(g, serial, mark_found_points);

		broadcast_node<image> broadcast1(g);

		auto get_inverted_img = [] (const image& img) -> image {
			image res(img);
			res.invert_intensity();
			return res;
		};

		function_node<image, image, queueing> inverter(g, serial, get_inverted_img);

		auto get_mean_intensity = [] (const image& img) -> double {
			return img.mean_intensity();
		};

		function_node<image, double, queueing> meanimizer(g, serial, get_mean_intensity);

		make_edge(source, limiter);
		make_edge(limiter, broadcast);

		make_edge(broadcast, maximizer);
		make_edge(broadcast, minimizer);
		make_edge(broadcast, matcher);

		make_edge(broadcast, input_port<0>(join));
		make_edge(maximizer, input_port<1>(join));
		make_edge(minimizer, input_port<2>(join));
		make_edge(matcher, input_port<3>(join));

		make_edge(join, marker);
		
		make_edge(marker, broadcast1);
		
		make_edge(broadcast1, inverter);
		make_edge(broadcast1, meanimizer);

		if (vm.count("file"))
		{
			cout << "Output file is: " << vm["file"].as<string>() << "\n";
			ofstream out(vm["file"].as<string>());

			auto print_result = [&out](double res) -> continue_msg {
				out << res << "\n";
				return continue_msg();
			};

			function_node<double, continue_msg, queueing> printer(g, serial, print_result);

			make_edge(meanimizer, printer);
			make_edge(printer, limiter.decrement);

			source.activate();
			g.wait_for_all();

			return 0;
		}
		cout << "Output to stdout\n";

		auto print_result = [](double res) -> continue_msg {
				cout << res << "\n";
				return continue_msg();
			};

		function_node<double, continue_msg, queueing> printer(g, serial, print_result);

		make_edge(meanimizer, printer);
		make_edge(printer, limiter.decrement);

		source.activate();
		g.wait_for_all();

		return 0;
	}
	cout << desc << "\n";
	return -1;
}
