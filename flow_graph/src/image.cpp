#include "image.h"
#include <vector>

image::image()
	: image(0, 0)
{
}

image::image(size_t width, size_t height)
	: width_(width)
	, height_(height)
{
	data_ = std::vector<std::vector<uint16_t>>(height, std::vector<uint16_t>(width));
}

image::image(const image& other): width_(other.width_),
                                  height_(other.height_),
                                  data_(other.data_)
{
}

image::image(image&& other) noexcept: width_(other.width_),
                                      height_(other.height_),
                                      data_(std::move(other.data_))
{
}

image& image::operator=(image other)
{
	using std::swap;
	swap(width_, other.width_);
	swap(height_, other.height_);
	swap(data_, other.data_);
	return *this;
}

void image::create_random()
{
	for (size_t i = 0; i < height_; ++i)
	{
		for (size_t j = 0; j < width_; ++j)
		{
			data_[i][j] = rand() % (max_intensity + 1);
		}
	}
}

points image::min_intensity_points() const
{
	points ret;
	uint16_t curr_min = max_intensity + 1;
	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			if (data_[i][j] < curr_min)
			{
				curr_min = data_[i][j];
			}
		}
	}

	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			if (data_[i][j] == curr_min) 
			{
				ret.push_back(std::make_pair(i,j));
			}
		}
	}
	return ret;
}

points image::max_intensity_points() const
{
	points ret;
	uint16_t curr_max = -(max_intensity + 1);
	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			if (data_[i][j] > curr_max) 
			{
				curr_max = data_[i][j];
			}
		}
	}

	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			if (data_[i][j] == curr_max) 
			{
				ret.push_back(std::make_pair(i, j));
			}
		}
	}
	return ret;
}

points image::target_intensity_points(uint16_t target) const
{
	points ret;
	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			if (data_[i][j] == target) 
			{
				ret.push_back(std::make_pair(i, j));
			}
		}
	}
	return ret;
}

void image::invert_intensity()
{
	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			data_[i][j] = max_intensity - data_[i][j];
		}
	}
}

double image::mean_intensity() const
{
	uint64_t sum = 0;
	for (size_t i = 0; i < height_; ++i) 
	{
		for (size_t j = 0; j < width_; ++j) 
		{
			sum += data_[i][j];
		}
	}
	return (sum * 1.0) / (width_ * height_);
}

void image::mark_points(points pts)
{
	for (auto point : pts)
	{
		mark_square(point.first, point.second);
	}
}

void image::mark_square(size_t center_y, size_t center_x)
{
	for (size_t i = center_y - 1; i < center_y + 1; ++i)
	{
		for (size_t j = center_x - 1; j < center_x + 1; ++j) 
		{
			mark_point(i, j, max_intensity);
		}
	}
}

void image::mark_point(size_t y, size_t x, uint16_t mark)
{
	if (y >= 0 || y < height_ && x >= 0 || x < width_)
	{
		data_[y][x] = mark;
	}
}
