#pragma once
#include <vector>
#include <cstdint>
#include <utility>

typedef std::vector<std::pair<size_t, size_t>> points;

class image
{
public:
	image();
	image(size_t width, size_t height);
	image(const image& other);
	image(image&& other) noexcept;
	image& operator=(image other);

	void create_random();
	points min_intensity_points() const;
	points max_intensity_points() const;
	points target_intensity_points(uint16_t target) const;
	void invert_intensity();
	double mean_intensity() const;
	void mark_points(points pts);
	const static uint16_t max_intensity = 255;
private:
	size_t width_;
	size_t height_;
	std::vector<std::vector<uint16_t>> data_;
	// ReSharper disable CppRedundantAccessSpecifier
private:
	// ReSharper restore CppRedundantAccessSpecifier
	void mark_square(size_t center_y, size_t center_x);
	void mark_point(size_t y, size_t x, uint16_t mark);
};
