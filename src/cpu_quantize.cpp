#include <array>
#include <cmath>
#include <iostream>
#include <limits>

std::array<int8_t, 7> quantize_int8(std::array<float, 7> &arr) {
  // find ABS max of arr/matrix
  float absMax{-std::numeric_limits<float>::infinity()};

  for (int i{}; i < arr.size(); i++) {
    absMax = std::max(absMax, std::abs(arr[i]));
  }

  std::cout << "ABSMAX: " << absMax << '\n';

  // calculate scale factor
  float scale{127.0f / absMax};

  std::cout << "scale factor: " << scale << '\n' << '\n';

  std::array<int8_t, 7> result{};

  // quantize arr/matrix using the scale factor
  for (int i{}; i < arr.size(); i++) {
    result[i] = static_cast<int8_t>(std::round(arr[i] * scale));
  }

  return result;
}

std::array<float, 7> dequantize_int8(std::array<int8_t, 7> arr) {

  const float scale{11.7593};

  std::array<float, 7> result{};

  // take in quantized arr/matrix and dequantize
  for (int i{}; i < arr.size(); i++) {
    result[i] = arr[i] / scale;
  }

  return result;
}

int main() {

  std::array<float, 7> arr{{5.47f, 3.08f, -7.59f, 0.0f, -1.95f, -4.57f, 10.8f}};

  std::cout << "Before quantization:\n";
  for (const auto &val : arr)
    std::cout << val << " ";

  std::cout << "\n\n";

  std::array<int8_t, 7> quant = quantize_int8(arr);

  std::cout << "After quantization (Int8 values cast to float):\n";
  for (const auto &val : quant)
    std::cout << static_cast<int>(val) << " ";

  std::cout << "\n\n";

  std::cout << "Dequantizing.." << "\n\n";

  std::array<float, 7> dequantized{dequantize_int8(quant)};

  std::cout << "After Dequantizaton: \n";
  for (const auto &val : dequantized)
    std::cout << val << " ";

  std::cout << "\n\n";

  return 0;
}
