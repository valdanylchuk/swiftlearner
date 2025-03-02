A quick port of BackpropClassifier to C++.
Just wanted to evaluate the difference in speed and syntax.
Learning MNIST with the same parameters is about 7x faster, no manual optimization, mainly just -ffast-math.
The syntax seems okay.

Prerequisites: cmake, gtest.

Usage:
mkdir build
cd build
cmake ..
make
test/swiftlearner_test
