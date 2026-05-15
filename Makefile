debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build build -j12

test:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_NUMPY_TESTS=ON -DBUILD_CXX_TESTS=ON
	cmake --build build -j12

release:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build build -j12

benchmark:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_BENCHMARKS=ON
	cmake --build build -j12
