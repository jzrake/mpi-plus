CXXFLAGS = -std=c++14 -Wall
CXX = mpicxx

mpi-plus: mpi-plus.cpp

clean:
	$(RM) mpi-plus
