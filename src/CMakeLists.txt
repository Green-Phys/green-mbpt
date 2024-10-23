add_library(mpi_read_lock buffer.cpp)
target_link_libraries(mpi_read_lock PUBLIC Eigen3::Eigen)
target_link_libraries(mpi_read_lock PUBLIC MPI::MPI_CXX)

