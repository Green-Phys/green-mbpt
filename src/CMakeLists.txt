add_library(mpi_read_lock 
buffer.cpp
chunk_reader.cpp
)
target_link_libraries(mpi_read_lock PUBLIC Eigen3::Eigen)
target_link_libraries(mpi_read_lock PUBLIC MPI::MPI_CXX)
target_link_libraries(mpi_read_lock PUBLIC hdf5::hdf5 /opt/local/lib/libsz.a /opt/local/lib/libz.a)


