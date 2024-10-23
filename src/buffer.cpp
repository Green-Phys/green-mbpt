#include<iostream>
#include"buffer.hpp"
#include <unistd.h>


int buffer::nelem_heuristics(double ratio) const{
    //figure out how much memory is available on the machine
    unsigned long long pages = sysconf(_SC_PHYS_PAGES);
    unsigned long long page_size = sysconf(_SC_PAGE_SIZE);
    unsigned long long total_memory=pages*page_size;
    if(verbose_) std::cout<<"total memory size: "<<total_memory/(1024.*1024.*1024.)<<" GB"<<std::endl;

    //figure out how many elements we could fit total
    unsigned long long total_elements=total_memory/element_size();

    //modify by proportion of memory, round and return
    int proposed_nelem=(int)(total_elements*ratio);


    if(proposed_nelem >= number_of_keys_) proposed_nelem=number_of_keys_;
    if(verbose_ && (proposed_nelem <100)) std::cerr<<"WARNING: ONLY "<<proposed_nelem<<" (<100) buffer elements fit to memory."<<std::endl; 

    return proposed_nelem;
}
void buffer::setup_mpi_shmem(){
  int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Info info; MPI_Info_create(&info);
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm_);
  MPI_Comm_size(shmem_comm_,&shmem_size_);
  MPI_Comm_rank(shmem_comm_,&shmem_rank_);

  MPI_Barrier(MPI_COMM_WORLD);
  int global_size; MPI_Comm_size(MPI_COMM_WORLD, &global_size);
  for(int i=0; i<global_size;++i){
    MPI_Barrier(MPI_COMM_WORLD);
    if(i==global_rank)
      std::cout<<"global rank "<<i<<" is local shmem rank: "<<shmem_rank_<<" with shmem size: "<<shmem_size_<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
