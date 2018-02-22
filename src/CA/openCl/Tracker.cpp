// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
///

//#define PRINT_TRACKLET_CNT
//#define PRINT_TRACKLET_LUT
//#define PRINT_TRACKLET
#define PRINT_TRACKLET_CNT_STDOUT
#include <unistd.h>

#include "ITSReconstruction/CA/Definitions.h"
#include <ITSReconstruction/CA/Tracklet.h>
#include <ITSReconstruction/CA/Cell.h>
#include <ITSReconstruction/CA/Constants.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <ITSReconstruction/CA/Tracklet.h>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Vector.h"
#include "boost/compute.hpp"
namespace compute = boost::compute;

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{



void computeLayerTracklets(PrimaryVertexContext &primaryVertexContext, const int layerIndex,
    Vector<Tracklet>& trackletsVector)
{

}

void computeLayerCells(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell>& cellsVector)
{
}

void layerTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> trackletsVector)
{
  computeLayerTracklets(primaryVertexContext, layerIndex, trackletsVector);
}

void sortTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> tempTrackletArray)
{

}

void layerCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> cellsVector)
{
//  computeLayerCells(primaryVertexContext, layerIndex, cellsVector);
}



void sortCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> tempCellsArray)
{

}

} /// End of GPU namespace


template<>
void TrackerTraits<true>::computeLayerTracklets(CA::PrimaryVertexContext& primaryVertexContext)
{

#ifdef PRINT_TRACKLET_CNT
	std::ofstream outfile;
	outfile.open("/home/frank/Scrivania/trackFoundOCL.txt", std::ios_base::app);
#endif
#ifdef PRINT_TRACKLET_LUT
	std::ofstream outfileLookUp;
	outfileLookUp.open("/home/frank/Scrivania/trackletLookUpOCL.txt", std::ios_base::app);
#endif
#ifdef PRINT_TRACKLET
	std::ofstream outfileLookTracklet;
	outfileLookTracklet.open("/home/frank/Scrivania/trackletOCL.txt", std::ios_base::app);
#endif

	std::cout<<"computeLayerTracklets"<<std::endl;
	int iClustersNum;
	int *firstLayerLookUpTable;
	int* trackletsFound;
	int totalTrackletsFound=0;
	cl::Buffer bTrackletLookUpTable;
	cl::CommandQueue oclCommandqueues[Constants::ITS::TrackletsPerRoad];
	cl::Kernel oclCountTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclCountTrackletKernel;
	cl::Kernel oclComputeTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeTrackletKernel;





	int workgroupSize=5*32;	//tmp value

	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

	//boost_compute
		compute::device device = compute::device(oclDevice(),true);
		compute::context ctx(oclContext(),true);
		compute::command_queue queue(ctx, device);
	///

		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
			oclCommandqueues[i]=cl::CommandQueue(oclContext, oclDevice, 0);
		}

		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[0];

		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			iClustersNum=(mult+1)*workgroupSize;
		}


		firstLayerLookUpTable=(int*)malloc(iClustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,iClustersNum*sizeof(int));
		bTrackletLookUpTable = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) &firstLayerLookUpTable[0]);

		time_t tx,ty;
		tx=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
			oclCountTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
			oclCountTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
			oclCountTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
			oclCountTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
			oclCountTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
			oclCountTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
			oclCountTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bClustersSize);
			if(iLayer==0)
				oclCountTrackletKernel.setArg(7, bTrackletLookUpTable);
			else
				oclCountTrackletKernel.setArg(7, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=iClustersNum;
			if((iClustersNum % workgroupSize)!=0){
				int mult=iClustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

//			time_t tx=clock();
			oclCommandqueues[iLayer].enqueueNDRangeKernel(
				oclCountTrackletKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));
				//cl::NullRange);
/*
			trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					Constants::ITS::TrackletsPerRoad*sizeof(int)
			);
			totalTrackletsFound+=trackletsFound[iLayer];
*/
			//std::cout<<"Layer:"<<iLayer<<" = "<<trackletsFound[iLayer]<<std::endl;
		}

#ifdef PRINT_TRACKLET_CNT_STDOUT
		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++)
			oclCommandqueues[i].finish();
		trackletsFound = (int *) oclCommandqueues[Constants::ITS::TrackletsPerRoad-1].enqueueMapBuffer(
				primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				Constants::ITS::TrackletsPerRoad*sizeof(int)
		);
		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
			std::cout<<"Layer:"<<i<<" = "<<trackletsFound[i]<<std::endl;
			totalTrackletsFound+=trackletsFound[i];
		}
		std::cout<<"Total:"<<totalTrackletsFound<<std::endl;
#endif
#ifdef PRINT_TRACKLET_CNT
	outfile<<"Total:"<<totalTrackletsFound<<"\n\n";
	outfile.close();
#endif

	ty=clock();
	float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total tracklet count time = "<<time<<" ms" << std::endl;

	//scan
	tx=clock();
	for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
		oclCommandqueues[iLayer].finish();

		if(iLayer==0){
			int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
				bTrackletLookUpTable,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				iClustersNum*sizeof(int)
			);

			// create vector on the device
			compute::vector<int> device_vector(iClustersNum, ctx);

			// copy data to the device
			compute::copy(lookUpFound, lookUpFound+iClustersNum, device_vector.begin(), queue);

			// sort data on the device
			compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);
			bTrackletLookUpTable=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);

#ifdef PRINT_TRACKLET_LUT
			outfileLookUp<<"Layer "<<iLayer<<"\n";
			//std::cout<<clustersNum<<std::endl;
			for(int j=0;j<iClustersNum;j++)
				outfileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
			outfileLookUp<<"\n";

#endif
		}
		else{

			int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				iClustersNum*sizeof(int)
			);

			// create vector on the device
			compute::vector<int> device_vector(iClustersNum, ctx);

			// copy data to the device
			compute::copy(lookUpFound, lookUpFound+iClustersNum, device_vector.begin(), queue);

			// sort data on the device
			compute::exclusive_scan(device_vector.begin(),device_vector.end(),device_vector.begin(),0,queue);
			// copy data back to the host
			compute::copy(device_vector.begin(), device_vector.end(), lookUpFound, queue);

			primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				iClustersNum*sizeof(int),
				(void *) lookUpFound);


#ifdef PRINT_TRACKLET_LUT
			outfileLookUp<<"Layer "<<iLayer<<"\n";
			//std::cout<<clustersNum<<std::endl;
			for(int j=0;j<iClustersNum;j++)
				outfileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
			outfileLookUp<<"\n";
#endif
#ifdef PRINT_TRACKLET
			trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					Constants::ITS::TrackletsPerRoad*sizeof(int)
			);
			for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++)
				totalTrackletsFound+=trackletsFound[iLayer];

#endif
		}
	}
	ty=clock();
	time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total scan time = "<<time<<" ms" << std::endl;





	//calcolo le tracklet
	tx=clock();
	for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
		iClustersNum=primaryVertexContext.mGPUContext.iClusterSize[iLayer];
		oclComputeTrackletKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);
		oclComputeTrackletKernel.setArg(1, primaryVertexContext.mGPUContext.bClusters[iLayer]);
		oclComputeTrackletKernel.setArg(2, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);
		oclComputeTrackletKernel.setArg(3, primaryVertexContext.mGPUContext.bIndexTables[iLayer]);
		oclComputeTrackletKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer]);
		oclComputeTrackletKernel.setArg(5, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]);
		oclComputeTrackletKernel.setArg(6, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);
		oclComputeTrackletKernel.setArg(7, primaryVertexContext.mGPUContext.bClustersSize);
		if(iLayer==0)
			oclComputeTrackletKernel.setArg(8, bTrackletLookUpTable);
		else
			oclComputeTrackletKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer-1]);

		int pseudoClusterNumber=iClustersNum;
		if((iClustersNum % workgroupSize)!=0){
			int mult=iClustersNum/workgroupSize;
			pseudoClusterNumber=(mult+1)*workgroupSize;
		}


		oclCommandqueues[iLayer].enqueueNDRangeKernel(
			oclComputeTrackletKernel,
			cl::NullRange,
			cl::NDRange(pseudoClusterNumber),
			cl::NDRange(workgroupSize));

#ifdef PRINT_TRACKLET


		TrackletStruct* output = (TrackletStruct *) oclCommandqueues[iLayer].enqueueMapBuffer(
			primaryVertexContext.mGPUContext.bTracklets[iLayer],
			CL_TRUE, // block
			CL_MAP_READ,
			0,
			trackletsFound[iLayer] * sizeof(TrackletStruct)
		);
		outfileLookTracklet<<"Tracklets between Layer "<<iLayer<<" and "<<iLayer+1<<"\n";
		for(int i=0;i<trackletsFound[iLayer];i++)
			outfileLookTracklet<<output[i].firstClusterIndex<<"\t"<<output[i].secondClusterIndex<<"\t"<<output[i].phiCoordinate<<"\t"<<output[i].tanLambda<<"\n";
		outfileLookTracklet<<"\n";
#endif


	}
	ty=clock();
	time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total compute tracklet time = "<<time<<" ms" << std::endl;



	}catch (...) {
		std::cout<<"Exception during compute cells phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}



}



}
}
}
