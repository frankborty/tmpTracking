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
//#define PRINT_TRACKLET_CNT_STDOUT

//#define PRINT_CELL_CNT
//#define PRINT_CELL_CNT_STDOUT

#include <unistd.h>

#include "ITSReconstruction/CA/Definitions.h"
#include <ITSReconstruction/CA/Tracklet.h>
#include <ITSReconstruction/CA/Cell.h>
#include <ITSReconstruction/CA/Constants.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <ITSReconstruction/CA/Tracklet.h>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Vector.h"
#include "ITSReconstruction/CA/openCl/Utils.h"
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
  computeLayerCells(primaryVertexContext, layerIndex, cellsVector);
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

	int iClustersNum;
	int *firstLayerLookUpTable;
	int* trackletsFound;
	int totalTrackletsFound=0;
	cl::Buffer bTrackletLookUpTable;
	cl::Kernel oclCountTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclCountTrackletKernel;
	cl::Kernel oclComputeTrackletKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeTrackletKernel;
	int workgroupSize=5*32;	//tmp value
	time_t tx,ty;

	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

	//boost_compute
		compute::device device = compute::device(oclDevice(),true);
		compute::context ctx(oclContext(),true);
		compute::command_queue queue(ctx, device);
	///


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
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
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
		free(firstLayerLookUpTable);
#ifdef PRINT_TRACKLET_CNT_STDOUT
		for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++)
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[i].finish();
		trackletsFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[Constants::ITS::TrackletsPerRoad-1].enqueueMapBuffer(
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
		GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();

		if(iLayer==0){
			int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
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

			int* lookUpFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
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
			for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++)
				totalTrackletsFound+=trackletsFound[iLayer];

#endif
		}
	}
	ty=clock();
	time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
	std::cout<< "\t>Total scan time = "<<time<<" ms" << std::endl;

	trackletsFound = (int *) oclCommandQueue.enqueueMapBuffer(
			primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
			CL_TRUE, // block
			CL_MAP_READ,
			0,
			Constants::ITS::TrackletsPerRoad*sizeof(int)
	);
	for(int i=0;i<Constants::ITS::TrackletsPerRoad;i++){
		primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[i]=trackletsFound[i];
	}

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



		GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
			oclComputeTrackletKernel,
			cl::NullRange,
			cl::NDRange(pseudoClusterNumber),
			cl::NDRange(workgroupSize));

#ifdef PRINT_TRACKLET


		TrackletStruct* output = (TrackletStruct *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueMapBuffer(
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

template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
#ifdef PRINT_CELL_CNT
	std::ofstream outfile;
	outfile.open("/home/frank/Scrivania/cellFoundOCL.txt", std::ios_base::app);
#endif
	time_t tx,ty;
	int iTrackletsNum;
	int *firstLayerLookUpTable;
	int* trackletsFound;
	int totalCellsFound=0;
	int *cellsFound;
	cl::Buffer bCellLookUpTable;
	cl::Kernel oclCountCellKernel=GPU::Context::getInstance().getDeviceProperties().oclCountCellKernel;
	cl::Kernel oclComputeCellKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeCellKernel;
	int workgroupSize=5*32;
	try{
		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

		iTrackletsNum=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[0];

		int pseudoTracletsNumber=iTrackletsNum;
		if((pseudoTracletsNumber % workgroupSize)!=0){
			int mult=pseudoTracletsNumber/workgroupSize;
			pseudoTracletsNumber=(mult+1)*workgroupSize;
		}
		firstLayerLookUpTable = new int[pseudoTracletsNumber];

		//std::fill(firstLayerLookUpTable,firstLayerLookUpTable+pseudoTracletsNumber,-1);
		memset(firstLayerLookUpTable,-1,pseudoTracletsNumber*sizeof(int));
		bCellLookUpTable = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			pseudoTracletsNumber*sizeof(int),
			(void *) &firstLayerLookUpTable[0]);


		for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {
			std::cout<<"iLayer: "<<iLayer<<std::endl;
			oclCountCellKernel.setArg(0, primaryVertexContext.mGPUContext.bPrimaryVertex);  //0 fPrimaryVertex
			oclCountCellKernel.setArg(1, primaryVertexContext.mGPUContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclCountCellKernel.setArg(2, primaryVertexContext.mGPUContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclCountCellKernel.setArg(3, primaryVertexContext.mGPUContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclCountCellKernel.setArg(4, primaryVertexContext.mGPUContext.bTracklets[iLayer+1]); //4 nextLayerTracklets				oclCountCellKernel.setArg(5, primaryVertexContext.mGPUContext.bTracklets[iLayer+2]); //5 next2LayerTracklets
			oclCountCellKernel.setArg(5, primaryVertexContext.mGPUContext.bClusters[iLayer]);  //5 currentLayerClusters
			oclCountCellKernel.setArg(6, primaryVertexContext.mGPUContext.bClusters[iLayer+1]);//6 nextLayerClusters
			oclCountCellKernel.setArg(7, primaryVertexContext.mGPUContext.bClusters[iLayer+2]);//7 next2LayerClusters
			oclCountCellKernel.setArg(8, primaryVertexContext.mGPUContext.bTrackletsLookupTable[iLayer]);//8  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclCountCellKernel.setArg(9, bCellLookUpTable);//9iCellsPerTrackletPreviousLayer;
			else
				oclCountCellKernel.setArg(9, primaryVertexContext.mGPUContext.bCellsLookupTable[iLayer-1]);//9iCellsPerTrackletPreviousLayer
			oclCountCellKernel.setArg(10, primaryVertexContext.mGPUContext.bCellsFoundForLayer);


			int pseudoTrackletsNumber=primaryVertexContext.mGPUContext.iTrackletFoundPerLayer[iLayer];
			if((pseudoTrackletsNumber % workgroupSize)!=0){
				int mult=pseudoTrackletsNumber/workgroupSize;
				pseudoTrackletsNumber=(mult+1)*workgroupSize;
			}

			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].finish();
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[iLayer].enqueueNDRangeKernel(
				oclCountCellKernel,
				cl::NullRange,
				cl::NDRange(pseudoTrackletsNumber),
				cl::NDRange(workgroupSize));

		}

		delete []firstLayerLookUpTable;
#ifdef PRINT_CELL_CNT_STDOUT
		for(int i=0;i<Constants::ITS::CellsPerRoad;i++){
			GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[i].finish();

			cellsFound = (int *) GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[i].enqueueMapBuffer(
				primaryVertexContext.mGPUContext.bCellsFoundForLayer,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				Constants::ITS::CellsPerRoad*sizeof(int)
		);
		}
		for(int i=0;i<Constants::ITS::CellsPerRoad;i++){
			std::cout<<"Layer:"<<i<<" = "<<cellsFound[i]<<std::endl;
			totalCellsFound+=cellsFound[i];
		}
		std::cout<<"Total:"<<totalCellsFound<<std::endl;
#endif
#ifdef PRINT_CELL_CNT
	for(int i=0;i<Constants::ITS::CellsPerRoad;i++){
		outfile<<"Layer:"<<i<<" = "<<cellsFound[i]<<"\n";
	}
	outfile<<"Total:"<<totalCellsFound<<"\n\n";
	outfile.close();
#endif


	}catch(const cl::Error &err){
		std::string errString=o2::ITS::CA::GPU::Utils::OCLErr_code(err.err());
		//std::cout<< errString << std::endl;
		throw std::runtime_error { errString };
	}
	catch( const std::exception & ex ) {
	       throw std::runtime_error { ex.what() };
	}
  	catch (...) {
		std::cout<<"Exception during compute cells phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}
}




}
}
}
