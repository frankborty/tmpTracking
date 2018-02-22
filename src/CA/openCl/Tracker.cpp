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

#include <unistd.h>

#include "ITSReconstruction/CA/Definitions.h"
#include <ITSReconstruction/CA/Tracklet.h>
#include <ITSReconstruction/CA/Cell.h>
#include <ITSReconstruction/CA/Constants.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <ITSReconstruction/CA/Tracklet.h>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Vector.h"

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
			oclCommandqueues[iLayer].finish();

			trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.mGPUContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					Constants::ITS::TrackletsPerRoad*sizeof(int)
			);
			totalTrackletsFound+=trackletsFound[iLayer];
#ifdef PRINT_TRACKLET_CNT
			outfile<<"Layer:"<<iLayer<<" = "<<trackletsFound[iLayer]<<"\n";
#endif
		}
#ifdef PRINT_TRACKLET_CNT
	  outfile<<"Total:"<<totalTrackletsFound<<"\n\n";
	  outfile.close();
#endif
	}catch (...) {
		std::cout<<"Exception during compute cells phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}



}



}
}
}
