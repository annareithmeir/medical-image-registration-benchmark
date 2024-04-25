/* Incremental diffusion regularisation of parametrised transformation
 using (globally optimal) belief-propagation on minimum spanning tree.
 Fast distance transform (see fastDT2.h) uses squared differences.
 Similarity cost for each node and label has to be given as input.
*/

void regularisation(float* marginals,int* selected,float* costall,int* ordered,int* parents,float* edgeweights,float* disp0,int hw,int sparse,int num_vertices){

	timeval time1,time2;
	
	int sz=num_vertices;
	
	//dense displacement space
	int len=hw*2+1;
	int len2=len*len*len;
	
	//int *selected=new int[sz];
	float *cost1=new float[len2];
	float *vals=new float[len2];
	int *inds=new int[len2];
	gettimeofday(&time1, NULL);
    
    float* message=new float[sz*len2];
    
	for(int i=0;i<sz*len2;i++){
        marginals[i]=costall[i];
        message[i]=0.0;
	}
	
	//float alpha1=(float)step1/(alpha*quant);
	//inverse of regularisation weighting alpha
	//includes (division by) distance of control points (step1) and quantisation

	int xs1,ys1,zs1,xx,yy,zz,xx2,yy2,zz2;

	for(int i=0;i<len2;i++){
		cost1[i]=0;
	}
	
	//calculate mst-cost
	for(int i=(sz-1);i>0;i--){ //do for each control point
	
		int ochild=ordered[i];
		int oparent=parents[ordered[i]];
        float edgew=edgeweights[ordered[i]];
        float edgew1=1.0f/edgew;

		
		for(int l=0;l<len2;l++){
			cost1[l]=marginals[ochild*len2+l]*edgew;
		}
		//important for INCREMENTAL regularisation (offset delta)
        float dx1=(disp0[oparent+sz]-disp0[ochild+sz])/(float)sparse;
        float dy1=(disp0[oparent]-disp0[ochild])/(float)sparse;
        float dz1=(disp0[oparent+2*sz]-disp0[ochild+2*sz])/(float)sparse;
        //fast distance transform see fastDT2.h
		dt3x(cost1,inds,len,dx1,dy1,dz1);
				
		//add mincost to parent node
		for(int l=0;l<len2;l++){
            message[ochild*len2+l]=cost1[l]*edgew1;
			marginals[oparent*len2+l]+=cost1[l]*edgew1;

		}
		
	}
    //backwards pass mst-cost
	for(int i=1;i<sz;i++){ //other direction
		int ochild=ordered[i];
		int oparent=parents[ordered[i]];
        float edgew=edgeweights[ordered[i]];
        float edgew1=1.0f/edgew;

		for(int l=0;l<len2;l++){
			cost1[l]=(marginals[oparent*len2+l]-message[ochild*len2+l]+message[oparent*len2+l])*edgew;
		}
        //important for INCREMENTAL regularisation (offset delta)
        //MUST BE OTHER-WAY AROUND FOR BACKWARD PASS
        float dx1=(disp0[ochild+sz]-disp0[oparent+sz])/(float)sparse;
        float dy1=(disp0[ochild]-disp0[oparent])/(float)sparse;
        float dz1=(disp0[ochild+2*sz]-disp0[oparent+2*sz])/(float)sparse;

        dt3x(cost1,inds,len,dx1,dy1,dz1);
		for(int l=0;l<len2;l++){
            message[ochild*len2+l]=cost1[l]*edgew1;
		}
		
	}
	
    for(int i=0;i<sz*len2;i++){
		marginals[i]+=message[i];
	}
    
    //select displacements
    for(int i=0;i<sz;i++){
        selected[i]=min_element(marginals+i*len2,marginals+(i+1)*len2)-(marginals+i*len2);
        
    }


    delete message;
	delete cost1;
	delete vals;
	delete inds;

	
}

void subMinimum(float* subDisplacements,float* marginals,float* flow,vector<int> keypoints,int hw,int sparse,int sz){

    int num_fixed=keypoints.size();
    int len2=pow(hw*2+1,3);
    
    float subdisp3[3];
    //select displacements
    for(int i=0;i<num_fixed;i++){
        
        int selected=min_element(marginals+i*len2,marginals+(i+1)*len2)-(marginals+i*len2);
        subMin(marginals+i*len2,subdisp3,hw,selected);
        //TO-DO fix inconsistency of x-y axes definition
        subDisplacements[i*3]=subdisp3[0]*(float)sparse+flow[keypoints[i]+sz];
        subDisplacements[1+i*3]=subdisp3[1]*(float)sparse+flow[keypoints[i]];
        subDisplacements[2+i*3]=subdisp3[2]*(float)sparse+flow[keypoints[i]+2*sz];
        
    }
}
//#include "mstSort.h"
#include "minimumSpanTree.h"

void regularisationMST(float* marginals,int* optimalIndices,float* similarityVolume,vector<int> keypoints,float* flow0,int hw,int sparse,Image scan_fixed){
    
    
    int m=scan_fixed.m; int n=scan_fixed.n; int o=scan_fixed.o;
    
    int num_fixed=keypoints.size();
    
    float* euclDistMatrix=new float[num_fixed*num_fixed];
    
    int* orderedList=new int[num_fixed];
    int* parentsList=new int[num_fixed];
    float* edgeWeight=new float[num_fixed];
    
//        Graph3d(parentsList,orderedList,edgeWeight,keypoints.data(),m,n,o,num_fixed);
    
    edgeGraph(euclDistMatrix,keypoints,scan_fixed);
    primsMST(orderedList,parentsList,edgeWeight,euclDistMatrix,num_fixed);

    regularisation(marginals,optimalIndices,similarityVolume,orderedList,parentsList,edgeWeight,flow0,hw,sparse,num_fixed);

}
