/*
 Copyright (c) 2015, Mattias P. Heinrich
 Contact: heinrich(at)imi.uni-luebeck.de
 http://www.mpheinrich.de
 
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
 */

/*
 If you use this implementation or parts of it please cite:
 
 "Estimating Large Lung Motion in COPD Patients by Symmetric Regularised Correspondence Fields"
 by Mattias P. Heinrich, Heinz Handels and Ivor J.A. Simpson
 Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015, LNCS, Springer (2015)
 
 AND
 
 "Edge- and Detail-Preserving Sparse Image Representations for Deformable Registration of Chest MRI and CT Volumes"
 by Mattias P. Heinrich, Mark Jenkinson, Bartlomiej W. Papiez, Fergus V. Gleeson, Sir Michael Brady, and Julia A. Schnabel
 Information Processing in Medical Imaging (IPMI) 2013. LNCS 7917, pp. 463-474, Springer (2013)
 
 Requires an installed zlib library (see http://zlib.net ) and the eigenlibrary (see http://eigen.tuxfamily.org )
 Tested with g++ on Mac OS X and Linux Ubuntu, compile with:
 
 g++ corrField.cpp -O3 -lpthread -std=c++11 -lz -msse4.2 -o corrField
 
 replace msse4.2 by your current SSE version if needed
 for Windows you might need MinGW or CygWin
 
 */

#include <sstream>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <functional>
#include <thread>
#include <sys/time.h>
#include <x86intrin.h>
#include <pthread.h>

using namespace std;


struct Image{
    float* values; uint64_t* descriptor;
    int m; int n; int o;
};

struct Options{
    char* file_fixed;
    char* file_mask;
    char* file_moving;
    char* file_output;
    int* maxlen_;
    int* radius_;
    int* hw_;
    int* sparse_;
    int* skip_;
    float alpha;
    float sigma;
};


template <typename TypeFO>
void writeOutput(TypeFO* data,string filestr,int length){
    //opens file for binary-output
    char* filename=new char[filestr.size()+1];
    copy(filestr.begin(),filestr.end(),filename);
    filename[filestr.size()]='\0';
    ofstream ofs1(filename,ios::out|ios::binary);
    ofs1.write(reinterpret_cast<char*>(data),length*sizeof(TypeFO));
    ofs1.close();
}

template <typename TypeF>
vector<TypeF> readFile(char filename[]){
    //opens file for binary-input
    ifstream file(filename,ios::binary|ios::ate);
    vector<TypeF> invalues;
    if(file.is_open()){
        int length=file.tellg()/(sizeof(TypeF));
        file.seekg(0,file.beg); //go to beginning
        char* charptr=new char[length*sizeof(TypeF)];
        file.read((char*)charptr,length*sizeof(TypeF));
        TypeF* floatptr=reinterpret_cast<TypeF*>(charptr);
        for(int i=0;i<length;i++){
            invalues.push_back(floatptr[i]);
        }
        delete charptr;
    }
    file.close();
    return invalues;
}

#include <Eigen/Dense>
#include <Eigen/QR>
using namespace Eigen;


#include "niftiInOutgz.h"
#include "featureExtraction.h"
#include "MIND-SSC.h"
#include "similarityCost.h"
#include "fastdt2.h"
#include "inferenceMarginals.h"
#include "thinPlateSpline.h"

Options parseArguments(int ,char* []);


int main(int argc,char* argv[]){
    
    timeval time1,time2;
    
    Options opts=parseArguments(argc,argv);
    
    Image scan_fixed=readImage(opts.file_fixed);
    Image mask_fixed=readImage(opts.file_mask);

    Image scan_moving=readImage(opts.file_moving);

    int m=scan_fixed.m; int n=scan_fixed.n; int o=scan_fixed.o;
    int sz=scan_fixed.m*scan_fixed.n*scan_fixed.o;

    printf("=== Estimating large motion correspondence field ===\n");
    
    vector<float> voxelsize(3,1.0f);
    
    //settings for keypoint detection
    //float sigma=1.4; //sigma of Gaussian in Foerstner operator
    //int maxlen_[]={6,3}; //{8,4} local nonmax suppression
    float thresh=1e-25; //keypoint distinctiveness threshold
    
    //parameters for first similarity search
    //int radius_[]={3,2}; int skip_[]={2,1}; //{3,2} integration patch with radius, but skip every other voxel
    //int hw_[]={10,8}; //{8,4} search half width of label space
    //int sparse_[]={2,1}; //{3,2} steps of two voxels
    //float alpha=1.0f; //similarity weight

    float sigma=opts.sigma; float alpha=opts.alpha;
    
    float* flow=new float[sz*3];
    for(int i=0;i<sz*3;i++){
        flow[i]=0.0f;
    }
    Image warped; warped.values=new float[sz]; warped.m=scan_fixed.m; warped.n=scan_fixed.n; warped.o=scan_fixed.o;
    for(int i=0;i<sz;i++){
        warped.values[i]=scan_moving.values[i];
    }
    int maxlevel=1;
    if(opts.hw_[1]==0){
        maxlevel=0;
        printf("without refinement stage.\n");
    }
    
    //Image distinctive=readImage(argv[4]);//
    Image distinctive=foerstnerOperator(scan_fixed,sigma);
    
    
    timeval time1a,time2a;
    gettimeofday(&time1a, NULL);

    for(int refinement=0;refinement<=maxlevel;refinement++){
        
        int hw=opts.hw_[refinement];
        int sparse=opts.sparse_[refinement];
        int radius=opts.radius_[refinement];
        int maxlen=opts.maxlen_[refinement];
        int skip=opts.skip_[refinement];
        
        int label_size=pow(hw*2+1,3);
        
        if(refinement==1){
            printf("refinement stage: ");
        }
        printf("alpha=%4.2f, search-radius:%d, non-max:%d,\nquant:%d, patch-radius:%d, skip:%d, sigma:%4.2f\n",alpha,hw,maxlen,sparse,radius,skip,sigma);
    
        
        gettimeofday(&time1, NULL);
        
        vector<int> keypoints=foerstnerKeypoints(distinctive,mask_fixed,maxlen,thresh);
        int num_fixed=keypoints.size();
        gettimeofday(&time2, NULL);
        float timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
        
        printf("%d keypoint features found in %4.2f secs.\n",num_fixed,timeP);
        float* flow0=new float[num_fixed*3];
        for(int i=0;i<num_fixed;i++){
            //TO-DO fix inconsistency of x-y axes definition, but this works
            flow0[i]=-flow[keypoints[i]+sz];
            flow0[i+1*num_fixed]=-flow[keypoints[i]];
            flow0[i+2*num_fixed]=-flow[keypoints[i]+2*sz];
        }
        
        gettimeofday(&time1, NULL);
        int delta=1; float sigma_ssc=0.8;

        Image ssc_fixed,ssc_moving;
        thread t1([&]{ ssc_fixed=quantisedMIND(scan_fixed,delta,sigma_ssc); });
        thread t2([&]{ ssc_moving=quantisedMIND(warped,delta,sigma_ssc); });
        t1.join(); t2.join();
        
        gettimeofday(&time2, NULL);
        timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
        
        printf("MIND-SSC descriptors extracted in %4.2f secs.\n",timeP);
        
        float* similarityVolume=new float[num_fixed*label_size];
        float* subDisplacements=new float[num_fixed*3];
        int* optimalIndices=new int[num_fixed];
        float* marginals=new float[num_fixed*label_size];
        float* marginalsBack=new float[num_fixed*label_size];
        
        
        //similarity cost calculations use two-threads
        gettimeofday(&time1, NULL);
        
        int num_half=num_fixed/2;
        vector<int> key1(keypoints.begin(),keypoints.begin()+num_half);
        vector<int> key2(keypoints.begin()+num_half,keypoints.end());
        thread ts1(similarityCost, similarityVolume,ssc_fixed,ssc_moving,key1,radius,hw,sparse,skip,alpha);
        thread ts2(similarityCost, similarityVolume+num_half*label_size,ssc_fixed,ssc_moving,key2,radius,hw,sparse,skip,alpha);
        ts1.join(); ts2.join();
        
        gettimeofday(&time2, NULL);
        timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
        printf("Similarity of SSC-based correspondences %4.2f secs.\n",timeP);
    
        
        gettimeofday(&time1, NULL);
        
        regularisationMST(marginals,optimalIndices,similarityVolume,keypoints,flow0,hw,sparse,scan_fixed);
        
        subMinimum(subDisplacements,marginals,flow,keypoints,hw,sparse,sz);

        gettimeofday(&time2, NULL);
        timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
        printf("Probabilistic MST-inference %4.2f secs.\n",timeP);
        
        //backward matching
        gettimeofday(&time1, NULL);
        vector<int> corrPoints=correspondencePoints(keypoints,optimalIndices,hw,sparse,m,n,o);
        vector<int> corr1(corrPoints.begin(),corrPoints.begin()+num_half);
        vector<int> corr2(corrPoints.begin()+num_half,corrPoints.end());
        for(int i=0;i<num_fixed;i++){
            //TO-DO fix inconsistency of x-y axes definition, but this works
            flow0[i]=flow[corrPoints[i]+sz];
            flow0[i+1*num_fixed]=flow[corrPoints[i]];
            flow0[i+2*num_fixed]=flow[corrPoints[i]+2*sz];
        }
        thread ts1b(similarityCost, similarityVolume,ssc_moving,ssc_fixed,corr1,radius,hw,sparse,skip,alpha);
        thread ts2b(similarityCost, similarityVolume+num_half*label_size,ssc_moving,ssc_fixed,corr2,radius,hw,sparse,skip,alpha);
        ts1b.join(); ts2b.join();
        regularisationMST(marginalsBack,optimalIndices,similarityVolume,corrPoints,flow0,hw,sparse,scan_fixed);
        //symmetric marginals
        for(int i=0;i<num_fixed;i++){
            for(int l=0;l<label_size;l++){
                marginals[l+i*label_size]+=marginalsBack[(label_size-l-1)+i*label_size];
            }
        }
        
        subMinimum(subDisplacements,marginals,flow,keypoints,hw,sparse,sz);
        
        gettimeofday(&time2, NULL);
        timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
        printf("Backward matching symmetric marginals %4.2f secs.\n",timeP);
        
        if(refinement==maxlevel){
            //correspondence field is a 6 x num_pts array where each column indicates
            //[y_fixed,x_fixed,z_fixed, y_moving,x_moving,z_moving] coordinates
            
            float* corrField=new float[num_fixed*6];
            for(int i=0;i<num_fixed;i++){
                int ind1=keypoints[i];
                int z=ind1/(m*n);
                int x=(ind1-z*m*n)/m;
                int y=ind1-z*m*n-x*m;
                corrField[0+i*6]=y; corrField[1+i*6]=x; corrField[2+i*6]=z;
                for(int j=0;j<3;j++){
                    corrField[j+3+i*6]=subDisplacements[j+i*3]+corrField[j+i*6];
                }
            }

            writeOutput<float>(corrField,opts.file_output,num_fixed*6);
        }
        
        if(refinement==0){
            
            gettimeofday(&time1, NULL);
            
            tpsDenseField(flow,subDisplacements,keypoints,scan_fixed.m,scan_fixed.n,scan_fixed.o);
            interp3(warped.values,scan_moving.values,flow,flow+sz,flow+sz*2,m,n,o,m,n,o,true);
            
            gettimeofday(&time2, NULL);
            timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            
            printf("Time for dense TPS interpolation %4.2f secs.\n",timeP);
           
        }
        
        
    }
    
    gettimeofday(&time2a, NULL);
    float timeA=time2a.tv_sec+time2a.tv_usec/1e6-(time1a.tv_sec+time1a.tv_usec/1e6);
    
    printf("Total computation time %4.2f secs.\n",timeA);

    
    return 0;
}

Options parseArguments(int argc,char* argv[]){
    
    Options opts;
    
    if(argc<5||argv[1][1]=='h'){
        cout<<"=============================================================\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./corrField -F fixed.nii.gz -M moving.nii.gz -m fixed_mask.nii.gz -O output.dat\n";
        cout<<"optional parameters:\n";
        cout<<" -a <regularisation parameter alpha> (default 1.0)\n";
        cout<<" -L <maximum search radius - each level> (default 16x8)\n";
        cout<<" -N <cube-length of non-maximum suppression> (default 6x3)\n";
        cout<<" -Q <quantisation of search step size> (default 2x1)\n";
        cout<<" -R <size (radius) of patch for similarity> (default 3x2)\n";
        cout<<" -S <skipping of patch for similarity> (default 2x1)\n";
        cout<<" -s <sigma of Foerstner operator> (default 1.4)\n";
        cout<<"=============================================================\n";
        exit(-1);
    }
    
    
    typedef pair<char,int> val;
    map<char,int> argin;
    argin.insert(val('F',0));
    argin.insert(val('M',1));
    argin.insert(val('m',2));
    argin.insert(val('O',3));
    argin.insert(val('a',4));
    argin.insert(val('L',5));
    argin.insert(val('N',6));
    argin.insert(val('Q',7));
    argin.insert(val('R',8));
    argin.insert(val('S',9));
    argin.insert(val('s',10));
    
    // parsing the input
    int requiredArgs=0;
    opts.file_fixed=new char[200];
    opts.file_mask=new char[200];
    opts.file_moving=new char[200];
    opts.file_output=new char[200];
    
    opts.alpha=1.0;
    opts.sigma=1.4;
    
    opts.maxlen_=new int[2];
    opts.radius_=new int[2];
    opts.hw_=new int[2];
    opts.sparse_=new int[2];
    opts.skip_=new int[2];
    
    opts.maxlen_[0]=6; opts.maxlen_[1]=3;
    opts.radius_[0]=3; opts.radius_[1]=2;
    opts.hw_[0]=16; opts.hw_[1]=8;
    opts.sparse_[0]=2; opts.sparse_[1]=1;
    opts.skip_[0]=2; opts.skip_[1]=1;
    
    char levelstr[]="%dx%d";
    
    int num;
   
    
    for(int k=1;k<argc;k++){
        if(argv[k][0]=='-'){
            if(argin.find(argv[k][1])==argin.end()){
                cout<<"Invalid option: "<<argv[k]<<" use -h for help\n";
            }
            switch(argin[argv[k][1]]){
                case 0: //F
                    sprintf(opts.file_fixed,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 1: //M
                    sprintf(opts.file_moving,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 2: //m
                    sprintf(opts.file_mask,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 3: //O
                    sprintf(opts.file_output,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 4: //a
                    opts.alpha=atof(argv[k+1]);
                    break;
                case 5: //L
                    num=sscanf(argv[k+1],levelstr,&opts.hw_[0],&opts.hw_[1]);
                    if(num!=2){
                        cout<<"Invalid length of L should be 2, e.g. 5x4\n";
                    }
                    break;
                case 6: //N
                    num=sscanf(argv[k+1],levelstr,&opts.maxlen_[0],&opts.maxlen_[1]);
                    if(num!=2){
                        cout<<"Invalid length of N should be 2, e.g. 5x4\n";
                    }
                    break;
                case 7: //Q
                    num=sscanf(argv[k+1],levelstr,&opts.sparse_[0],&opts.sparse_[1]);
                    if(num!=2){
                        cout<<"Invalid length of Q should be 2, e.g. 5x4\n";
                    }
                    break;
                case 8: //R
                    num=sscanf(argv[k+1],levelstr,&opts.radius_[0],&opts.radius_[1]);
                    if(num!=2){
                        cout<<"Invalid length of R should be 2, e.g. 5x4\n";
                    }
                    break;
                case 9: //S
                    num=sscanf(argv[k+1],levelstr,&opts.skip_[0],&opts.skip_[1]);
                    if(num!=2){
                        cout<<"Invalid length of S should be 2, e.g. 5x4\n";
                    }
                    break;
             
                default:
                    cout<<"Invalid option: "<<argv[k]<<" use -h for help\n";
                    break;
            }
        }
    }
    if(requiredArgs!=4){
        cout<<"Missing argmuents (only "<<requiredArgs<<" provided) use -h for help.\n";
        exit(-1);
    }
    
    
    return opts;
}
