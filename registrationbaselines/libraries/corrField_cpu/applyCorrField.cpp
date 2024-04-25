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
#include <string>

using namespace std;


struct Image{
    float* values; uint64_t* descriptor;
    int m; int n; int o;
};

struct Options{
    std::string file_moving;
    std::string file_corr;
    std::string file_warped;
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
vector<TypeF> readFile(std::string filename){
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
    
    Options opts=parseArguments(argc,argv);

    vector<float> corrField=readFile<float>(opts.file_corr);
    
    Image scan_moving=readImage(opts.file_moving);
    
    int m=scan_moving.m; int n=scan_moving.n; int o=scan_moving.o;
    int sz=m*n*o;
    
    int num_fixed=corrField.size()/6;
    vector<int> keypoints;
    float* subDisplacements=new float[3*num_fixed];
    
    for(int i=0;i<num_fixed;i++){
        int y=corrField[0+i*6]; int x=corrField[1+i*6]; int z=corrField[2+i*6];
        int ind1=y+x*m+z*m*n;
        keypoints.push_back(ind1);
        for(int j=0;j<3;j++){
            subDisplacements[j+i*3]=corrField[j+3+i*6]-corrField[j+i*6];
        }
    }
    
    float* flow=new float[sz*3];
    Image warped; warped.values=new float[sz]; warped.m=m; warped.n=n; warped.o=o;
    timeval time1,time2;
    gettimeofday(&time1, NULL);

    tpsDenseField(flow,subDisplacements,keypoints,m,n,o);
    interp3(warped.values,scan_moving.values,flow,flow+sz,flow+sz*2,m,n,o,m,n,o,true);
    
    gettimeofday(&time2, NULL);
    float timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    
    printf("Time for dense TPS interpolation %4.2f secs.\n",timeP);
    vector<float> voxelsize(3,1.0f);


    writeImage(opts.file_warped,warped,voxelsize);
    return 0;
}


Options parseArguments(int argc,char* argv[]){
    
    Options opts;
    
    if(argc<4||argv[1][1]=='h'){
        cout<<"=============================================================================\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./applyCorrField -M moving.nii.gz -O output.dat -W warped.nii.gz\n";
        cout<<"=============================================================================\n";
        exit(-1);
    }
    
    
    typedef pair<char,int> val;
    map<char,int> argin;
    argin.insert(val('M',0));
    argin.insert(val('O',1));
    argin.insert(val('W',2));
    
    
    // parsing the input
    int requiredArgs=0;

    for(int k=1;k<argc;k++){
        if(argv[k][0]=='-'){
            if(argin.find(argv[k][1])==argin.end()){
                cout<<"Invalid option: "<<argv[k]<<" use -h for help\n";
            }
            switch(argin[argv[k][1]]){
                case 0: //M
                    opts.file_moving = argv[k + 1];
                    requiredArgs++;
                    break;
                case 1: //O
                    opts.file_corr = argv[k + 1];
                    requiredArgs++;
                    break;
                case 2: //W
                    opts.file_warped = argv[k + 1];
                    requiredArgs++;
                    break;
                
                
                default:
                    cout<<"Invalid option: "<<argv[k]<<" use -h for help\n";
                    break;
            }
        }
    }
    if(requiredArgs!=3){
        cout<<"Missing argmuents (only "<<requiredArgs<<" provided) use -h for help.\n";
        exit(-1);
    }
    
    
    return opts;
}
