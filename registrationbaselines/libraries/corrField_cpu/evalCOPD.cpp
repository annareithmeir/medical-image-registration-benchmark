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
    char* folder;
    char* file_corr;
    int casenum;
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

#include "evalCOPD.h"

Options parseArguments(int ,char* []);


int main(int argc,char* argv[]){
    
    Options opts=parseArguments(argc,argv);
    
    vector<float> corrField=readFile<float>(opts.file_corr);
    
    Image scan_fixed=readImage(opts.file_fixed);
    
    int m=scan_fixed.m; int n=scan_fixed.n; int o=scan_fixed.o;
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
    timeval time1,time2;
    gettimeofday(&time1, NULL);
    
    tpsDenseField(flow,subDisplacements,keypoints,m,n,o);
    
    gettimeofday(&time2, NULL);
    float timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    
    printf("Time for dense TPS interpolation %4.2f secs.\n",timeP);
    
    float* dists=new float[300];
    float tre=copdDistance(opts.folder,dists,flow,flow+sz,flow+sz*2,m,n,o,opts.casenum);
    nth_element(dists,dists+300,dists+150);
    float median=dists[150]; float maxerr=*max_element(dists,dists+300);
    printf("Average TRE for case#%d: %4.3f mm, median: %4.3f mm, maximum: %4.3f mm\n",opts.casenum,tre,median,maxerr);
    
    return 0;
}


Options parseArguments(int argc,char* argv[]){
    
    Options opts;
    
    if(argc<5||argv[1][1]=='h'){
        cout<<"=============================================================================\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./evalCOPD -f landmark_folder -C casenum -F fixed.nii.gz -O output.dat\n";
        cout<<"=============================================================================\n";
        exit(-1);
    }
    
    
    typedef pair<char,int> val;
    map<char,int> argin;
    argin.insert(val('F',0));
    argin.insert(val('O',1));
    argin.insert(val('f',2));
    argin.insert(val('C',3));
    
    
    // parsing the input
    int requiredArgs=0;
    opts.file_fixed=new char[200];
    opts.file_corr=new char[200];
    opts.folder=new char[200];
    
    opts.casenum=1;
    
    
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
                case 1: //O
                    sprintf(opts.file_corr,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 2: //f
                    sprintf(opts.folder,"%s",argv[k+1]);
                    requiredArgs++;
                    break;
                case 3: //C
                    opts.casenum=atoi(argv[k+1]);
                    requiredArgs++;
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
