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
    char* file_moving;
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
    
    timeval time1,time2;
    
    Options opts=parseArguments(argc,argv);
    
    Image scan_fixed=copdAffine(opts.folder,opts.casenum,'i');
    vector<float> voxelsize(3,1.0f);
    writeImage(opts.file_fixed,scan_fixed,voxelsize);
    
    Image scan_moving=copdAffine(opts.folder,opts.casenum,'e');
    writeImage(opts.file_moving,scan_moving,voxelsize);

    return 0;
}


Options parseArguments(int argc,char* argv[]){
    
    Options opts;
    
    if(argc<5||argv[1][1]=='h'){
        cout<<"=============================================================================\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./processCOPD -f copd_img_folder -C casenum -F fixed.nii.gz -M moving.nii.gz\n";
        cout<<"=============================================================================\n";
        exit(-1);
    }
    
    
    typedef pair<char,int> val;
    map<char,int> argin;
    argin.insert(val('F',0));
    argin.insert(val('M',1));
    argin.insert(val('f',2));
    argin.insert(val('C',3));
    
    
    // parsing the input
    int requiredArgs=0;
    opts.file_fixed=new char[200];
    opts.file_moving=new char[200];
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
                case 1: //M
                    sprintf(opts.file_moving,"%s",argv[k+1]);
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
