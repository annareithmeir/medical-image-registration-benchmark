#include "zlib.h"
/*
 Reader/Writer for Nifti images (or volumes)
 Mattias P. Heinrich
 Universitaet Luebeck, 2015
 */


Image readImage(string filestr){
    //convert input string into char* array
    char* filename=new char[filestr.size()+1];
    copy(filestr.begin(),filestr.end(),filename);
    filename[filestr.size()]='\0';
    int m,n,o;
    Image img;
    
    vector<float> voxel={1,1,1};
    //opens file for binary-input
    gzFile file=gzopen(filename,"rb");
    if(not(!file)){
        //read nifti header
        char* header=new char[352];
        gzread(file,header,352);
        //read image dimensions
        short* dim;
        dim=reinterpret_cast<short*>(header+40);
        m=(int)dim[1]; n=(int)dim[2]; o=(int)dim[3];
        img.m=m; img.n=n; img.o=o;
        //read voxel-size
        float* vox;
        vox=reinterpret_cast<float*>(header+76);
        voxel[0]=(float)vox[1]; voxel[1]=(float)vox[2]; voxel[2]=(float)vox[3];
        //pre-allocate vector size
        img.values=new float[m*n*o];
        short* datatype; //read datatype (and bitpix)
        datatype=reinterpret_cast<short*>(header+70);
        short bitpix=datatype[1];
        //raw empty datapointers of all 'types'
        unsigned char* ucharptr; float* floatptr; short* shortptr; int* intptr; double* doubleptr;
        //read input data values depending on datatype and convert to float
        //binary are read character by character, reinterpret_cast converts them
        //copy them into float* array afterwards
        switch(datatype[0]){
            case 2:
                ucharptr=new unsigned char[m*n*o];
                gzread(file,ucharptr,m*n*o*sizeof(unsigned char));
                ucharptr=reinterpret_cast<unsigned char*>(ucharptr);
                for(int i=0;i<m*n*o;i++){
                    img.values[i]=ucharptr[i];
                }
                break;
            case 4:
                shortptr=new short[m*n*o];
                gzread(file,shortptr,m*n*o*sizeof(short));
                shortptr=reinterpret_cast<short*>(shortptr);
                for(int i=0;i<m*n*o;i++){
                    img.values[i]=shortptr[i];
                }
                break;
            case 8:
                intptr=new int[m*n*o];
                gzread(file,intptr,m*n*o*sizeof(int));
                intptr=reinterpret_cast<int*>(intptr);
                for(int i=0;i<m*n*o;i++){
                    img.values[i]=intptr[i];
                }
                break;
            case 16:
                floatptr=new float[m*n*o];
                gzread(file,floatptr,m*n*o*sizeof(float));
                floatptr=reinterpret_cast<float*>(floatptr);
                for(int i=0;i<m*n*o;i++){
                    img.values[i]=floatptr[i];
                }
                break;
            case 64:
                doubleptr=new double[m*n*o];
                gzread(file,doubleptr,m*n*o*sizeof(double));
                doubleptr=reinterpret_cast<double*>(doubleptr);
                for(int i=0;i<m*n*o;i++){
                    img.values[i]=doubleptr[i];
                }
                break;
            default:
                printf("Datatype %d not supported. Exiting.\n",datatype[0]);
                exit(1);
        }
        printf("Read image with dimensions %dx%dx%d of datatype %d\n",m,n,o,datatype[0]);
    }
    else{
        printf("File error. Did not find file. Exiting.\n");
        exit(1);
    }
    gzclose(file);
    return img;
}


void createNiftiHeader(char* header,int m,int n,int o,vector<float> voxelsize,int databit){
    for(int i=0;i<352;i++){
        header[i]=0; //initialise everything with zeros
    }
    //size of header
    int size_of_hdr=348;
    memcpy(header+0,&size_of_hdr,sizeof(int));
    //image dimensions (first value is number of dim's)
    vector<short> dim(8,1);
    dim[0]=o>1?3:2; dim[1]=m; dim[2]=n; dim[3]=o;
    //datatype and bitpix
    vector<short> datatype={16,32}; //float-datatype 32 bit
    if(databit==2){
        datatype[0]=4; datatype[1]=16; //short-integer 16 bit
    }
    if(databit==3){
        datatype[0]=128; datatype[1]=24; //24-bit RGB
        dim[3]=1;
    }
    memcpy(header+40,dim.data(),8*sizeof(short));

    memcpy(header+70,datatype.data(),2*sizeof(short));
    //pixel dimensions in units (mm)
    vector<float> pixdim(8,1); pixdim[0]=0;
    //copy the three values of voxelsize
    pixdim[1]=voxelsize[0]; pixdim[2]=voxelsize[1]; pixdim[3]=voxelsize[2];
    memcpy(header+76,pixdim.data(),8*sizeof(float));
    //vox-offset = header size
    float vox_offset=352;
    memcpy(header+108,&vox_offset,sizeof(float));
    //sform-code (use affine not quaternion fields)
    short sform_code=1;
    memcpy(header+254,&sform_code,sizeof(short));
    //srow_x,y,z affine transform
    vector<float> srow_xyz={1,0,0,1, 0,-1,0,1, 0,0,-1,1}; //unity matrix
    memcpy(header+280,srow_xyz.data(),12*sizeof(float));
    //magic string
    vector<char> magic={'n','+','1','\0'};
    memcpy(header+344,magic.data(),4*sizeof(char));

}

void writeImage(string filestr,Image img,vector<float> voxelsize){
    int m=img.m; int n=img.n; int o=img.o;
    
    //create default nifti header with given dimensions
    char* header=new char[352];
    int datatype=1;
    createNiftiHeader(header,img.m,img.n,img.o,voxelsize,datatype);
    
    
    //convert filename string into char* array
    char* filename=new char[filestr.size()+1];
    copy(filestr.begin(),filestr.end(),filename);
    filename[filestr.size()]='\0';
    
    printf("Writing image with dimensions %dx%dx%d\n",m,n,o);

    //opens file for binary-output
    gzFile file=gzopen(filename,"wb");
    if(not(!file)){
		gzwrite(file,header,352);
        //copy casted values
		gzwrite(file,reinterpret_cast<char*>(img.values),m*n*o*sizeof(float));
		gzclose(file);
		cout<<"File "<<filename<<" written.\n";
	}
    else{
        printf("File error. Could not write file.\n");
    }
}

