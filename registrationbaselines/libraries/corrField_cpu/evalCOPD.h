void homogMult(float* ptsout,float* mat44,float* ptsin,int len){
    for(int i=0;i<len;i++){
        float pt4[4]={ptsin[i],ptsin[i+len],ptsin[i+2*len],1};
        for(int l=0;l<3;l++){
            ptsout[i+l*len]=inner_product(mat44+l*4,mat44+l*4+4,pt4,0.0f);
        }
    }
}


float copdDistance(string folder,float* dists,float* ux,float* vx,float* wx,int m,int n,int o,int num){
    
    float* pts1a=new float[900];
    float* pts2a=new float[900];
    
    char* lmfile1=new char[200];
    sprintf(lmfile1,"%s/copd%d_300_iBH_xyz_r1.txt",folder.c_str(),num);
    
    fstream myfile1(lmfile1,std::ios_base::in);
    float pt1i; int count1=0;
    float* pts1i=new float[900];
    while (myfile1>>pt1i){
        int row=count1/3;
        int col=count1%3;
        pts1i[row+col*300]=pt1i-1.0f;
        count1++;
    }
    char* lmfile2=new char[200];
    sprintf(lmfile2,"%s/copd%d_300_eBH_xyz_r1.txt",folder.c_str(),num);
    
    fstream myfile2(lmfile2,std::ios_base::in);
    float pt2i; int count2=0;
    float* pts2i=new float[900];
    while (myfile2>>pt2i){
        int row=count2/3;
        int col=count2%3;
        pts2i[row+col*300]=pt2i-1.0f;
        count2++;
    }
    //printf("count1: %d, count2: %d, pts1i[0]: %2.2f\n",count1,count2,pts1i[0]);
    if(count1<300|count2<300){
        printf("Error in landmark file, only %d+%d locations founds\n",count1,count2);
    }
    
    float voxax[10]={0.625,0.645,0.652,0.590,0.647,0.633,0.625,0.586,0.664,0.742};
    float cropall[120]={55,103,1,454,444,121,69,156,1,440,449,112,38,103,3,479,436,98,44,120,5,472,440,95,
        33,102,3,472,420,120,46,107,4,466,420,108,34,98,3,469,408,116,55,143,3,438,413,90,
        44,73,9,459,412,125,63,121,11,440,413,105,52,117,5,455,387,107,66,151,8,439,402,98,
        46,115,6,470,403,112,56,127,3,466,404,101,39,107,6,473,419,115,56,125,6,456,414,97,
        33,103,5,469,410,110,51,113,6,458,411,93,35,95,7,467,400,112,57,104,13,457,401,101};
    
    float vox1=voxax[num-1];
    float* crop1=cropall+(num-1)*12;
    float* crop2=cropall+(num-1)*12+6;
    
    float voxel1=1.0f;
    float ST1[16]={voxel1/vox1,0,0,crop1[0]-1.0f,0,voxel1/vox1,0,crop1[1]-1.0f,0,0,voxel1/2.5f,crop1[2]-1.0f,0,0,0,1.0f};
    
    float newsize[3]={(float)round((double)(crop1[3]-crop1[0]+1.0)*vox1/voxel1),(float)round((double)(crop1[4]-crop1[1]+1.0)*vox1/voxel1),(float)round((double)(crop1[5]-crop1[2]+1.0)*2.5/voxel1)};
    
    float vox2[3]={newsize[0]/(crop2[3]-crop2[0]+1.0f),newsize[1]/(crop2[4]-crop2[1]+1.0f),newsize[2]/(crop2[5]-crop2[2]+1.0f)};
    
    float ST2[16]={1.0f/vox2[0],0,0,crop2[0]-1,0,1.0f/vox2[1],0,crop2[1]-1,0,0,1.0f/vox2[2],crop2[2]-1,0,0,0,1};
    
    float invA1[16]={vox1/voxel1,0,0,-(crop1[0]-1.0f)*vox1/voxel1,0,vox1/voxel1,0,-(crop1[1]-1.0f)*vox1/voxel1,0,0,2.5f/voxel1,-(crop1[2]-1.0f)*2.5f/voxel1,0,0,0,1.0f};
    
    float invA2[16]={vox2[0],0,0,-(crop2[0]-1)*vox2[0],0,vox2[1],0,-(crop2[1]-1)*vox2[1],0,0,vox2[2],-(crop2[2]-1)*vox2[2],0,0,0,1};
    

    /*
     printf("vox2: %4.3f, %4.3f, %4.3f\n",vox2[0],vox2[1],vox2[2]);
     for(int i=0;i<4;i++){
     printf("invA1: %4.3f, %4.3f, %4.3f, %4.3f \n",invA1[i],invA1[i+4],invA1[i+8],invA1[i+12]);
     }
     printf("newsize: %4.1f, %4.1f, %4.1f\n",newsize[0],newsize[1],newsize[2]);
     printf("crop1: %4.2f, %4.2f, %4.2f; %4.2f, %4.2f, %4.2f\n",crop1[0],crop1[1],crop1[2],crop1[3],crop1[4],crop1[5]);
     
     printf("\n");
     for(int i=0;i<4;i++){
     printf("invA2: %4.3f, %4.3f, %4.3f, %4.3f \n",invA2[i],invA2[i+4],invA2[i+8],invA2[i+12]);
     }
     printf("\n");
     
     for(int i=0;i<4;i++){
     printf("ST2: %4.3f, %4.3f, %4.3f, %4.3f \n",ST2[i],ST2[i+4],ST2[i+8],ST2[i+12]);
     }*/
    
    
    homogMult(pts1a,invA1,pts1i,300);
    homogMult(pts2a,invA2,pts2i,300);
    
    float vox123[3]={vox1,vox1,2.5f};
    float pt1t[3];
    float* subdisp=new float[900];
    interp3(subdisp,vx,pts1a+300,pts1a,pts1a+600,300,1,1,m,n,o,false);
    interp3(subdisp+300,ux,pts1a+300,pts1a,pts1a+600,300,1,1,m,n,o,false);
    interp3(subdisp+600,wx,pts1a+300,pts1a,pts1a+600,300,1,1,m,n,o,false);
    
    for(int i=0;i<300;i++){
        
        //int ind=min(max((int)round(pts1a[i]),0),m-1)+min(max((int)round(pts1a[i+300]),0),n-1)*m+min(max((int)round(pts1a[i+600]),0),o-1)*m*n;
        float pt1est[4]={subdisp[i]+pts1a[i],subdisp[i+300]+pts1a[i+300],subdisp[i+600]+pts1a[i+600],1};
        //float pt1est[4]={vx[ind]+pts1a[i],ux[ind]+pts1a[i+300],wx[ind]+pts1a[i+600],1};
        for(int l=0;l<3;l++){
            pt1t[l]=round(inner_product(ST2+l*4,ST2+l*4+4,pt1est,0.0f));
        }
        float dist1=0.0f;
        for(int l=0;l<3;l++){
            dist1+=pow((pt1t[l]-pts2i[i+l*300])*vox123[l],2);
        }
        dists[i]=sqrt(dist1);
        
        
    }
    return accumulate(dists,dists+300,0.0f)/300.f;
}


void warpAffine(float* warped,short* input,float* X,int m,int n,int o,int m_orig,int n_orig,int o_orig){
    int m2=m_orig; int n2=n_orig; int o2=o_orig;
    float min_val=*min_element(input,input+m2*n2*o2);
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                //assumes translation is the last row of affine matrix
                float y1=(float)i*X[0]+(float)j*X[1]+(float)k*X[2]+(float)X[3];
                float x1=(float)i*X[4]+(float)j*X[5]+(float)k*X[6]+(float)X[7];
                float z1=(float)i*X[8]+(float)j*X[9]+(float)k*X[10]+(float)X[11];
                int x=floor(x1); int y=floor(y1);  int z=floor(z1);
                float dx=x1-x; float dy=y1-y; float dz=z1-z;
                
                if(y<0|y>=m2-1|x<0|x>=n2-1|z<0|z>=o2-1){
                    warped[i+j*m+k*m*n]=min_val;
                }
                else{
                    warped[i+j*m+k*m*n]=(1.0-dx)*(1.0-dy)*(1.0-dz)*(float)input
                    [min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                    (1.0-dx)*dy*(1.0-dz)*(float)input
                    [min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                    dx*(1.0-dy)*(1.0-dz)*(float)input
                    [min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                    (1.0-dx)*(1.0-dy)*dz*(float)input
                    [min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                    dx*dy*(1.0-dz)*(float)input
                    [min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                    (1.0-dx)*dy*dz*(float)input
                    [min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                    dx*(1.0-dy)*dz*(float)input
                    [min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                    dx*dy*dz*(float)input
                    [min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2];
                }
            }
        }
    }
    
    
}

Image copdAffine(string folder,int num,char inhexh){
    
    char* volfile1=new char[200];
    sprintf(volfile1,"%s/copd%d_%cBHCT.img",folder.c_str(),num,inhexh);
    cout<<"reading binary volume from: "<<volfile1<<"\n";
    vector<short> rawscan=readFile<short>(volfile1);
    int sz1=rawscan.size();
    int m1=512; int n1=512; int o1=sz1/512/512;
    printf("size (%d) of raw scan is %d x %d x %d\n",sz1,m1,n1,o1);
    
    float voxax[10]={0.625,0.645,0.652,0.590,0.647,0.633,0.625,0.586,0.664,0.742};
    float cropall[120]={55,103,1,454,444,121,69,156,1,440,449,112,38,103,3,479,436,98,44,120,5,472,440,95,
        33,102,3,472,420,120,46,107,4,466,420,108,34,98,3,469,408,116,55,143,3,438,413,90,
        44,73,9,459,412,125,63,121,11,440,413,105,52,117,5,455,387,107,66,151,8,439,402,98,
        46,115,6,470,403,112,56,127,3,466,404,101,39,107,6,473,419,115,56,125,6,456,414,97,
        33,103,5,469,410,110,51,113,6,458,411,93,35,95,7,467,400,112,57,104,13,457,401,101};
    
    float vox1=voxax[num-1];
    float* crop1=cropall+(num-1)*12;
    float* crop2=cropall+(num-1)*12+6;
    
    float voxel1=1.0f;
    float ST1[16]={voxel1/vox1,0,0,crop1[0]-1.0f,0,voxel1/vox1,0,crop1[1]-1.0f,0,0,voxel1/2.5f,crop1[2]-1.0f,0,0,0,1.0f};
    
    float newsize[3]={(float)round((double)(crop1[3]-crop1[0]+1.0)*vox1/voxel1),(float)round((double)(crop1[4]-crop1[1]+1.0)*vox1/voxel1),(float)round((double)(crop1[5]-crop1[2]+1.0)*2.5/voxel1)};
    
    float vox2[3]={newsize[0]/(crop2[3]-crop2[0]+1.0f),newsize[1]/(crop2[4]-crop2[1]+1.0f),newsize[2]/(crop2[5]-crop2[2]+1.0f)};
    
    float ST2[16]={1.0f/vox2[0],0,0,crop2[0]-1,0,1.0f/vox2[1],0,crop2[1]-1,0,0,1.0f/vox2[2],crop2[2]-1,0,0,0,1};
    
    int m=(int)newsize[0]; int n=(int)newsize[1]; int o=(int)newsize[2];
    printf("new size of (cropped) affine scan is %d x %d x %d\n",m,n,o);
    int sz=m*n*o;
    
    
    
    Image rawfloat;
    rawfloat.values=new float[sz];
    rawfloat.m=m; rawfloat.n=n; rawfloat.o=o;
    if(inhexh=='i'){
        warpAffine(rawfloat.values,rawscan.data(),ST1,m,n,o,m1,n1,o1);
    }
    else if(inhexh=='e'){
        warpAffine(rawfloat.values,rawscan.data(),ST2,m,n,o,m1,n1,o1);
        
    }
    else{
        printf("requires letter to be either 'e' for exhale or 'i' for inhale\n");
    }
    
    for(int i=0;i<sz;i++){
        rawfloat.values[i]=max(rawfloat.values[i]-1024.0f,-1024.0f);
    }
   
    
    return rawfloat;
    /*

    
    fstream myfile1(lmfile1,std::ios_base::in);
    float pt1i; int count1=0;
    float* pts1i=new float[900];
    while (myfile1>>pt1i){
        int row=count1/3;
        int col=count1%3;
        pts1i[row+col*300]=pt1i-1.0f;
        count1++;
    }
    char* lmfile2=new char[200];
    sprintf(lmfile2,"/Users/mattias/Documents/COPD/copd%d_300_eBH_xyz_r1.txt",num);
    
    fstream myfile2(lmfile2,std::ios_base::in);
    float pt2i; int count2=0;
    float* pts2i=new float[900];
    while (myfile2>>pt2i){
        int row=count2/3;
        int col=count2%3;
        pts2i[row+col*300]=pt2i-1.0f;
        count2++;
    }
    //printf("count1: %d, count2: %d, pts1i[0]: %2.2f\n",count1,count2,pts1i[0]);
    if(count1<300|count2<300){
        printf("Error in landmark file, only %d+%d locations founds\n",count1,count2);
    }
    
    
    
    for(int i=0;i<300;i++){
        
        //int ind=min(max((int)round(pts1a[i]),0),m-1)+min(max((int)round(pts1a[i+300]),0),n-1)*m+min(max((int)round(pts1a[i+600]),0),o-1)*m*n;
        float pt1est[4]={subdisp[i]+pts1a[i],subdisp[i+300]+pts1a[i+300],subdisp[i+600]+pts1a[i+600],1};
        //float pt1est[4]={vx[ind]+pts1a[i],ux[ind]+pts1a[i+300],wx[ind]+pts1a[i+600],1};
        for(int l=0;l<3;l++){
            pt1t[l]=round(inner_product(ST2+l*4,ST2+l*4+4,pt1est,0.0f));
        }
        float dist1=0.0f;
        for(int l=0;l<3;l++){
            dist1+=pow((pt1t[l]-pts2i[i+l*300])*vox123[l],2);
        }
        dists[i]=sqrt(dist1);
        
        
    }
    return accumulate(dists,dists+300,0.0f)/300.f;
     
     */
}



