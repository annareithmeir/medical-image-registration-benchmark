/* dense displacement field from sparse keypoint correspondences */

void TPSproduct(float* flow_low,double* param,int* keypoints,int num_pts,int m,int n,int o,int step1){
    int szp=num_pts+4;
    int P4[4]={1,1,1,1};
    //m1,n1,o1 are lowres dimensions
    int m1=m/step1;
    int n1=n/step1;
    int o1=o/step1;
    int sz1=m1*n1*o1;

    //convert all multiplative factors into floats
    //to enable vectorisation with SSE (fourfold speed-up)
    
    //xyz coordinates of keypoints
    float* key=new float[num_pts*3];
    float* param_f=new float[(num_pts+4)*3];
    for(int i=0;i<(num_pts+4)*3;i++){
        param_f[i]=param[i];
    }

    for(int a=0;a<num_pts;a++){
        int ind1=keypoints[a];
        int zk=ind1/(m*n);
        int xk=(ind1-zk*m*n)/m;
        int yk=ind1-zk*m*n-xk*m;
        key[a]=yk;
        key[a+num_pts]=xk;
        key[a+2*num_pts]=zk;
    }
    
    
    for(int k=0;k<o1;k++){
        for(int j=0;j<n1;j++){
            for(int i=0;i<m1;i++){
                //sparse regular grid of interpolation control points
                int x=j*step1+(step1-1)/2; int y=i*step1+(step1-1)/2; int z=k*step1+(step1-1)/2;
                float xf=x; float yf=y; float zf=z;

                //for(int i=0;i<locations;i++){
                //  int x=eval[i+locations]; int y=eval[i]; int z=eval[i+2*locations];
                P4[1]=y; P4[2]=x; P4[3]=z;
                
                float objx=0.0f; float objy=0.0f; float objz=0.0f;
                for(int a=0;a<num_pts;a++){
                    float dist=sqrt((xf-key[a+num_pts])*(xf-key[a+num_pts])+
                                    (yf-key[a])*(yf-key[a])+(zf-key[a+2*num_pts])*(zf-key[a+2*num_pts]));
                    objx+=dist*param_f[a+szp]; objy+=dist*param_f[a]; objz+=dist*param_f[a+2*szp];
                }
                for(int a=0;a<4;a++){
                    objx+=P4[a]*param_f[a+num_pts+szp]; objy+=P4[a]*param_f[a+num_pts]; objz+=P4[a]*param_f[a+num_pts+szp*2];
                }
                flow_low[i+j*m1+k*m1*n1]=objx-x;
                flow_low[i+j*m1+k*m1*n1+sz1]=objy-y;
                flow_low[i+j*m1+k*m1*n1+2*sz1]=objz-z;
            }
        }
    }
    
    delete key; delete param_f;
}

void interp3(float* interp,float* input,float* x1,float* y1,float* z1,int m,int n,int o,int m2,int n2,int o2,bool flag){
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                int x=floor(x1[i+j*m+k*m*n]); int y=floor(y1[i+j*m+k*m*n]);  int z=floor(z1[i+j*m+k*m*n]);
                float dx=x1[i+j*m+k*m*n]-x; float dy=y1[i+j*m+k*m*n]-y; float dz=z1[i+j*m+k*m*n]-z;
                
                if(flag){
                    x+=j; y+=i; z+=k;
                }
                interp[i+j*m+k*m*n]=(1.0-dx)*(1.0-dy)*(1.0-dz)*input[min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                (1.0-dx)*dy*(1.0-dz)*input[min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                dx*(1.0-dy)*(1.0-dz)*input[min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                (1.0-dx)*(1.0-dy)*dz*input[min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                dx*dy*(1.0-dz)*input[min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                (1.0-dx)*dy*dz*input[min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                dx*(1.0-dy)*dz*input[min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                dx*dy*dz*input[min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2];
            }
        }
    }
}


void interp3_0(float* interp,short* input,float* x1,float* y1,float* z1,int m,int n,int o,int m2,int n2,int o2,bool flag){
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                int x=floor(x1[i+j*m+k*m*n]); int y=floor(y1[i+j*m+k*m*n]);  int z=floor(z1[i+j*m+k*m*n]);
                float dx=x1[i+j*m+k*m*n]-x; float dy=y1[i+j*m+k*m*n]-y; float dz=z1[i+j*m+k*m*n]-z;
                
                if(flag){
                    x+=j; y+=i; z+=k;
                }
                if(x>=0&&x<n2-1&&y>=0&&y<m2-1&&z>=0&&z<o2-1){
                    interp[i+j*m+k*m*n]=(1.0-dx)*(1.0-dy)*(1.0-dz)*(float)input[y+x*m2+z*m2*n2]+
                    (1.0-dx)*dy*(1.0-dz)*(float)input[y+1+x*m2+z*m2*n2]+
                    dx*(1.0-dy)*(1.0-dz)*(float)input[y+(x+1)*m2+z*m2*n2]+
                    (1.0-dx)*(1.0-dy)*dz*(float)input[y+x*m2+(z+1)*m2*n2]+
                    dx*dy*(1.0-dz)*(float)input[y+1+(x+1)*m2+z*m2*n2]+
                    (1.0-dx)*dy*dz*(float)input[y+1+x*m2+(z+1)*m2*n2]+
                    dx*(1.0-dy)*dz*(float)input[y+(x+1)*m2+(z+1)*m2*n2]+
                    dx*dy*dz*(float)input[y+1+(x+1)*m2+(z+1)*m2*n2];
                }
                else{
                    interp[i+j*m+k*m*n]=0.0f;
                }
                
            }
        }
    }
}


void upsampleDeformations2(float* flow_high,float* flow_low,int m,int n,int o,int m2,int n2,int o2){
    
    
    float scale_m=(float)m/(float)m2;
    float scale_n=(float)n/(float)n2;
    float scale_o=(float)o/(float)o2;
    
    float* x1=new float[m*n*o];
    float* y1=new float[m*n*o];
    float* z1=new float[m*n*o];
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                x1[i+j*m+k*m*n]=j/scale_n;
                y1[i+j*m+k*m*n]=i/scale_m;
                z1[i+j*m+k*m*n]=k/scale_o;
            }
        }
    }
    
    int sz2=m2*n2*o2;
    int sz=m*n*o;
    
    interp3(flow_high,flow_low,x1,y1,z1,m,n,o,m2,n2,o2,false);
    interp3(flow_high+sz,flow_low+sz2,x1,y1,z1,m,n,o,m2,n2,o2,false);
    interp3(flow_high+2*sz,flow_low+sz2*2,x1,y1,z1,m,n,o,m2,n2,o2,false);
    
    delete x1;
    delete y1;
    delete z1;
    
}



void tpsDenseField(float* flow,float* subDisplacements,vector<int> keypoints,int m,int n,int o){

    int num_pts=keypoints.size();
    double* matrixL=new double[(num_pts+4)*(num_pts+4)];
    double* displacedPoints=new double[(num_pts+4)*3];
    for(int i=0;i<(num_pts+4)*(num_pts+4);i++){
        matrixL[i]=0.0;
    }
    for(int i=0;i<(num_pts+4)*3;i++){
        displacedPoints[i]=0.0;
    }

    for(int i=0;i<num_pts;i++){
        int ind1=keypoints[i];
        int z=ind1/(m*n);
        int x=(ind1-z*m*n)/m;
        int y=ind1-z*m*n-x*m;
        for(int j=0;j<num_pts;j++){
            int ind2=keypoints[j];
            int z2=ind2/(m*n);
            int x2=(ind2-z2*m*n)/m;
            int y2=ind2-z2*m*n-x2*m;
            matrixL[i+j*(num_pts+4)]=sqrt(max((double)((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2)),1e-20));
        }
        displacedPoints[i]=(double)y+subDisplacements[i*3];
        displacedPoints[i+(num_pts+4)]=(double)x+subDisplacements[1+i*3];
        displacedPoints[i+2*(num_pts+4)]=(double)z+subDisplacements[2+i*3];
        
        matrixL[i+(num_pts)*(num_pts+4)]=1.0;
        matrixL[i+(num_pts+1)*(num_pts+4)]=y;
        matrixL[i+(num_pts+2)*(num_pts+4)]=x;
        matrixL[i+(num_pts+3)*(num_pts+4)]=z;
        matrixL[num_pts+i*(num_pts+4)]=1.0;
        matrixL[num_pts+1+i*(num_pts+4)]=y;
        matrixL[num_pts+2+i*(num_pts+4)]=x;
        matrixL[num_pts+3+i*(num_pts+4)]=z;
    }
    
    //using Eigen-matrices for solving the linear system
    MatrixXd L(num_pts+4,num_pts+4);
    MatrixXd dP(num_pts+4,3);
    //copy values
    copy(matrixL,matrixL+(num_pts+4)*(num_pts+4),L.data());
    copy(displacedPoints,displacedPoints+(num_pts+4)*3,dP.data());
    
    MatrixXd tps=L.householderQr().solve(dP);

    double* tpsParameters=new double[(num_pts+4)*3];
    copy(tps.data(),tps.data()+(num_pts+4)*3,tpsParameters);
    
    //slightly lower resolution of dense TPS for speed
    int step1=3;
    int m1=m/step1;  int n1=n/step1; int o1=o/step1;
    int sz1=m1*n1*o1;
    float* flow_low=new float[sz1*3];
    timeval time1,time2;

    TPSproduct(flow_low,tpsParameters,keypoints.data(),num_pts,m,n,o,step1);
    

    upsampleDeformations2(flow,flow_low,m,n,o,m1,n1,o1);


    delete flow_low;
    delete matrixL; delete displacedPoints;
    //cout<<tps<<"\n";

    
}
