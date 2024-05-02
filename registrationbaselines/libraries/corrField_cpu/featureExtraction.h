/* keypoint extraction using Foerstner operator in 3D  */


//squared distance transform according to Felzenszwalb & Huttenlocher

void dt1sqi(float *val,int len,int k,int* v,float* f,float* z){
    float INF=1e15f;
    
    int j=0;
    z[0]=-INF;
    z[1]=INF;
    v[0]=0;
    for(int q=1;q<len;q++){
        float s=((val[q*k]+q*q)-(val[v[j]*k]+v[j]*v[j]))/(2.0*(q-v[j]));
        while(s<=z[j]){
            j--;
            s=((val[q*k]+q*q)-(val[v[j]*k]+v[j]*v[j]))/(2.0*(q-v[j]));
        }
        j++;
        v[j]=q;
        z[j]=s;
        z[j+1]=INF;
    }
    j=0;
    for(int q=0;q<len;q++){
        f[q]=val[q*k];
    }
    for(int q=0;q<len;q++){
        while(z[j+1]<q){
            j++;
        }
        val[q*k]=(q-v[j])*(q-v[j])+f[v[j]];
    }
    
}

void dt3x(float* r,int m,int n,int o){
    
    
    int* v; //slightly faster if not intitialised in each loop
    float* z;
    float* f;
    
    //horizontal pass
    v=new int[n];
    z=new float[n+1];
    f=new float[n];
    for(int k=0;k<o;k++){
        for(int i=0;i<m;i++){
            dt1sqi(r+i+k*m*n,n,m,v,f,z);
        }
    }
    
    //vertical pass
    v=new int[m];
    z=new float[m+1];
    f=new float[m];
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            dt1sqi(r+j*m+k*m*n,m,1,v,f,z);
        }
    }
    
    //depth pass
    v=new int[o];
    z=new float[o+1];
    f=new float[o];
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            dt1sqi(r+i+j*m,o,m*n,v,f,z);
        }
    }
    
    
    delete f;
    delete v;
    delete z;
    
}


void filter1(float* imagein,float* imageout,int m,int n,int o,float* filter,int length,int dim){
	int i,j,k,f;
	int i1,j1,k1;
	int hw=(length-1)/2;
	
	for(i=0;i<(m*n*o);i++){
		imageout[i]=0.0;
	}
	
    float* tempm=new float[m];
    //replicate-padding
    
    if(dim==1){

        for(k=0;k<o;k++){
            for(j=0;j<n;j++){
                for(i=0;i<m;i++){
                    float temp=0.0;
                    for(f=0;f<length;f++){
                        temp+=filter[f]*imagein[max(min(i+f-hw,m-1),0)+j*m+k*m*n];
                    }
                    imageout[i+j*m+k*m*n]=temp;
                }
            }
        }
    }
    
    if(dim==2){
        for(k=0;k<o;k++){
            for(j=0;j<n;j++){
                for(i=0;i<m;i++){
                    tempm[i]=0.0;
                    for(f=0;f<length;f++){
                        tempm[i]+=filter[f]*imagein[i+max(min(j+f-hw,n-1),0)*m+k*m*n];
                    }
                }
                for(i=0;i<m;i++){
                    imageout[i+j*m+k*m*n]=tempm[i];
                }
            }
        }
    }
    
    if(dim==3){
        for(k=0;k<o;k++){
            for(j=0;j<n;j++){
                for(i=0;i<m;i++){
                    tempm[i]=0.0;
                    for(f=0;f<length;f++){
                        tempm[i]+=filter[f]*imagein[i+j*m+max(min(k+f-hw,o-1),0)*m*n];
                    }
                }
                for(i=0;i<m;i++){
                    imageout[i+j*m+k*m*n]=tempm[i];
                }
            }
        }
    }

    delete tempm;

}

void maxfilt(float* filt,float* img,int m,int n,int o,int len){

    
    float* temp=new float[m*n*o];
    
    int hw=(len-1)/2;
    float* array=new float[len];
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                for(int f=-hw;f<(len-hw);f++){
                    array[f+hw]=img[min(max(f+i,0),m-1)+j*m+k*m*n];
                }
                filt[i+j*m+k*m*n]=*max_element(array,array+len);
            }
        }
    }
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                for(int f=-hw;f<(len-hw);f++){
                    array[f+hw]=filt[i+min(max(f+j,0),n-1)*m+k*m*n];
                }
                temp[i+j*m+k*m*n]=*max_element(array,array+len);
            }
        }
    }
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                for(int f=-hw;f<(len-hw);f++){
                    array[f+hw]=temp[i+j*m+min(max(f+k,0),o-1)*m*n];
                }
                filt[i+j*m+k*m*n]=*max_element(array,array+len);
            }
        }
    }
    
    delete temp;
    
}


void volfilter(float* imagein,int m,int n,int o,float sigma){
    
    int length=ceil(sigma*3.0/2.0)*2+1;
    int hw=(length-1)/2;
    int i,j,f;
    float hsum=0;
    float* filter=new float[length];
    for(i=0;i<length;i++){
        filter[i]=exp(-pow((i-hw),2)/(2*pow(sigma,2)));
        hsum=hsum+filter[i];
    }
    for(i=0;i<length;i++){
        filter[i]=filter[i]/hsum;
    }
    float* image1=new float[m*n*o];
    for(i=0;i<m*n*o;i++){
        image1[i]=imagein[i];
    }
    filter1(image1,imagein,m,n,o,filter,length,1);
    filter1(imagein,image1,m,n,o,filter,length,2);
    filter1(image1,imagein,m,n,o,filter,length,3);
    
    delete image1;
    delete filter;
    
}

float det(float* a,float* b,float* c){
    return a[0]*b[1]*c[2]+b[0]*c[1]*a[2]+c[0]*a[1]*b[2]-c[0]*b[1]*a[2]-b[0]*a[1]*c[2]-a[0]*c[1]*b[2];
}

void cramers(float* A,float* b,float* x){
    float detA1=1.0/det(A,A+3,A+6);
    x[0]=detA1*det(b,A+3,A+6);
    x[1]=detA1*det(A,b,A+6);
    x[2]=detA1*det(A,A+3,b);
}


Image foerstnerOperator(Image img,float sigma){
    
    
    int m=img.m; int n=img.n; int o=img.o;
    int sz=m*n*o;
    
    float* grad=new float[sz*3];
    float filter[5]={1.0/12.0,-8.0/12.0,0.0,8.0/12.0,-1.0/12.0};
    filter1(img.values,grad,m,n,o,filter,5,1);
    filter1(img.values,grad+sz,m,n,o,filter,5,2);
    filter1(img.values,grad+2*sz,m,n,o,filter,5,3);
    
    timeval time1,time2;

    //build structure tensor
    float* tensor=new float[sz*9];
    for(int i=0;i<3;i++){
        for(int j=i;j<3;j++){
            transform(grad+i*sz,grad+(i+1)*sz,grad+j*sz,tensor+(i+j*3)*sz,multiplies<float>());
            volfilter(tensor+(i+j*3)*sz,m,n,o,sigma); //inplace smoothing
        }
    }
    
    float b[]={1,0,0,0,1,0,0,0,1}; //3x3 unity matrix
    float inv[]={0,0,0,0,0,0,0,0,0};
    float A[]={0,0,0,0,0,0,0,0,0};
    
    Image feature;
    feature.values=new float[sz];
    feature.m=m; feature.n=n; feature.o=o;
    
    for(int i=0;i<sz;i++){
        for(int l=0;l<3;l++){
            for(int k=l;k<3;k++){
                A[l+k*3]=tensor[i+(l+k*3)*sz];
                A[k+l*3]=tensor[i+(l+k*3)*sz];
            }
        }
        
        for(int l=0;l<3;l++){
            cramers(A,b+l*3,inv+l*3);
        }
        feature.values[i]=1.0f/(inv[0]+inv[4]+inv[8]);
    }
    
    
    delete grad; delete tensor;
    return feature;
    
}
vector<int> foerstnerKeypoints(Image distinctive,Image mask,int maxlen,float thresh){

    int m=distinctive.m; int n=distinctive.n; int o=distinctive.o;
    int sz=m*n*o;

    float* maxfeat=new float[sz];
    
    float* distMask=new float[sz];
    for(int i=0;i<sz;i++){
        distMask[i]=(mask.values[i]<0.5)?0.0f:1e10f;
        
    }
    dt3x(distMask,m,n,o); //3d euclidean distance transform
    
    for(int i=0;i<sz;i++){
        distMask[i]=sqrt(distMask[i]);
    }
    
    maxfilt(maxfeat,distinctive.values,m,n,o,maxlen);
    vector<int> indices;
    for(int i=0;i<sz;i++){
        if(distMask[i]>1.0&&maxfeat[i]==distinctive.values[i]&&distinctive.values[i]>thresh){
            indices.push_back(i);
        }
    
    }
    
    delete maxfeat; //delete feature;
    
    return indices;
    // feature;
    
}

vector<int> correspondencePoints(vector<int> keypoints,int* selected,int hw,int sparse,int m,int n,int o){
    int len=hw*2+1;
    int searchsz=pow(len,3);
    int* xs=new int[searchsz];
    int* ys=new int[searchsz];
    int* zs=new int[searchsz];
    
    for(int i=0;i<len;i++){
        for(int j=0;j<len;j++){
            for(int k=0;k<len;k++){
                xs[i+j*len+k*len*len]=(j-hw)*sparse;
                ys[i+j*len+k*len*len]=(i-hw)*sparse;
                zs[i+j*len+k*len*len]=(k-hw)*sparse;
            }
        }
    }

    vector<int> corrPoints;
    for(int i=0;i<keypoints.size();i++){
        int ind1=keypoints[i];
        int z=ind1/(m*n);
        int x=(ind1-z*m*n)/m;
        int y=ind1-z*m*n-x*m;
        //int x2=x+round(subDisp[i*3+1]);
        //int y2=y+round(subDisp[i*3]);
        //int z2=z+round(subDisp[i*3+2]);
      
        int x2=min(max(x+xs[selected[i]],0),n-1);
        int y2=min(max(y+ys[selected[i]],0),m-1);
        int z2=min(max(z+zs[selected[i]],0),o-1);
       
        
        corrPoints.push_back(y2+x2*m+z2*m*n);
        
    }
    return corrPoints;
}



