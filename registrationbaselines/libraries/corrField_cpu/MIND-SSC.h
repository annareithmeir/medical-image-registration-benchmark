void movingfilter2(float* data,float* temp1,float* temp2,int m,int n,int o,int hw){
    
    int r_m=hw; int r_n=hw; int r_o=hw;
    float* summ=new float[m];
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            int i=-r_m;
            float sum=0.0f;
            for(;i<=r_m;i++){
                sum+=data[i+r_m+j*m+k*m*n];
            }
            for(;i+r_m<m;i++){
                sum-=data[i-r_m-1+j*m+k*m*n];
                sum+=data[i+r_m+j*m+k*m*n];
                temp1[i+j*m]=sum;
            }
            for(;i<m;i++){
                sum-=data[i-r_m-1+j*m+k*m*n];
                temp1[i+j*m]=sum;
            }
        }
        
        int j=-r_n;
        for(int i=0;i<m;i++){
            summ[i]=0.0f;
        }
        for(;j<=r_n;j++){
            for(int i=0;i<m;i++){
                summ[i]+=temp1[i+(j+r_n)*m];
            }
        }
        for(;j+r_n<n;j++){
            for(int i=0;i<m;i++){
                summ[i]-=temp1[i+(j-r_n-1)*m];
                summ[i]+=temp1[i+(j+r_n)*m];
                temp2[i+j*m+k*m*n]=summ[i];
            }
        }
        for(;j<n;j++){
            for(int i=0;i<m;i++){
                summ[i]-=temp1[i+(j-r_n-1)*m];
                temp2[i+j*m+k*m*n]=summ[i];
            }
        }
    }
    
    for(int j=0;j<n;j++){
        int k=-r_o;
        for(int i=0;i<m;i++){
            summ[i]=0.0f;
        }
        for(;k<=r_o;k++){
            for(int i=0;i<m;i++){
                summ[i]+=temp2[i+j*m+(k+r_o)*m*n];
            }
        }
        for(;k+r_o<o;k++){
            for(int i=0;i<m;i++){
                summ[i]-=temp2[i+j*m+(k-r_o-1)*m*n];
                summ[i]+=temp2[i+j*m+(k+r_o)*m*n];
                data[i+j*m+k*m*n]=summ[i];
            }
        }
        for(;k<o;k++){
            for(int i=0;i<m;i++){
                summ[i]-=temp2[i+j*m+(k-r_o-1)*m*n];
                data[i+j*m+k*m*n]=summ[i];
            }
        }
    }
}

void movingfilter(float* data,float* temp1,float* temp2,int m,int n,int o,int hw){
    
    int length=hw*2+1;
    
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            int i=-hw;
            float sum=0.0f;
            for(;i<=hw;i++){
                sum+=data[i+hw+j*m+k*m*n];
            }
            for(;i+hw<m;i++){
                sum-=data[i-hw-1+j*m+k*m*n];
                sum+=data[i+hw+j*m+k*m*n];
                temp1[i+j*m+k*m*n]=sum;
            }
            for(;i<m;i++){
                sum-=data[i-hw-1+j*m+k*m*n];
                temp1[i+j*m+k*m*n]=sum;
            }
        }
    }
    for(int k=0;k<o;k++){
        for(int i=0;i<m;i++){
            int j=-hw;
            float sum=0.0f;
            for(;j<=hw;j++){
                sum+=temp1[i+(j+hw)*m+k*m*n];
            }
            for(;j+hw<n;j++){
                sum-=temp1[i+(j-hw-1)*m+k*m*n];
                sum+=temp1[i+(j+hw)*m+k*m*n];
                temp2[i+j*m+k*m*n]=sum;
            }
            for(;j<n;j++){
                sum-=temp1[i+(j-hw-1)*m+k*m*n];
                temp2[i+j*m+k*m*n]=sum;
            }
        }
    }
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            int k=-hw;
            float sum=0.0f;
            for(;k<=hw;k++){
                sum+=temp2[i+j*m+(k+hw)*m*n];
            }
            for(;k+hw<o;k++){
                sum-=temp2[i+j*m+(k-hw-1)*m*n];
                sum+=temp2[i+j*m+(k+hw)*m*n];
                data[i+j*m+k*m*n]=sum;
            }
            for(;k<o;k++){
                sum-=temp2[i+j*m+(k-hw-1)*m*n];
                data[i+j*m+k*m*n]=sum;
            }
        }
    }
    
    
}

void imshift(float* input,float* output,int dx,int dy,int dz,int m,int n,int o){
	for(int k=0;k<o;k++){
		for(int j=0;j<n;j++){
			for(int i=0;i<m;i++){
				if(i+dy>=0&&i+dy<m&&j+dx>=0&&j+dx<n&&k+dz>=0&&k+dz<o)
					output[i+j*m+k*m*n]=input[i+dy+(j+dx)*m+(k+dz)*m*n];
				else
					output[i+j*m+k*m*n]=input[i+j*m+k*m*n];
			}
		}
	}
}

void volshift(float* input,float* output,int dx,int dy,int dz,int m,int n,int o,int d){
	for(int k=0;k<o;k++){
		for(int j=0;j<n;j++){
			for(int i=0;i<m;i++){
				if(i+dy>=0&&i+dy<m&&j+dx>=0&&j+dx<n&&k+dz>=0&&k+dz<o)
					for(int q=0;q<d;q++){
						output[i+j*m+k*m*n+q*m*n*o]=input[i+dy+(j+dx)*m+(k+dz)*m*n+q*m*n*o];
					}
				else
					for(int q=0;q<d;q++){
						output[i+j*m+k*m*n+q*m*n*o]=input[i+j*m+k*m*n+q*m*n*o];
					}
			}
		}
	}
}


//__builtin_popcountll(left[i]^right[i]); absolute hamming distances
void descriptor(float* mind,float* im1,int m,int n,int o,int qs,float sigma){
	
	//MIND with self-similarity context
	
	int dx[6]={+qs,+qs,-qs,+0,+qs,+0};
	int dy[6]={+qs,-qs,+0,-qs,+0,+qs};
	int dz[6]={0,+0,+qs,+qs,+qs,+qs};
    
	int sx[12]={-qs,+0,-qs,+0,+0,+qs,+0,+0,+0,-qs,+0,+0};
	int sy[12]={+0,-qs,+0,+qs,+0,+0,+0,+qs,+0,+0,+0,-qs};
	int sz[12]={+0,+0,+0,+0,-qs,+0,-qs,+0,-qs,+0,-qs,+0};
	
	int index[12]={0,0,1,1,2,2,3,3,4,4,5,5};
	
	int len1=6;
	int len2=12;
    
    int image_d=12;
	int sz1=m*n*o;
    
	float* w1=new float[sz1];
	float* sum1=new float[sz1];
	float* noise1=new float[sz1];
	float* d1=new float[sz1*len1];
	
	for(int i=0;i<sz1*len2;i++){
		mind[i]=1.0;
	}
	
	float mean1=0.0;
	for(int i=0;i<sz1;i++){
		w1[i]=0.0;
		sum1[i]=0.0;
		noise1[i]=0.0;
	}
	for(int i=0;i<sz1*len1;i++){
		d1[i]=0.0;
	}
    float* temp1=new float[sz1]; float* temp2=new float[sz1];
	for(int l=0;l<len1;l++){
		imshift(im1,w1,dx[l],dy[l],dz[l],m,n,o);
		for(int i=0;i<sz1;i++){
			w1[i]=pow(w1[i]-im1[i],2.0);
		}
        //movingfilter2(w1,temp1,temp2,m,n,o,sigma);
        volfilter(w1,m,n,o,sigma);
		for(int i=0;i<sz1;i++){
			d1[i+l*sz1]=w1[i];
		}
	}
    delete temp1; delete temp2;
	
	for(int l=0;l<len2;l++){
		imshift(d1+index[l]*sz1,mind+l*sz1,sx[l],sy[l],sz[l],m,n,o);
	}
	
	//subtract mininum
	for(int i=0;i<sz1;i++){
		sum1[i]=1e20;
		for(int l=0;l<len2;l++){
			if(mind[i+l*sz1]<sum1[i]){
				sum1[i]=mind[i+l*sz1];
			}
		}
	}
	for(int i=0;i<sz1;i++){
		for(int l=0;l<len2;l++){
			mind[i+l*sz1]-=sum1[i];
			noise1[i]+=mind[i+l*sz1];
		}
	}
	
	for(int i=0;i<sz1;i++){
		noise1[i]/=(float)(len2);
		mean1+=noise1[i];
	}
	mean1/=(float)(sz1);
	for(int i=0;i<sz1;i++){
		noise1[i]=min(max((float)noise1[i],(float)0.001*mean1),(float)1000.0*mean1);
	}
    
	//-exp/noise
	for(int l=0;l<len2;l++){
		for(int i=0;i<sz1;i++){
			mind[i+l*sz1]=exp(-mind[i+l*sz1]/noise1[i]);
		}
	}
    delete w1;
	delete d1;
    
    
	delete sum1;
	delete noise1;
}

Image quantisedMIND(Image img,int delta,float sigma){
	
    int m=img.m; int n=img.n; int o=img.o;
    
    Image ssc;
    ssc.m=m; ssc.n=n; ssc.o=o;
    ssc.descriptor=new uint64_t[m*n*o];
    
	int d=12;
	int sz=m*n*o;
	
	float* mindf=new float[m*n*o*d];
	
	descriptor(mindf,img.values,m,n,o,delta,sigma);
    int val=6;
    
	int* mindi=new int[sz*d];
	
	for(int i=0;i<sz*d;i++){
		mindi[i]=min(max((int)(mindf[i]*val-0.5),0),val-1);
	}
	delete mindf;
	uint64_t* tablei=new uint64_t[val]; //intensity values
	for(int i=0;i<val;i++){
		tablei[i]=0ULL;
	}
	uint64_t* tabled=new uint64_t[d]; //descriptor entries
	for(int i=0;i<d;i++){
		tabled[i]=0ULL;
	}
	uint64_t power=1ULL;
	tablei[0]=0;
	for(int i=1;i<val;i++){
		power+=power;
		tablei[i]=power-1ULL;
	}
	//printf("power: %d \n",power);
	
	tabled[0]=1;
	for(int i=1;i<d;i++){
		tabled[i]=tabled[i-1]*power;
		
	}
    
	for(int i=0;i<sz;i++){
		ssc.descriptor[i]=0ULL;
		for(int q=0;q<d;q++){
			ssc.descriptor[i]+=tablei[mindi[i+q*sz]]*tabled[q];
		}
	}
	
	delete tabled;
	delete tablei;
	delete mindi;
	
    return ssc;
	
}

