void dt1e(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1){
    for(int i=0;i<len;i++){
        f[i]=val[i*k];
        ind1[i]=ind[i*k];
    }
    for(int i=0;i<len;i++){
        float minval=val[0]+z[i+len];//(i-offset)*(i-offset);//log(1+0.25*pow(i-offset,2.0));
        int minind=0;
        for(int j=0;j<len;j++){
            float newval=f[j]+z[i-j+len];//(i-j-offset)*(i-j-offset);//log(1+0.25*pow(i-j-offset,2.0));
            if(newval<minval){
                minind=j;
                minval=newval;
            }
        }
        val[i*k]=minval;
        ind[i*k]=ind1[minind];
    }
    
    
}

void dt1ei(float *val,int len,float offset,int k,float* z,float* f){
    for(int i=0;i<len;i++){
        f[i]=val[i*k];
    }
    for(int i=0;i<len;i++){
        float minval=val[0]+z[i+len];//(i-offset)*(i-offset);//log(1+0.25*pow(i-offset,2.0));
        for(int j=0;j<len;j++){
            float newval=f[j]+z[i-j+len];//(i-j-offset)*(i-j-offset);//log(1+0.25*pow(i-j-offset,2.0));
            if(newval<minval){
                minval=newval;
            }
        }
        val[i*k]=minval;
    }
    
    
}


void dt1sqi(float *val,int len,float offset,int k,int* v,float* z,float* f){
	float INF=1e10;
    
	int j=0;
	z[0]=-INF;
	z[1]=INF;
	v[0]=0;
	for(int q=1;q<len;q++){
		float s=((val[q*k]+(q+offset)*(q+offset))-(val[v[j]*k]+(v[j]+offset)*(v[j]+offset)))/(2.0*(q-v[j]));
        //        float s=((val[q*k]+pow((float)q+offset,2.0))-(val[v[j]*k]+pow((float)v[j]+offset,2.0)))/(2.0*(float)(q-v[j]));
        while(s<=z[j]){
			j--;
            s=((val[q*k]+(q+offset)*(q+offset))-(val[v[j]*k]+(v[j]+offset)*(v[j]+offset)))/(2.0*(q-v[j]));
			//s=((val[q*k]+pow((float)q+offset,2.0))-(val[v[j]*k]+pow((float)v[j]+offset,2.0)))/(2.0*(float)(q-v[j]));
		}
		j++;
		v[j]=q;
		z[j]=s;
		z[j+1]=INF;
	}
	j=0;
	for(int q=0;q<len;q++){
		f[q]=val[q*k]; //needs to be added to fastDT2 otherwise incorrect
	}
	for(int q=0;q<len;q++){
		while(z[j+1]<q){
			j++;
		}
		//val[q*k]=pow((float)q-((float)v[j]+offset),2.0)+f[v[j]];//val[v[j]*k];
        val[q*k]=(q-v[j]-offset)*(q-v[j]-offset)+f[v[j]];
	}
    
}
void dt1sq(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1){
	float INF=1e10;
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
    for(int q=0;q<len;q++){
		f[q]=val[q*k]; //needs to be added to fastDT2 otherwise incorrect
		ind1[q]=ind[q*k];
	}
    
    j=0;
    for(int q=0;q<len;q++){
        while(z[j+1]<(q-offset)){  //was wrong -offset is now correct
            j++;
        }
        ind[q*k]=ind1[v[j]];
        val[q*k]=(q-offset-v[j])*(q-offset-v[j])+f[v[j]];
    }
}

/*
 void dt1sq(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1){
 float INF=1e10;
 
 int j=0;
 z[0]=-INF;
 z[1]=INF;
 v[0]=0;
 for(int q=1;q<len;q++){
 float s=((val[q*k]+(q+offset)*(q+offset))-(val[v[j]*k]+(v[j]+offset)*(v[j]+offset)))/(2.0*(q-v[j]));
 while(s<=z[j]){
 j--;
 s=((val[q*k]+(q+offset)*(q+offset))-(val[v[j]*k]+(v[j]+offset)*(v[j]+offset)))/(2.0*(q-v[j]));
 }
 j++;
 v[j]=q;
 z[j]=s;
 z[j+1]=INF;
 }
 j=0;
 for(int q=0;q<len;q++){
 f[q]=val[q*k]; //needs to be added to fastDT2 otherwise incorrect
 ind1[q]=ind[q*k];
 }
 for(int q=0;q<len;q++){
 while(z[j+1]<q){
 j++;
 }
 ind[q*k]=ind1[v[j]];//ind[v[j]*k];
 //val[q*k]=pow((float)q-((float)v[j]+offset),2.0)+f[v[j]];//val[v[j]*k];
 val[q*k]=(q-v[j]-offset)*(q-v[j]-offset)+f[v[j]];
 }
 
 }*/

void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz){
    //rl is length of one side
    for(int i=0;i<rl*rl*rl;i++){
        indr[i]=i;
    }
    int* v=new int[rl]; //slightly faster if not intitialised in each loop
    float* z=new float[rl+1];
    float* f=new float[rl];
    int* i1=new int[rl];
    
    for(int k=0;k<rl;k++){
        for(int i=0;i<rl;i++){
            dt1sq(r+i+k*rl*rl,indr+i+k*rl*rl,rl,-dx,rl,v,z,f,i1);
        }
    }
    
    for(int k=0;k<rl;k++){
        for(int j=0;j<rl;j++){
            dt1sq(r+j*rl+k*rl*rl,indr+j*rl+k*rl*rl,rl,-dy,1,v,z,f,i1);//);
        }
    }
    
    for(int j=0;j<rl;j++){
        for(int i=0;i<rl;i++){
            dt1sq(r+i+j*rl,indr+i+j*rl,rl,-dz,rl*rl,v,z,f,i1);//);
        }
    }
    float min1=*min_element(r,r+rl*rl*rl);
    for(int i=0;i<rl*rl*rl;i++){
        r[i]-=min1;
    }
    delete []i1;
    delete []f;
    
    delete []v;
    delete []z;
    
    
}
 /*
void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz){
	//rl is length of one side
	for(int i=0;i<rl*rl*rl;i++){
		indr[i]=i;
	}
	int* v=new int[rl]; //slightly faster if not intitialised in each loop
	float* z=new float[rl*2];
	float* f=new float[rl];
	int* i1=new int[rl];
    
    for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dx,2);
    }
	for(int k=0;k<rl;k++){
		for(int i=0;i<rl;i++){
            dt1e(r+i+k*rl*rl,indr+i+k*rl*rl,rl,-dx,rl,v,z,f,i1);//);
			//dt1sq(r+i+k*rl*rl,indr+i+k*rl*rl,rl,-dx,rl,v,z,f,i1);//);
		}
	}
    for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dy,2);
        // z[i]=(1.0-exp(-fabs((float)(i-rl+dy)/(float)rl)))*pow((float)rl,2);
    }
	for(int k=0;k<rl;k++){
		for(int j=0;j<rl;j++){
			dt1e(r+j*rl+k*rl*rl,indr+j*rl+k*rl*rl,rl,-dy,1,v,z,f,i1);//);
		}
	}
	for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dz,2);
        //z[i]=(1.0-exp(-fabs((float)(i-rl+dz)/(float)rl)))*pow((float)rl,2);
    }
	for(int j=0;j<rl;j++){
		for(int i=0;i<rl;i++){
			dt1e(r+i+j*rl,indr+i+j*rl,rl,-dz,rl*rl,v,z,f,i1);//);
		}
	}
    float min1=*min_element(r,r+rl*rl*rl);
    for(int i=0;i<rl*rl*rl;i++){
        r[i]-=min1;
    }
	delete []i1;
	delete []f;
    
	delete []v;
	delete []z;
    
	
}

*/


void dt3xi(float* r,int rl){//,float dx,float dy,float dz){
	//rl is length of one side
    float dx=0.0; float dy=0.0; float dz=0.0;
	float* z=new float[rl*2];
	float* f=new float[rl];
    
    for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dx,2);
    }
	for(int k=0;k<rl;k++){
		for(int i=0;i<rl;i++){
            dt1ei(r+i+k*rl*rl,rl,-dx,rl,z,f);
		}
	}
    for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dy,2);
    }
    
	for(int k=0;k<rl;k++){
		for(int j=0;j<rl;j++){
            dt1ei(r+j*rl+k*rl*rl,rl,-dy,1,z,f);
		}
	}
    for(int i=0;i<rl*2;i++){
        z[i]=pow(i-rl+dz,2);
    }
	for(int j=0;j<rl;j++){
		for(int i=0;i<rl;i++){
            dt1ei(r+i+j*rl,rl,-dz,rl*rl,z,f);
		}
	}
    float min1=*min_element(r,r+rl*rl*rl);
    for(int i=0;i<rl*rl*rl;i++){
        r[i]-=min1;
    }
    
	delete f;
    delete z;
    
	
}

