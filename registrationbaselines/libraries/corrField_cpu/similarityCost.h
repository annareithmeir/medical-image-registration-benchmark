float subMin(float* dist,float* subdisp,int search,int ind){
    int len=search*2+1;
    int z=ind/(len*len);
    int x=(ind-z*len*len)/len;
    int y=ind-z*len*len-x*len;
    float v[7]={-dist[max(y-1,0)+x*len+z*len*len],-dist[min(y+1,len-1)+x*len+z*len*len],
        -dist[y+max(x-1,0)*len+z*len*len],-dist[y+min(x+1,len-1)*len+z*len*len],
        -dist[y+x*len+max(z-1,0)*len*len],-dist[y+x*len+min(z+1,len-1)*len*len],-dist[y+x*len+z*len*len]};
    float ay=(v[0]+v[1])*0.5-v[6];
    float by=ay+v[6]-v[0];
    float dy=-by/(2.0*ay);
    float ax=(v[2]+v[3])*0.5-v[6];
    float bx=ax+v[6]-v[2];
    float dx=-bx/(2.0*ax);
    float az=(v[4]+v[5])*0.5-v[6];
    float bz=az+v[6]-v[4];
    float dz=-bz/(2.0*az);
    if(not(dx<1|dx>-1))
        dx=0;
    if(not(dy<1|dy>-1))
        dy=0;
    if(not(dz<1|dz>-1))
        dz=0;
    
    subdisp[0]=(float)(y-search)+dy;
    subdisp[1]=(float)(x-search)+dx;
    subdisp[2]=(float)(z-search)+dz;
    float err1=sqrt(pow((float)(x-search)+dx,2.0)+pow((float)(y-search)+dy,2.0)+pow((float)(z-search)+dz,2.0));
    
    return err1;
}


void similarityCost(float* simVolume,Image ssc_left,Image ssc_right,vector<int> keypoints,int r,int search,int sparse,int skip,float alpha){
    
    int m=ssc_left.m; int n=ssc_left.n; int o=ssc_left.o;
    uint64_t* left=ssc_left.descriptor;
    uint64_t* right=ssc_right.descriptor;
    
    
    int sz=m*n*o;
    int boxlen=r*2+1;
    int boxsz=pow(boxlen,3);
    
    int count=0;
    for(int iz=0;iz<boxlen;iz+=skip){
        for(int ix=0;ix<boxlen;ix+=skip){
            for(int iy=0;iy<boxlen;iy+=skip){
                count++;
            }
        }
    }
    float alpha2=alpha/(float)count/(float)sparse/(float)sparse;
    
    int len=search*2+1;
    int searchsz=pow(len,3);
    float* subdisp=new float[3];
    int* xs=new int[searchsz];
    int* ys=new int[searchsz];
    int* zs=new int[searchsz];
    
    for(int i=0;i<len;i++){
        for(int j=0;j<len;j++){
            for(int k=0;k<len;k++){
                xs[i+j*len+k*len*len]=(j-search)*sparse;
                ys[i+j*len+k*len*len]=(i-search)*sparse;
                zs[i+j*len+k*len*len]=(k-search)*sparse;
            }
        }
    }
    float* newdist=new float[searchsz];
    int xx2,yy2,zz2;
    
    int* pixind=keypoints.data();
    int num=keypoints.size();
    
    for(int i=0;i<num;i++){ //for each search location
        for(int j=0;j<searchsz;j++){
            newdist[j]=0.0;
        }
        int ind1=pixind[i];
        int z=ind1/(m*n);
        int x=(ind1-z*m*n)/m;
        int y=ind1-z*m*n-x*m;
        
        //integration over patch
        for(int iz=0;iz<boxlen;iz+=skip){
            for(int ix=0;ix<boxlen;ix+=skip){
                for(int iy=0;iy<boxlen;iy+=skip){
                    int zz=iz; int yy=iy; int xx=ix;
                    zz+=(z-r);
                    xx+=(x-r);
                    yy+=(y-r);
                    yy=max(min(yy,m-1),0);
                    xx=max(min(xx,n-1),0);
                    zz=max(min(zz,o-1),0);
                    
                    bool boundaries=true; //check image boundaries to save min/max computations
                    if(xx+search*sparse>=n|yy+search*sparse>=m|zz+search*sparse>=o)
                        boundaries=false;
                    if(xx-search*sparse<0|yy-search*sparse<0|zz-search*sparse<0)
                        boundaries=false;
                    
                    for(int l=0;l<searchsz;l++){ //for all possible displacements
                        if(not(boundaries)){
                            xx2=max(min(xx+(int)(xs[l]),n-1),0);
                            yy2=max(min(yy+(int)(ys[l]),m-1),0);
                            zz2=max(min(zz+(int)(zs[l]),o-1),0);
                        }
                        else{
                            xx2=xx+(int)xs[l];
                            yy2=yy+(int)ys[l];
                            zz2=zz+(int)zs[l];
                        }
                        //hamming distance
                        newdist[l]+=(float)__builtin_popcountll(left[yy+xx*m+zz*m*n]^right[yy2+xx2*m+zz2*m*n]);
                    }
                }
            }
        }
        for(int l=0;l<searchsz;l++){
            simVolume[l+i*searchsz]=newdist[l]*alpha2;//subMin(newdist,search,minind);
        }
        
    }
    
    delete xs; delete ys; delete zs;
    delete newdist;
    
}



/*
 if(newdist[l]<minval){
 minind=l;
 minval=newdist[l];
 }
 }
 indices[i]=min_element(dist+i*searchsz,dist+(i+1)*searchsz)-(dist+i*searchsz);
 
 // indices[i]=minind;
 subMin(newdist,subdisp,search,minind);
 for(int ax=0;ax<3;ax++){
 sub[ax+i*3]=subdisp[ax]*(float)sparse;
 }
*/

