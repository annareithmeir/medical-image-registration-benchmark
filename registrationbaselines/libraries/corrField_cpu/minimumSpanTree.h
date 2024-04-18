/* Minimum-spanning-tree calcuation using Prim's algorithm.
 Average run-time should be of n*log(n) complexity. 
 Requires full matrix of edge weights of size n x n as input.
 Uses heap data structure to speed-up finding the next lowest edge weight.
*/


struct Edge{
    float weight;
    int startIndex;
    int endIndex;
    friend bool operator<(Edge a,Edge b){
        return a.weight>b.weight;
    }
};

void filterImage(Image scan_fixed,float* meanFixed,int rad){
    int m=scan_fixed.m; int n=scan_fixed.n; int o=scan_fixed.o;

    float* filter=new float[rad*2+1];
    for(int i=0;i<rad*2+1;i++){
        filter[i]=1.0f/(float)(rad*2+1);
    }
    float* temp1=new float[m*n*o];
    
    filter1(scan_fixed.values,meanFixed,m,n,o,filter,rad*2+1,1);
    filter1(meanFixed,temp1,m,n,o,filter,rad*2+1,2);
    filter1(temp1,meanFixed,m,n,o,filter,rad*2+1,3);
    
    delete temp1;
    
}

void edgeGraph(float* edgeWeight,vector<int> keypoints,Image scan_fixed){
    
    int m=scan_fixed.m; int n=scan_fixed.n; int o=scan_fixed.o;

    float* meanFixed=new float[m*n*o];
    filterImage(scan_fixed,meanFixed,2);
    float ms=150.0f;
    
    int num_pts=keypoints.size();
    int countDuplicated=0;
    for(int i=0;i<num_pts;i++){
        int ind1=keypoints[i];
        int z=ind1/(m*n);
        int x=(ind1-z*m*n)/m;
        int y=ind1-z*m*n-x*m;
        for(int j=0;j<num_pts;j++){
            if(i==j){
                edgeWeight[i+j*num_pts]=1e15;
            }
            else{
                int ind2=keypoints[j];
                int z2=ind2/(m*n);
                int x2=(ind2-z2*m*n)/m;
                int y2=ind2-z2*m*n-x2*m;
                float dist=sqrt(max((double)((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2)),1e-20));
                if(dist<0.01){
                    countDuplicated++;
                    dist=0.1;
                }
                dist+=(fabs(meanFixed[ind1]-meanFixed[ind2]))/ms;
                edgeWeight[i+j*num_pts]=dist;
            }
        }
    }
    if(countDuplicated>0){
        printf("%d duplicated keypoints!\n",countDuplicated/2);
    }
}




void primsMST(int* orderedList,int* parentsList,float* edgeWeight,float* euclDistMatrix,int numNodes){
   
    int currentNode=0; //arbritary root node
    //list of nodes already in MST
    bool* addedToMST=new bool[numNodes];
    for(int i=0;i<numNodes;i++){
        addedToMST[i]=false;
    }
    addedToMST[currentNode]=true;
    pair<short,int>* treeLevel=new pair<short,int>[numNodes];
    treeLevel[currentNode]={0,currentNode};
    
    parentsList[currentNode]=-1; //root has no parent
    priority_queue<Edge> priority; //priority queue
    
    //vector<int>* childList=new vector<int>[numNodes];
    
    float mincost=0.0f;
    //run n-1 times so that all nodes added
    for(int i=0;i<numNodes-1;i++){
        //add edges of new node to priority queue
        for(int j=0;j<numNodes;j++){
                float weight=euclDistMatrix[currentNode+j*numNodes];
                priority.push({weight,currentNode,j});
            
        }
        currentNode=-1;
        while(currentNode==-1){
            Edge bestEdge=priority.top();
            priority.pop();
            //test whether endIndex of edge is already in MST
            if(addedToMST[bestEdge.startIndex]&&not(addedToMST[bestEdge.endIndex])){
                mincost+=bestEdge.weight;

                edgeWeight[bestEdge.endIndex]=bestEdge.weight;

                currentNode=bestEdge.endIndex;
                addedToMST[bestEdge.endIndex]=true;
                parentsList[bestEdge.endIndex]=bestEdge.startIndex;
                treeLevel[bestEdge.endIndex]={treeLevel[bestEdge.startIndex].first+1,bestEdge.endIndex};

                //childList[bestEdge.startIndex].push_back(bestEdge.endIndex);
               // printf("adding edge: %d->%d (%2.2f)\n",bestEdge.startIndex,bestEdge.endIndex,bestEdge.weight);
            }
            
        }
        
    }
    //generate list of nodes ordered by tree depth
    
    sort(treeLevel,treeLevel+numNodes);
    //printf("max tree depth: %d\n",treeLevel[numNodes-1].first);
    for(int i=0;i<numNodes;i++){
        orderedList[i]=treeLevel[i].second;
    }

    /*
    orderedList[0]=0; //root node is first
    vector<int> nextLevel(1,0);
    int count=1;
    //reset is addedList
    for(int i=0;i<numNodes;i++){
        addedToMST[i]=false;
    }
    addedToMST[0]=true;

    for(int i=0;i<numNodes;i++){ //large enough number of levels
        vector<int> currentLevel=nextLevel;
        nextLevel.clear();
        for(int j=0;j<currentLevel.size();j++){
            int index=currentLevel[j];
            vector<int> thisList=childList[index];
            for(int k=0;k<thisList.size();k++){
                if(not(addedToMST[thisList[k]])){
                    nextLevel.push_back(thisList[k]);
                    orderedList[count]=thisList[k];
                    addedToMST[thisList[k]]=true;
                    count++;
                }
                
            }
        }
    }
    printf("MST-cost: %2.4f\n",mincost);
    */
}

/*
 if(not(addedToMST[bestEdge.startIndex])&&addedToMST[bestEdge.endIndex]){
 mincost+=bestEdge.weight;
 currentNode=bestEdge.startIndex;
 addedToMST[bestEdge.startIndex]=true;
 printf("case2: adding edge: %d->%d (%2.2f)\n",bestEdge.endIndex,bestEdge.startIndex,bestEdge.weight);
 
 }*/

