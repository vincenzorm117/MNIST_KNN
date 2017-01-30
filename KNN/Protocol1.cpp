


// LAPACK test code

#include <stdio.h>
#include <fstream>
#include <string>
//#include <kdtree.h>
#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <ctime>

#define TRAIN 50000
#define TEST  10000
#define DIM 784

using namespace std;

typedef struct {
    double com[DIM];
    int digit;
} point;

typedef struct {
    double dot;
    int digit;
} dist;


bool compare (dist a, dist b){return a.dot < b.dot; }

bool compare(dist,dist);

int main() {
    

    
    printf("%f",DBL_MAX);
    exit(0);
    
    clock_t start = clock();
    
    FILE *IN = fopen("mnist", "r");
    FILE *OUT = fopen("out.txt", "a");
    
    point save;
    vector<point> training(TRAIN);
    vector<point> test(TEST);
    
    
    
    // Read Input file
    cout << "Reading Input..." << endl;
    
    int i,j,index,value;
    for(i = 0; i < TRAIN; i++){
        fscanf(IN, "%d",&save.digit);
        
        
        for (j = 0; j < DIM; j++)
            save.com[j] = 0;
        
        while (2 == fscanf(IN, " %d:%d",&index,&value)){
            save.com[index] = value;
        }
        
        fseek(IN, -1, SEEK_CUR);
        
        training[i] = save;
        
        
        //        if(TRAIN-10 < i){
        //            printf("%d %d\n", i+1, training[i].digit);
        //            for (j = 0; j < DIM; j++)
        //                printf(" %d:%.0f",j,training[i].com[j]);
        //            cout << endl << endl;
        //        }
    }
    
    
    
    for(i = 0; i < TEST; i++){
        
        fscanf(IN, "%d",&save.digit);
        
        
        for (j = 0; j < DIM; j++)
            save.com[j] = 0;
        
        while (2 == fscanf(IN, " %d:%d",&index,&value)){
            save.com[index] = value;
        }
        
        fseek(IN, -1, SEEK_CUR);
        
        test[i] = save;
        
        //        if(TEST-10 < i){
        //            printf("%d %d\n", i+1, training[i].digit);
        //            for (j = 0; j < DIM; j++)
        //                printf(" %d:%.0f",j,training[i].com[j]);
        //            cout << endl << endl;
        //        }
    }
    
    fclose(IN);
    cout << "Done reading input" << endl;
    
    //Calculate K-Nearest Neighbors
    cout << "Finding K-Nearest Neighbors..." << endl;
    cout << "==============================" << endl;
    
    
    vector<dist> c(DIM);
    point x,y;
    dist d;
    int k,mode, digits[10];
    double error = 0;
    for(k = 40000; k < 40001; k += 2){
        error = 0;
        
        cout << "Finding " << k << " nearest neighbors..." << endl;
        for(i = 0; i < TEST; i++){
            x = test[i];
            
            for (j = 0; j < TRAIN; j++) {
                y = training[j];
                
                cblas_daxpy(DIM, -1, x.com, 1, y.com, 1);
                d.dot = cblas_ddot(DIM, y.com, 1, y.com, 1);
                d.digit = y.digit;
                
                c.push_back(d);
            }
            
            for(j = 0; j < 10; j++)
                digits[j] = 0;
            
            partial_sort(c.begin(), c.begin()+k, c.end(), compare);
            
            // Find mode
            for(std::vector<dist>::iterator it = c.begin(); it != c.begin()+k; it++){
                digits[it->digit]++;
            }
            
            mode = 0;
            for (j=1; j < 10; j++)
                if(digits[mode] < digits[j])
                    mode = j;
            
            
            if(x.digit != mode)
                error = error + 1;
            
            
            c.clear();
        }
        cout << error/TEST << " *100 %"<< endl;
    }
    
    
    
    cout << "==============================" << endl;
    cout << ( clock() - start) / (double) CLOCKS_PER_SEC << endl << endl;
    return 0;
}








