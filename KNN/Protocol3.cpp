

#include <stdio.h>
#include <fstream>
#include <string>
//#include <kdtree.h>
#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <limits.h>

#define ALL 60000
#define VALIDATION 6000
#define DIM 784
#define K   15
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
    
    FILE *IN = fopen("mnist", "r");
    FILE *OUT = fopen("out.txt", "a");
    
    
    point save;
    vector<point> training(ALL);
    vector<point> validation(VALIDATION);
    
    
    // Reading Input file
    fprintf(OUT,"Reading Input...\n");
    printf("Reading Input...\n");
    
    int i,j,index,value;
    for (i = 0; i < ALL; i++) {
        fscanf(IN, "%d",&save.digit);
        
        for (j = 0; j < DIM; j++)
            save.com[j] = 0;
        
        while( 2 == fscanf(IN, " %d:%d",&index,&value))
            save.com[index] = value;
        
        fseek(IN, -1, SEEK_CUR);
        
        training[i] = save;
    }
    
    
    
    fclose(IN);
    printf("Done reading input\n");
    fprintf(OUT, "Done reading input\n");
    
    
    //Calculate 10-fold cross validation
    printf("Starting K-Fold Cross Validations\n");
    printf("=================================\n");
    fprintf(OUT,"Starting K-Fold Cross Validations\n");
    fprintf(OUT,"=================================\n");
    
    
    dist d;
    vector<dist> c(DIM);
    point val,train;
    int error,minK[2], digits[10],mode;
    int range,k,A;
    for(range = 0; range < 10; range++){
        printf("Running range %d\n===================================\n",range);
        fprintf(OUT,"Running range %d\n===================================\n",range);
        
        minK[1] = DBL_MAX;
        A = range*VALIDATION;
        
        for (i = 0; i < VALIDATION; i++)
            validation[ i ] = training[ i + A ];
        
        // Find K-Nearest Neighbors
        for (k = 1; k < 16; k++) {
            error = 0;
            printf("Finding %d Nearest Neighbor\n--  ",k);
            fprintf(OUT,"Finding %d Nearest Neighbor\n--  ",k);
            fflush(OUT);
            
            for (i = 0; i < VALIDATION; i++) {
                val = validation[i];
                
                
                for (j = 0; j < ALL; j++) {
                    if(j == A){
                        j += VALIDATION;
                        if(j == ALL)
                            break;
                    }
                    
                    train = training[j];
                    
                    cblas_daxpy(DIM, -1, val.com, 1, train.com, 1);
                    d.dot = cblas_ddot(DIM, train.com, 1, train.com, 1);
                    d.digit = train.digit;
                    
                    c.push_back(d);
                }
                
                for(j = 0; j < 10; j++)
                    digits[j] = 0;
                
                partial_sort(c.begin(), c.begin()+k, c.end(), compare);
                
                // Find mode
                for(std::vector<dist>::iterator it = c.begin(); it != c.begin()+k; it++)
                    digits[it->digit]++;
                
                
                mode = 0;
                for (j=1; j < 10; j++)
                    if(digits[mode] < digits[j])
                        mode = j;
                
                
                if(val.digit != mode)
                    error = error + 1;
                
                
                c.clear();
                
            }
            
            if(error < minK[1]){
                minK[0] = k;
                minK[1] = error;
            }
            
            
            printf("Error = %f %%\n",(double)error * 100 / VALIDATION);
            fprintf(OUT,"Error = %f %%\n",(double)error * 100 / VALIDATION);
            fflush(OUT);
        }
        
        printf("====================================\n");
        printf("Range: %d | Min K = %d with %.4f", range, minK[0], (double) minK[1] * 100 / VALIDATION );
        fprintf(OUT,"====================================\n");
        fprintf(OUT,"Range: %d | Min K = %d with %.4f", range, minK[0], (double) minK[1] * 100 / VALIDATION );
        
        validation.clear();
    }
    
    fclose(OUT);
    return  0;
}









