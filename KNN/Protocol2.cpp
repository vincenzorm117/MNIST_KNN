


// LAPACK test code

#include <stdio.h>
#include <fstream>
#include <string>
//#include <kdtree.h>
#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <climits>

#define TRAIN       40000
#define TEST        10000
#define VALIDATION  10000
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


    clock_t start = clock();

    FILE *IN = fopen("mnist", "r");
    

    point save;
    vector<point> training(TRAIN);
    vector<point> test(TEST);
    vector<point> validation(VALIDATION);



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

    }
    
    
    for(i = 0; i < VALIDATION; i++){
        
        fscanf(IN, "%d",&save.digit);
        
        
        for (j = 0; j < DIM; j++)
            save.com[j] = 0;
        
        while (2 == fscanf(IN, " %d:%d",&index,&value)){
            save.com[index] = value;
        }
        
        fseek(IN, -1, SEEK_CUR);
        
        validation[i] = save;
        
    }
    
    

    fclose(IN);
    cout << "Done reading input" << endl;

    //Calculate K-Nearest Neighbors
    cout << "Finding K-Nearest Neighbors..." << endl;
    cout << "==============================" << endl;


    vector<dist> c(DIM);
    point x,y;
    dist d;
    int k,mode, digits[10], minK = -1;
    double error = 0, minKValue;
    
    minKValue = DBL_MAX;
    for(k = 10; k < 11; k++){
        error = 0;

        cout << "Finding " << k << " nearest neighbors: ";
        for(i = 0; i < VALIDATION; i++){
            // Extract current test/validation vector
            x = validation[i];

            // Compute euclidean distances for all training set vectors
            for (j = 0; j < TRAIN; j++) {
                // Extract current training vector
                y = training[j];
                // Compute euclidean distance
                cblas_daxpy(DIM, -1, x.com, 1, y.com, 1);
                // Compute dot product from euclidean distance
                // and store it in struct
                d.dot = cblas_ddot(DIM, y.com, 1, y.com, 1);
                // Store digit in struct and store struct in C++ vector
                d.digit = y.digit;
                c.push_back(d);
            }

            // Clear frequency array
            for(j = 0; j < 10; j++){
                digits[j] = 0;
            }

            //Perform Partial sort
            partial_sort(c.begin(), c.begin()+k, c.end(), compare);
            // Increase frequency in frequency array of closest k neighbors
            for(std::vector<dist>::iterator it = c.begin(); it != c.begin()+k; it++){
                digits[it->digit]++;
            }

            //Find mode
            mode = 0;
            for (j=1; j < 10; j++){
                if(digits[mode] < digits[j])
                    mode = j;
            }

            // Check if digit matches test/validation's digit
            if(x.digit != mode){
                error = error + 1;
            }

            // Clear temp vector containing euclidean distances
            // with respective training set example digit
            c.clear();
        }
        
        if(error < minKValue){
            minKValue = error;
            minK = k;
        }
            
            
        cout << "VALIDATION: " << error * 100 / VALIDATION << "Error. With min K: " << minK << " at: "<< minKValue << endl;
    }
    
    
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
    
    cout << "TEST: " << error * 100 / TEST << endl;
    


    cout << "==============================" << endl;
    cout << ( clock() - start) / (double) CLOCKS_PER_SEC << endl << endl;
    return 0;
}








