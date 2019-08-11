//#####################################################################################################################
//#This c/c++ code serves two purposes:                                                                               #
//#1. Add gaps to sinograms from VersaPET scanner to match STIR requirement in both transverse and axial planes       #
//#2. Sort the sinogram data output from Michelogram to STIR projection data format                                   #
//#Functions of the code:                                                                                             #
//#1. The code takes the sinogram .scn from the scanner and convert it to Michelogram                                #
//#2. The code converts the Michelogram to STIR Projection data without adding gaps                                   #
//#3. The code adds gaps to Michelogram and then coverts the gap-added Michelogram to STIR projection data            #
//#Author: Alex Shouyi Wei, PhD student in Dept. Biomedical Engineering, Stony Brook University                       #
//#Feb. 14, 2016, Stony Brook, New York, USA                                                                          #
//#####################################################################################################################

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

using namespace std;

#define N_RINGS 16
#define MAX_D_RING 15
#define N_SLICE 256
#define N_DET 192
#define S_WIDTH 191


//currently I don't know how exactly the 3D sinograms from VersaPET scanner are ordered, I defaulted the order of segments as 0, negative 1, positive 1, negative 2, positive 2,..., but it could be the opposite as 0, positive 1, negative 1,...
//following is a lookup table that corresponds the indexes of 3D sinograms from VersaPET scanner to those of Michelograms which is the default one

//another possibility 
int raw_sinogram_index[N_SLICE] = {1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239, 256, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 2, 19, 36, 53, 70, 87, 104, 121, 138, 155, 172, 189, 206, 223, 240, 33, 50, 67, 84, 101, 118, 135, 152, 169, 186, 203, 220, 237, 254, 3, 20, 37, 54, 71, 88, 105, 122, 139, 156, 173, 190, 207, 224, 49, 66, 83, 100, 117, 134, 151, 168, 185, 202, 219, 236, 253, 4, 21, 38, 55, 72, 89, 106, 123, 140, 157, 174, 191, 208, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 81, 98, 115, 132, 149, 166, 183, 200, 217, 234, 251, 6, 23, 40, 57, 74, 91, 108, 125, 142, 159, 176, 97, 114, 131, 148, 165, 182, 199, 216, 233, 250, 7, 24, 41, 58, 75, 92, 109, 126, 143, 160, 113, 130, 147, 164, 181, 198, 215, 232, 249,  8, 25, 42, 59, 76, 93, 110, 127, 144, 129, 146, 163, 180, 197, 214, 231, 248, 9, 26, 43, 60, 77, 94, 111, 128, 145, 162, 179, 196, 213, 230, 247, 10, 27, 44, 61, 78, 95, 112, 161, 178, 195, 212, 229, 246, 11, 28, 45, 62, 79, 96, 177, 194, 211, 228, 245, 12, 29, 46, 63, 80, 193, 210, 227, 244, 13, 30, 47, 64, 209, 226, 243, 14, 31, 48, 225, 242, 15, 32, 241, 16}; 

unsigned short ans;
float Mich_sfu[N_RINGS][N_RINGS][N_DET/2][S_WIDTH]={0};
float Sino_sfu[N_RINGS*N_RINGS][N_DET/2][S_WIDTH]={0};
//float Histogram[N_RINGS*N_DET*(N_RINGS*N_DET-1)/2]={0};

int main(int argc, char** argv){
    string filedir, inputfilename;
    string filename, Michoutputfilename, VersaPEToutputfilename;
    int ring1, ring2;
    int id1=0;
    int id2=0;
    int phi_new=0;
    int u_new=0;
    int phi_temp = 0;
    int u_temp=0;
    int index1=0;
    int index2=0;
    int ind_r1 = 0;
    int ind_r2 = 0;

    if(argc<2) {
    	cout<<" Right number of input argument please !! "<<endl ;
    	return 1;
    }
    
    inputfilename = argv[1];
    cout<<"input file name is"<<inputfilename<<endl;
    filename = argv[2];
    Michoutputfilename = "Mich_" + filename + ".s";
    cout << "New Michelogram file name is = " << Michoutputfilename << endl;     
    VersaPEToutputfilename = "Sino_" + filename + ".s";
    cout << "New Sinogram file name is = " << VersaPEToutputfilename << endl; 
    FILE *Mich_File, *Sino_File;
    Mich_File = fopen(Michoutputfilename.c_str(), "wb");
    Sino_File = fopen(VersaPEToutputfilename.c_str(), "wb");
    FILE *rawdata = fopen(inputfilename.c_str(), "rb");
    vector<float> buffer(N_RINGS*N_DET*(N_RINGS*N_DET-1)/2);
    cout<<"buffer.size() = "<<buffer.size()<<endl;
    fread(&buffer[0], sizeof(float), buffer.size(), rawdata);
    cout<<"is it okay?"<<endl;
   for(int h=0; h<N_RINGS; h++){
      for(int i=0; i<N_RINGS; i++){
	for(int j=0; j<N_DET/2; j++){
          for(int k=0; k<S_WIDTH; k++){

                for(int m=0; m<N_DET; m++){
                  for(int n=m+1; n<N_DET; n++){
                     if(m>n)
                        u_temp = N_DET/2-m+n;
                     else 
                        u_temp = N_DET/2-n+m;
                     
                     phi_temp = m+n-(N_DET/2-1);
                     
                     if(phi_temp < 0){
                        phi_temp += N_DET;
                        u_temp = -u_temp;
                     }
                     
                     if(phi_temp >= N_DET){
                        phi_temp -= N_DET;
                        u_temp = -u_temp;
                     }
                     if(u_temp+S_WIDTH/2 == k && phi_temp/2 == j){
                        id1 = m + h*N_DET;
		        id2 = n + i*N_DET;
                        int id = 0;
			if(id1<id2){
                          
                          for(int x = 0; x<id1; x++){
                              id += N_DET*N_RINGS-x-1;
                          }
                          id+=id2-id1-1;
                        }
                        else{
                         
                          for(int x = 0; x<id2; x++){
                              id += N_DET*N_RINGS-x-1;
                          }
                          id+=id1-id2-1;
                        }
                        //cout<<"id = "<<id<<endl;
                        Mich_sfu[h][i][j][k] = buffer[id];  
                        for(int ss=0; ss<256; ss++){
                            if( raw_sinogram_index[ss] == h*16+i+1){
                                Sino_sfu[ss][j][k] = buffer[id];
                                break;
                            }
                        }
                                                                 
                     
                   
                }
              }
             }
           }
         }

      }  
    }
   
  
    fwrite(Mich_sfu,4,((N_RINGS )*(N_RINGS)*N_DET/2*S_WIDTH),Mich_File);
    fwrite(Sino_sfu,4,((N_RINGS*N_RINGS)*N_DET/2*S_WIDTH),Sino_File);
    fclose(Mich_File);
    fclose(Sino_File);
    return 0;

}
    
     








    
        
        
