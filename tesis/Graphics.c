#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#define rows 10
#define columns 10000

int main()
{
	FILE *File_1; 							
	double data[rows][columns];
	File_1 = fopen( "data.dat", "r" );

	int i;
	int j;
	for( i=0; i<rows; i++ ){					
		for(j=0; j<columns; j++){		    
			fscanf( File_1, "%lf", &data[i][j] );
		}
	}
	
	FILE *File_new;						    
	File_new = fopen("DataInvert.txt", "w");
	
	for( i=0; i<columns; i++ ){			
		for(j=0; j<rows; j++){
			fprintf( File_new, "%lf\t", data[j][i] );
		}
		fprintf(File_new, "\n");
	}
	fclose(File_new);
    
	FILE *gnuplot=NULL;
	gnuplot=popen("gnuplot", "w");
	fprintf(gnuplot, "set term postscript eps enhanced color\n");
	for(i=1; i<=rows; i++ ){
		fprintf(gnuplot, "set output 'graphic_%i.eps'\n", i);
		fprintf(gnuplot, "plot 'DataInvert.txt' u %d with lines\n", i);
		fprintf(gnuplot, "set output\n"); 
		fflush(gnuplot);
	}
	pclose(gnuplot);
}
