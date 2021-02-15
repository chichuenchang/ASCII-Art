//reflect of improvement for the next version
//1) using canny edge-detection to do the edge tracing
//2) supress the trivial pixels that is not effectively contribute to the overall shape
//3) after the previous 2 is done using gradient detection again to get the direction of the lines and print

//*****************************
//comment out the following line to remove parrallel computing feature
#define PARRALLEL_FLAG
//****************************
//#define DEBUG

#ifdef PARRALLEL_FLAG
#include <omp.h>
#endif
#include <iostream>
#include "CImg.h"
#include "math.h"
#include <time.h>


using namespace std;
using namespace cimg_library;

//Declare Gaussian Mask
float GaussianM[5][5] = {
	2,	4,	5,	4,	2,
	4,	9,	12,	9,	4,
	5,	12,	15,	12,	5,
	4,	9,	12,	9,	4,
	2,	4,	5,	4,	2,
};

//Declare Sobel Mask
int GxMask[3][3] = {
	-1,	0, 1,
	-2, 0, 2,
	-1, 0, 1 };
int GyMask[3][3] = {
	 1,  2,	 1,
	 0,  0,	 0,
	-1, -2, -1 };


#ifdef DEBUG
int parrallelTest() {
	//#pragma omp parallel for
	for (int i = 0; i < 3; i++) {
	#pragma omp parallel for
		for (int j = 0; j < 3; j++) {
			printf("i = %d, j = %d,  I am Thread %d\n", i, j, omp_get_thread_num());
		}
	}
}
#endif

int main() {

	clock_t tStart = clock();

#ifdef DEBUG
	parrallelTest();
#endif
	//Playable Variables========================
	//edge detection treshold
	float thres = 50;
	//color interval (1 - 8)
	int colInt = 8;
	//customize image size
	const int custSize = 140;
	//Load in the image (input0 to input15)
	CImg<float> input("input/input0.bmp");
	//=========================================
	//set up the ascii char to fill the empty space
	char asciiFill[9] = " .:;?#%$";//.;(!at8%//".:;<!(#%$";//.:-=+*#%@
	//char asciiFill[9] = " .;(!t8%";//.;(!at8%//".:;<!(#%$";//.:-=+*#%@ //char asciiFill[67] = ".`^,:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
	//this variable just resizes the display size. Might be useful if you can't fit the full resolution of the image onto your monitor.
	const int imgscale = 1.4 * 400;

	//Create an empty image of the same size as green with 3 color components and defaulting to 0 (black)
	//CImg<float> merged(green.width(),green.height(),1,3,0);

	CImg<float> stp1Gray(input.width(), input.height(), 1, 3, 0);
	CImg<float> stp2Gaus(input.width(), input.height(), 1, 3, 0);
	CImg<float> stp3Grad(input.width(), input.height(), 1, 3, 0);
	CImg<float> stp3edgDir(input.width(), input.height(), 1, 3, 0);

	//const of separate unit
	const int numPixel = input.height()*input.width();
	const int w = input.width();
	const int h = input.height();
	
	//**STEP 1: GrayScale==================================
	//col = 0.299*r + 0.587*g + 0.114*g;

	#ifdef PARRALLEL_FLAG
	#pragma omp parallel for collapse (2)
	#endif
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			stp1Gray[i*w + j] = 0.299*input[i*w + j] + 0.587*input[i*w + j + numPixel] + 0.114*input[i*w + j + 2 * numPixel];
			stp1Gray[i*w + j + numPixel] = 0.299*input[i*w + j] + 0.587*input[i*w + j + numPixel] + 0.114*input[i*w + j + 2 * numPixel];
			stp1Gray[i*w + j + 2 * numPixel] = 0.299*input[i*w + j] + 0.587*input[i*w + j + numPixel] + 0.114*input[i*w + j + 2 * numPixel];
		}
	}
	

	CImgDisplay Step1_disp(stp1Gray, "Stage1 Grayscale ");
	Step1_disp.resize(imgscale, imgscale, true); //change size
	Step1_disp.move(30 + 0.5*imgscale, 50); //place it nicely on screen.
	//save it:
	stp1Gray.save("output/stage1.bmp");

	//**STEP 2: Filter the noise with Gaussian Mask=======================
	//loop through the entire pixel of destination image
	#ifdef PARRALLEL_FLAG
	#pragma omp parallel for collapse(2)
	#endif
	for (int i = 2; i < h - 2; i++) {
		for (int j = 2; j < w - 2; j++) {

			//loop through all the grid in kernel	
			for (int iOff = -2; iOff <= 2; iOff++) {
				for (int jOff = -2; jOff <= 2; jOff++) {
					int iPos = i + iOff;
					int jPos = j + jOff;

					int iGaus = 2 + iOff;
					int jGaus = 2 + jOff;

					#ifdef PARRALLEL_FLAG
					//#pragma omp parallel for
					#endif
					for (int k = 0; k <= 2; k++) {//loop through 3 channel
						stp2Gaus[i*w + j + k * numPixel] += stp1Gray[iPos*w + jPos + k * numPixel] * GaussianM[iGaus][jGaus];
					}
				}
			}

			
			for (int k = 0; k <= 2; k++) {
				stp2Gaus[i*w + j + k * numPixel] = stp2Gaus[i*w + j + k * numPixel] / 159;
			}
		}
	}

	CImgDisplay Step2_disp(stp2Gaus, "stage2 Gaussian Blur");
	Step2_disp.resize(imgscale, imgscale, true); //change size
	Step2_disp.move(30 + imgscale, 50 + 20); //place it nicely on screen.
	//save it:
	stp2Gaus.save("output/stage2.bmp");


	//step3 Gradient =======================================================
	#ifdef PARRALLEL_FLAG
	#pragma omp parallel for collapse (2)
	#endif
	for ( int i = 1; i < h - 1; i++) {//loop through all pixel
		for (int j = 1; j < w - 1; j++) {
			float angle = 0.0;
			float apxAngle = 0.0;

			//the adjustment of approximate angle
			float adjust = 8.0;
			float Gx = 0.0;
			float Gy = 0.0;
				
			//int iPic, jPic, iSob, jSob(0);
			//#pragma omp parallel for collapse(2) /*reduction (+: Gx, Gy)*/ /*private(iPic, jPic, iSob, jSob)*/
			for (int iOff = -1; iOff <= 1; iOff++) {//loop though sobel kernel
				for (int jOff = -1; jOff <= 1; jOff++) {

					int iPic = i + iOff;
					int jPic = j + jOff;

					int iSob = iOff + 1;
					int jSob = jOff + 1;

					Gx += stp2Gaus[iPic*w + jPic] * GxMask[iSob][jSob];
					Gy += stp2Gaus[iPic*w + jPic] * GyMask[iSob][jSob];
				}
			}

			// clamp Gx and Gy value
			if (Gx < thres && Gx > -thres) Gx = 0;
			if (Gy < thres && Gx > -thres) Gy = 0;
			if (Gx >= thres) Gx = 250;
			if (Gx <= -thres) Gx = -250;

			#ifdef PARRALLEL_FLAG
			//#pragma omp parallel for 
			#endif
			for (int k = 0; k <= 2; k++) {//loop through the rgb channel
				stp3Grad[i*w + j + k * numPixel] = sqrt(Gx*Gx + Gy * Gy);
			}

			//atan2 returns precise value of the angle
			angle = (atan2(Gx, Gy) / 3.1415)*180.0;

			if (((angle < (22.5 - adjust)) && (angle > (-22.5 + adjust))) || ((angle > (157.5 + adjust)) && (angle < (-157.5 - adjust))))
				apxAngle = 0;
			if (((angle >= (22.5 - adjust)) && (angle <= (67.5 + adjust))) || ((angle <= (-112.5 + adjust)) && (angle >= (-157.5 - adjust))))
				apxAngle = 45;
			if (((angle > (67.5 + adjust)) && (angle < (112.5 - adjust))) || ((angle < (-67.5 -adjust)) && (angle > (-112.5 + adjust))))
				apxAngle = 90;
			if (((angle > (112.5 - adjust)) && (angle < (157.5 + adjust))) || ((angle < (-22.5 + adjust)) && (angle > (-67.5 - adjust))))
				apxAngle = 135;

			//don't print anything if color is not changing
			if (Gx == 0 && Gy == 0)
				apxAngle = -999.0;

			stp3edgDir[i*w + j] = apxAngle;

		}
	}

	std::cout << "Time taken : " << (double)(clock() - tStart) / CLOCKS_PER_SEC << std::endl;

	CImgDisplay Step3_disp(stp3Grad, "stage3 Gradient");
	Step3_disp.resize(imgscale, imgscale, true); //change size
	Step3_disp.move(30 + 1.5*imgscale, 50 + 40); //place it nicely on screen.
	//save it:
	stp3Grad.save("output/stage3.bmp");

	//step4 print ==========================================================

	const int resizeH = custSize;
	const int resizeW = 2*resizeH;
	stp3edgDir.resize(resizeW, resizeH, true);

	int intens = 0;
	stp2Gaus.resize(resizeW, resizeH, true);
	for (int r = 0; r < 2; r++) {
		for (int i = 1; i < resizeH - 1; i++) {//loop through all pixel
			for (int j = 1; j < resizeW - 1; j++) {
				intens = stp2Gaus[i*resizeW + j];

				if (stp3edgDir[i*resizeW + j] == 0) std::cout << "-";
				else if (stp3edgDir[i*resizeW + j] == 45) std::cout << "\\";
				else if (stp3edgDir[i*resizeW + j] == 90) std::cout << "|";
				else if (stp3edgDir[i*resizeW + j] == 135) std::cout << "/";
				else {
					if (r==0) std::cout << " ";
					else std::cout << asciiFill[(intens)* colInt / 256];
				}
			}

			std::cout << std::endl;
		}
	}
	//show source image after print

	CImgDisplay input_disp(input, "Original 'input.bmp'");
	input_disp.resize(imgscale, imgscale, true); //change size
	input_disp.move( 1.3*imgscale, 250 ); //place it nicely on screen.

	//===================================================

	std::cout << "If you can clearly see this line, that means after the program is runned you won't see a proper output" << std::endl;
	std::cout << "You need to click the upper left corner of your console window." << std::endl;
	std::cout << "Go to property -> Font -> Size -> change to 5." << std::endl;
	std::cout << "Then property -> layout -> window size -> width -> change to 500" << std::endl;
	std::cout << "Then property -> layout -> window size -> height -> change to 500" << std::endl;
	//Prevent closure:
	//printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	
	std::cin.ignore();
	std::cin.get();
	return 0;
}