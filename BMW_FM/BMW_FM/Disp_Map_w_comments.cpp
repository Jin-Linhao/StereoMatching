//
//  main.cpp
//  BMW_FM
//
//  Created by Linhao Jin on 22/11/15.
//  Copyright © 2015 Linhao Jin. All rights reserved.
//

//
//  main.cpp
//  Opencv tutorial
//
//  Created by Linhao Jin on 10/10/15.
//  Copyright © 2015 Linhao Jin. All rights reserved.
//


#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"


#include <stdio.h>

using namespace cv;
using namespace std;




static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
    printf("\nUserguide: In terminal, cd to /Users/LH_Mac/Desktop/BMW_FMRL_Image_Depth/OpenCV TR/Opencv tutorial/build/Debug, type ./Opencv\ tutorial LEFT_IMAGE_PATH RIGHT_IMAGE_PATH --algorithm=sgbm");
}










// save image from command line   as an matrix. fabs means compute absolute value
static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}










//select option in command line

//argv and argc are how command line arguments are passed to main() in C and C++.

//argc will be the number of strings pointed to by argv. This will (in practice) be 1 plus the number of arguments, as virtually all implementations will prepend the name of the program to the array.

//The variables are named argc (argument count) and argv (argument vector) by convention, but they can be given any valid identifier: int main(int num_args, char** arg_strings) is equally valid.

//They can also be omitted entirely, yielding int main(), if you do not intend to process command line arguments.

int main(int argc, char** argv)
{
    
    //disparity map
    int BlockSize = 5;
    int temp11;
    int number_of_disparities = 80;
    int temp2;
    int pre_filter_size = 5;
    int temp3;
    int pre_filter_cap = 23;
    int temp4;
    int min_disparity = 1;
    int temp5;
    int texture_threshold = 500;
    int temp6;
    int uniqueness_ratio = 0;
    int temp7;
    int max_diff = 100;
    float temp8;
    int speckle_window_size = -10;
    int temp9;
    
    
    //tracking bar parameters
    int erosion_size = 0;
    int dilation_size = 0;
    
    
    
    //these are the options, they don't have to be in the command window
    const char* algorithm_opt = "--algorithm=";
    //const char* maxdisp_opt = "--max-disparity=";
    //const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display";
    const char* scale_opt = "--scale=";
    
    //if the input is less than 3 items (executable name, left image, right image),print_help. This will usually happen when directly click the executable
    if(argc < 3)
    {
        print_help();
        return 0;
    }
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* intrinsic_filename = 0;
    const char* extrinsic_filename = 0;
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;
    
    //I think this one is to assign the values to different algorithms and set default algorithm as SGBM
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
    int alg = STEREO_SGBM;
    //int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;
    
    //static Ptr<> load (const String &filename, const String &objname=String()), Loads algorithm from the file.
    
    //static Ptr<StereoBM> cv::StereoBM::create	(	int 	numDisparities = 0,
    //                                              int 	blockSize = 21      )
    //Class for computing stereo correspondence using the block matching algorithm, introduced and contributed to OpenCV by K. Konolige.
    //
    //Creates StereoBM object.
    
    //Parameters
    //numDisparities:  disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity.
    //blockSize:  the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.
    //
    //The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for a specific stereo pair.
    
    Ptr<StereoBM> bm = StereoBM::create();
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    
    for( int i = 1; i < argc; i++ )
    {
        //since strings are characters, argv[i][0] means the first characater ("0") in the "i+1"th string
        //So this means if the first character of command line argument is not equal to "-" (which is the start of the selection), the program will run the if-else function
        //For integral types, ! returns true if the operand is zero, and false otherwise. For !img1_filename here just means img1_filename = 0. Means if it's not exist before, it will be assigned to argv[i]. Note that i starts from 1, so i=0 which is the executable name will not be examined.
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        
        //strcmp means compare two strings(probably to see if they are equal). strlen returns the length of the given byte string, that is, the number of characters in a character array whose first element is pointed to by str up to and not including the first null character.
        //this one test the similarity between input parameter and exist algorithms
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
            strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
            strcmp(_alg, "hh") == 0 ? STEREO_HH :
            strcmp(_alg, "var") == 0 ? STEREO_VAR :
            strcmp(_alg, "sgbm3way") == 0 ? STEREO_3WAY : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        
        
        
        //for maximun disparity option
        //        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        //        {
        //            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
        //               numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
        //            {
        //                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        //                print_help();
        //                return -1;
        //            }
        //        }
        
        
        
        //for block size option
        //        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        //        {
        //            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
        //               SADWindowSize < 1 || SADWindowSize % 2 != 1 )
        //            {
        //                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        //                return -1;
        //            }
        //        }
        else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
            {
                printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
                return -1;
            }
        }
        
        //for other options
        else if( strcmp(argv[i], nodisplay_opt) == 0 )
            no_display = true;
        else if( strcmp(argv[i], "-i" ) == 0 )
            intrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-e" ) == 0 )
            extrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-o" ) == 0 )
            disparity_filename = argv[++i];
        else if( strcmp(argv[i], "-p" ) == 0 )
            point_cloud_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    
    
    //test input options (if we miss a input)
    if( !img1_filename || !img2_filename )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    
    
    if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }
    
    if( extrinsic_filename == 0 && point_cloud_filename )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }
    
    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);
    
    
    
    
    //test the whether the string is empty
    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }
    
    
    //input scale factor
    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }
    
    Size img_size = img1.size();
    
    Rect roi1, roi2;
    Mat Q;
    
    if( intrinsic_filename )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename);
            return -1;
        }
        
        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;
        
        M1 *= scale;
        M2 *= scale;
        
        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename);
            return -1;
        }
        
        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;
        
        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
        
        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
        
        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);
        
        img1 = img1r;
        img2 = img2r;
    }
    
    
    //this one should be calculate the number of disparities
    //numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
    
    
    //setting the block matching and semi-global bm algorithm parameters
    
    
    
    namedWindow("disp", 20);
    //create disp parameters tracking bar
    createTrackbar("WindowSize", "disp", & BlockSize, 50, NULL);
    createTrackbar("no_of_disparities", "disp", &number_of_disparities,255, NULL);
    createTrackbar("filter_size", "disp", &pre_filter_size,255, NULL);
    createTrackbar("filter_cap", "disp", &pre_filter_cap,63, NULL);
    createTrackbar("min_disparity", "disp", &min_disparity,60, NULL);
    createTrackbar("texture_thresh", "disp", &texture_threshold,2000, NULL);
    createTrackbar("uniquness", "disp", &uniqueness_ratio,30, NULL);
    createTrackbar("disp12MaxDiff", "disp", &max_diff,100, NULL);
    createTrackbar("Speckle Window", "disp", &speckle_window_size,50, NULL);
    
    
    
    /// Create Erosion Trackbar
    createTrackbar( "Erode Kernel size", "disp", &erosion_size, 25, NULL);
    
    /// Create Dilation Trackbar
    createTrackbar( "Dilate Kernel size", "disp", &dilation_size, 25, NULL);
    
    
    
    
    
    while(1)
    {
        int i1;
        i1 = BlockSize;
        if(i1%2==0 && i1>=7)
        {
            temp11 = i1-1;
            bm->setBlockSize(temp11);
            sgbm->setBlockSize(temp11);
            
        }
        if(i1%2!=0 && i1>=7)
        {
            temp11 = i1;
            bm->setBlockSize(temp11);
            sgbm->setBlockSize(temp11);
        }
        
        
        int i2;
        i2 = number_of_disparities;
        if(i2%16!=0 && i2>16)
        {
            temp2 = i2 - i2%16;
            bm->setNumDisparities(temp2);
            sgbm->setNumDisparities(temp2);
        }
        if(i2%16==0 && i2>16)
        {
            temp2 =	i2;
            bm->setNumDisparities(temp2);
            sgbm->setNumDisparities(temp2);
        }
        if(i2<=16)
        {
            temp2 =	16;
            bm->setNumDisparities(temp2);
            sgbm->setNumDisparities(temp2);
            
        }
        
        int i3;
        i3 = pre_filter_cap;
        if(i3%2==0 && i3>=7)
        {
            temp3 = i3-1;
            bm->setPreFilterCap(temp3);
            sgbm->setPreFilterCap(temp3);
        }
        if(i3<7)
        {
            temp3 =	7;
            bm->setPreFilterCap(temp3);
            sgbm->setPreFilterCap(temp3);
            
        }
        if(i3%2!=0 && i3>=7)
        {
            temp3 =	i3;
            bm->setPreFilterCap(temp3);
            sgbm->setPreFilterCap(temp3);
        }
        
        
        int i5;
        i5 = min_disparity;
        temp5 = -i5;
        bm->setMinDisparity(temp5);
        sgbm->setMinDisparity(temp5);
        
        int i6;
        i6 = texture_threshold;
        temp6 = i6;
        bm->setTextureThreshold(temp6);
        
        int i7;
        i7 = uniqueness_ratio;
        temp7 = i7;
        bm->setUniquenessRatio(temp7);
        sgbm->setUniquenessRatio(temp7);
        
        int i8;
        i8 = max_diff;
        temp8 = 0.01*((float)i8);
        bm->setDisp12MaxDiff(temp8);
        sgbm->setDisp12MaxDiff(temp8);
        
        int i9;
        i9 = speckle_window_size;
        temp9 = i9;
        bm->setSpeckleWindowSize(temp9);
        sgbm->setSpeckleWindowSize(temp9);
        
        bm->setROI1(roi1);
        bm->setROI2(roi2);
        //bm->setPreFilterCap(63);
        //bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 3);
        //bm->setMinDisparity(0);
        //bm->setNumDisparities(numberOfDisparities);
        //bm->setTextureThreshold(10);
        //bm->setUniquenessRatio(15);
        //bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        //bm->setDisp12MaxDiff(1);
        
        //sgbm->setPreFilterCap(63);
        //int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
        //sgbm->setBlockSize(sgbmWinSize);
        
        int cn = img1.channels();
        
        //sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
        //sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
        //sgbm->setMinDisparity(0);
        //sgbm->setNumDisparities(numberOfDisparities);
        //sgbm->setUniquenessRatio(10);
        //sgbm->setSpeckleWindowSize(100);
        //sgbm->setSpeckleRange(32);
        //sgbm->setDisp12MaxDiff(1);
        if(alg==STEREO_HH)
            sgbm->setMode(StereoSGBM::MODE_HH);
        else if(alg==STEREO_SGBM)
            sgbm->setMode(StereoSGBM::MODE_SGBM);
        
        
        Mat disp, disp8, disp9, disp10;
        //Mat img1p, img2p, dispp;
        //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
        //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
        
        
        
        //function below calculates the disparity map from 2 images and assigned parameters. Actually there are 4 algorithms. When choosing SGBM, inside the function it will direct to HH or SGBM3WAY
        //&& means "and", || means "or"
        //gettickcount calculate the processing time
        int64 t = getTickCount();
        if( alg == STEREO_BM ){
            
            bm->compute(img1, img2, disp);
        }
        else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
            sgbm->compute(img1, img2, disp);
        t = getTickCount() - t;
        printf("Time elapsed: %fms\n", t*1000/getTickFrequency());
        
        
        
        
        
        
        //disp = dispp.colRange(numberOfDisparities, img1p.cols);
        
        //following code first assign a greyscale value to different disparity level
        //then show the disparity map as well as the input pictures
        
        //Flush stream
        //If the given stream was open for writing (or if it was open for updating and the last i/o operation was an output operation) any unwritten data in its output buffer is written to the file.
        
        if( alg != STEREO_VAR )
            disp.convertTo(disp8, CV_8U, 255/(number_of_disparities*16.));
        else
            disp.convertTo(disp8, CV_8U);
        
        
        Mat element1 = getStructuringElement( MORPH_ELLIPSE,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );
        /// Apply the dilation operation
        dilate( disp8, disp9, element1 );
        
        
        Mat element2 = getStructuringElement( MORPH_ELLIPSE,
                                             Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                             Point( erosion_size, erosion_size ) );
        
        // Apply the erosion operation
        erode( disp9, disp10, element2 );
        
        
        
        //dilation and erodation
        
        
        
        
        
        
        
        if( !no_display )
        {
            namedWindow("left", 1);
            imshow("left", img1);
            namedWindow("right", 1);
            imshow("right", img2);
            namedWindow("disparity", 0);
            imshow("disparity", disp10);
            printf("press any key to continue...");
            fflush(stdout);
            waitKey();
            printf("\n");
        }
        
        //write the disparity matrix into a file
        if(disparity_filename)
            imwrite(disparity_filename, disp8);
        
        if(point_cloud_filename)
        {
            printf("storing the point cloud...");
            fflush(stdout);
            Mat xyz;
            reprojectImageTo3D(disp, xyz, Q, true);
            saveXYZ(point_cloud_filename, xyz);
            printf("\n");
        }
    }
    
    
    return 0;
}






