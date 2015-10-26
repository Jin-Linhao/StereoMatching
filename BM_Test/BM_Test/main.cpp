//
//  main.cpp
//  BM_Test
//
//  Created by Linhao Jin on 25/10/15.
//  Copyright © 2015 Linhao Jin. All rights reserved.
//

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat img1, img2, g1, g2;
    Mat disp, disp8;
    char* method = argv[3];
    img1 = imread(argv[1]);
    img2 = imread(argv[2]);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    
    
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);

    
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    
    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);
    
    waitKey(0);
    
    return(0);
}