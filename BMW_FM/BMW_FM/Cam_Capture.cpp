//
//  Cam_Capture.cpp
//  BMW_FM
//
//  Created by Linhao Jin on 28/11/15.
//  Copyright © 2015 Linhao Jin. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

std::string format(int nr, int places)
{
    ostringstream str;
    str << std::setw(places) << std::setfill('0') << nr;
    
    return str.str();
}

int main()
{
    VideoCapture cam[] = { VideoCapture(0), VideoCapture(2) };
    
    if (!cam[0].isOpened() || !cam[1].isOpened())
        return -1;
    
    cv::Mat frame[2];
    
    int count = 0;
    
    while (true)
    {
        cam[0].grab();
        cam[1].grab();
        
        cam[0].retrieve(frame[0]);
        cam[1].retrieve(frame[1]);
        
        imshow("Cam 0:", frame[0]);
        imshow("Cam 1:", frame[1]);
        
        
        if (cv::waitKey(1) == 32) // 32 == spacebar
        {
            imwrite("../data/" + format(count, 4) + "R.png", frame[0]);
            imwrite("../data/" + format(count, 4) + "L.png", frame[1]);
            
            count++;
        }
    }
    
    return 0;
}
