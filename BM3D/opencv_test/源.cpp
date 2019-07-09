//#include <iostream>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/opencv.hpp>
//
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat image;
//	image = imread("E:/opencv_learn/opencv_test/draw.jpg");
//	resize(image,image, Size(),0.4,0.4);
//	cout << image.size() << endl;
//	VideoCapture cam(0);
//	Mat frame;
//	cam.set(CAP_PROP_FOURCC, 'GPJM');
//	cam.set(CAP_PROP_FRAME_WIDTH, 1920);
//	cam.set(CAP_PROP_FRAME_HEIGHT, 1080);
//	cam >> frame;
//	cout << image.cols << ' ' << image.rows<<endl;
//	Rect roi(250, 250, image.cols, image.rows);
//	for (; waitKey(1) != 27; cam >> frame) { //esc 的ascall码是27；
//		image.copyTo(frame(roi));
//		imshow("frame", frame);
//
//	}
//
//	/*namedWindow("Display window", WINDOW_AUTOSIZE);
//	imshow("Display window", image);
//
//	waitKey(0);
//
//	std::cout << "Hello World!\n";*/
//}

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2\imgproc\types_c.h>
//#include <opencv2/gpu/gpu.hpp>
using namespace cv;
using namespace std;



string xmlPath = "E:/opencv_learn/opencv_test/haarcascade_frontalface_default.xml";
//xmlpath 字符串记录那个.xml文件的路径
Mat detectAndDisplay(Mat image);

string HashValue(Mat& src);
int HanmingDistance(string& str1, string& str2);
int total_people_number = 8;
string str2[9];

int main(int argc, char** argv)
{
	string path = "E:/opencv_learn/opencv_test/1.jpg";//以检测图片1.jpg为例
	Mat image = imread(path, -1);

	CascadeClassifier a;     //创建脸部对象
	if (!a.load(xmlPath))     //如果读取文件不出错，则检测人脸
	{
		cout << "无法加载xml文件" << endl;
		return 0;
	}

	//视频读取
	VideoCapture cam(0);
	Mat frame;
	cam.set(CAP_PROP_FOURCC, 'GPJM');
	cam.set(CAP_PROP_FRAME_WIDTH, 1920);
	cam.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cam >> frame;
	Rect roi(250, 250, image.cols, image.rows);

	string path2;
	//计算
	for (int i = 1; i <= 8; i++)//因为我完成的就是8张图片的检测，所以循环值为8
	{
		string path2 = format("E:/opencv_learn/opencv_test/face/%d.jpg", i);
		Mat image2 = imread(path2, -1);
		Mat face=detectAndDisplay(image2);
		str2[i] = HashValue(face);
	}
	int index=9;
	for (; waitKey(10) != 27; cam >> frame) { //esc 的ascall码是27；
		/*cout << "you can press s to save" << endl;
		if (waitKey(2) == 83) {
			string save_path= format("E:/opencv_learn/opencv_test/face/%d.jpg", index);
			imwrite(save_path, frame);
			cout << "save image " << index << ".jpg success" << endl;
			index++;
		}
		cout << "time out" << endl;*/
		Mat image2=detectAndDisplay(frame);
		imshow("frame", image2);
	}
	//detectAndDisplay(image);// 检测人脸
	return 0;

}

Mat detectAndDisplay(Mat image)
{
	CascadeClassifier ccf;      //创建脸部对象
	ccf.load(xmlPath);           //导入opencv自带检测的文件
	vector<Rect> faces;
	Mat gray;
	//image *= 1. / 255;
	cvtColor(image, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	ccf.detectMultiScale(gray, faces, 1.1, 3, 0, Size(50, 50), Size(500, 500));
	//for (vector<Rect>::const_iterator iter = faces.begin(); iter != faces.end(); iter++)
	//{	
	//	//if (faces[i].width * faces[i].height < 10000) continue;
	//	cout << *iter << endl;
	//	Rect a = *iter;
	//	cout << a << endl;
	//	rectangle(image, *iter, Scalar(0, 0, 255), 2, 8); //画出脸部矩形
	//}
	Mat image1;
	
	for (size_t i = 0; i < faces.size(); i++)
	{	
		//cout << faces[i].width <<' '<<faces[i].height << endl;
		Point center(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		if (faces[i].width * faces[i].height < 10000) continue;
		rectangle(image, Point(faces[i].x, faces[i].y), center, Scalar(0, 0, 255), 2, 8);
		image1 = image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
		boxFilter(image1, image1, -1, Size(77,77));
		image1.copyTo(image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)));

		int diff[100];
		string str1 = HashValue(image1);
		for (int i = 1; i <= total_people_number; i++)//因为我完成的就是8张图片的检测，所以循环值为8
		{
			diff[i] = HanmingDistance(str1, str2[i]);
		}
		int min = 1000, t;
		for (int i = 1; i <= total_people_number; i++)    //循环值为8，求与原图片汉明距离最小的那张图片
		{
			cout << i << ' ' << diff[i] << endl;
			if (min > diff[i] && diff[i] != 0)
			{
				min = diff[i];
				t = i;
			}           //检测出的标记为t
		}
		string name[9] = {"chenli","faqian","lvcai" ,"jinshen" ,"minda" ,"langting" ,"xiuyu" ,"qilong" ,"wuyanzhu" };
	
		putText(image, name[t], Point(faces[i].x, faces[i].y), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255));
		//waitKey(0);
	}
	//imshow("1", image);
	return image;

}

string HashValue(Mat& src)      //得到图片的哈希值
//很久之前写的，现在想不起来了...注释就先不写了.....抱歉哈。但是是可以运行的
{
	string rst(256, '\0');
	Mat img;
	if (src.channels() == 3)
		cvtColor(src, img, CV_BGR2GRAY);
	else
		img = src.clone();
	resize(img, img, Size(16, 16));
	uchar* pData;
	for (int i = 0; i < img.rows; i++)
	{
		pData = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			pData[j] = pData[j] / 4;
		}
	}

	int average = mean(img).val[0];
	Mat mask = (img >= (uchar)average);
	int index = 0;
	for (int i = 0; i < mask.rows; i++)
	{
		pData = mask.ptr<uchar>(i);
		for (int j = 0; j < mask.cols; j++)
		{
			if (pData[j] == 0)
				rst[index++] = '0';
			else
				rst[index++] = '1';
		}
	}
	return rst;
}
int HanmingDistance(string& str1, string& str2)       //求两张图片的汉明距离
{
	if ((str1.size() != 256) || (str2.size() != 256))
		return -1;
	int diff = 0;
	for (int i = 0; i < 256; i++)
	{
		if (str1[i] != str2[i])
			diff++;
	}
	return diff;
}

