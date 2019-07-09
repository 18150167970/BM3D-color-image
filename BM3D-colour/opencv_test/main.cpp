#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\imgproc\types_c.h>
#include <time.h>
using namespace cv;
using namespace std;


void addNoise(const int sigma, const Mat origin, Mat& noisy);
void uchar2float(const Mat tyuchar, Mat& tyfloat);
void float2uchar(const Mat tyfloat, Mat& tyuchar);
float cal_psnr(const Mat x, const Mat y);

int runBm3d(const int sigma, const Mat image_noisy,
	Mat& image_basic, Mat& image_denoised);

void getPatches(const Mat img, const Mat img_U, const Mat img_V, const int width, const int height, const int channels,
	const int patchSize, const int step, vector<Mat>& block, vector<Mat>& block_U, vector<Mat>& block_V, vector<int>& row_idx, vector<int>& col_idx);

void tran2d(vector<Mat>& input);
int log2(const int N);

void getSimilarPatch(const vector<Mat> block, const vector<Mat> block_U, const vector<Mat> block_V,
	vector<Mat>& sim_patch, vector<Mat>& sim_patch_U, vector<Mat>& sim_patch_V, vector<int>& sim_num,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao);

float cal_distance(const Mat a, const Mat b);

void tran1d(vector<Mat>& input);

float shrink(vector<Mat>& input, float threshhold, int sigma);

float calculate_weight_hd(const vector<Mat>input, int sigma);
float calculate_weight_wien(const vector<Mat>input, int sigma);

void inv_tran_3d(vector<Mat>& inpute);

void aggregation(Mat& numerator, Mat& denominator, Mat& numerator_U, Mat& denominator_U, Mat& numerator_V, Mat& denominator_V, vector<int>idx_r, vector<int>idx_c,
	const vector<Mat> input, const vector<Mat> input_U, const vector<Mat> input_V, float weight, int patchSize, Mat window);

void gen_wienFilter(vector<Mat>& input, int sigma);

void wienFiltering(vector<Mat>& input, const vector<Mat>wien, int patchSize);

Mat gen_kaiser(int beta, int length);
void wavedec(float* input, int length);
void waverec(float* input, int length, int N);



int main()
{
	string path = "house.png";//�Լ��ͼƬ1.jpgΪ��
	Mat image = imread("draw.jpg", 1);
	if (!image.data) {
		cout << "image don't exist, please check your image path" << endl;
			return -1;
	}
	resize(image, image, Size(0, 0), 0.3,0.3);
	Mat Pic(image.size(), CV_32FC3);
	Mat Noisy(image.size(), CV_32FC3);
	Mat Basic(image.size(), CV_32FC3);
	Mat Denoised(image.size(), CV_32FC3);

	Mat basic(image.size(), CV_8UC3);
	Mat noisy(image.size(), CV_8UC3);
	Mat denoised(image.size(), CV_8UC3);

	int sigma = 25;

	//imread() ��imshow()������������ʹ��uchar��ʽ��CV_8U�����Խ��д����ʱ����Ҫ�Ѹ�ʽת����float�ſ��Խ��в�����������ɺ���Ҫת������
	//���ܹ���ʾ����Ȼֱ����ʾfloat����ȫ��ͼ��
	uchar2float(image, Pic);

	//���뺯��
	addNoise(sigma, Pic, Noisy);
	
	float2uchar(Noisy, noisy);

	//imshow("orgin", image);
	//imshow("noise", noisy);
	//waitKey(0);


	//ʱ��
	double start, stop, duration;
	start = clock();
	runBm3d(sigma, Noisy, Basic, Denoised);//main denoising method
	stop = clock();
	duration = double(stop - start)/1000;
	cout << "denoised time of use noise:" << duration << " s" << endl;

	float2uchar(Basic, basic);
	float2uchar(Denoised, denoised);
	namedWindow("basic", 1);

	imshow("orgin", image);
	imshow("noise", noisy);
	imshow("basic", basic);
	imshow("denoised", denoised);

	waitKey(0);
	

	return 0;
}

int runBm3d(const int sigma, const Mat image_noisy, Mat& image_basic, Mat& image_denoised) {

	const unsigned nHard = 39;//search area
	const unsigned nWien = 39;
	const unsigned kHard = 8;//patch size
	const unsigned kWien = 8;
	const unsigned NHard = 16;//max number
	const unsigned NWien = 32;
	const unsigned pHard = 3;//step
	const unsigned pWien = 3;

	const int tao_hard = 2500;
	const int tao_wien = 400;

	int beta = 2;
	float lambda3d = 2.7;
	float lambda2d = 0;

	int Height = image_noisy.rows;
	int Width = image_noisy.cols;
	int Channels = image_noisy.channels();

	vector<Mat> block_noisy_Y;
	vector<Mat> block_noisy_U;
	vector<Mat> block_noisy_V;
	vector<int> row_idx;
	vector<int> col_idx;


	//�ָ�ͨ��
	Mat Noisy(image_noisy.size(), CV_32FC3);
	cvtColor(image_noisy, Noisy, CV_BGR2YUV);
	vector<Mat> hsv_vec;
	split(Noisy, hsv_vec);

	Mat image_noisy_Y(Noisy.size(), CV_32FC1);
	Mat image_noisy_U(Noisy.size(), CV_32FC1);
	Mat image_noisy_V(Noisy.size(), CV_32FC1);

	image_noisy_Y = hsv_vec[0].clone();
	image_noisy_U = hsv_vec[1].clone();
	image_noisy_V = hsv_vec[2].clone();

	getPatches(image_noisy_Y, image_noisy_U, image_noisy_V, Width, Height, Channels, 
		kHard, pHard, block_noisy_Y, block_noisy_U, block_noisy_V, row_idx, col_idx);

	tran2d(block_noisy_Y);
	//tran2d(block_noisy_U);
	//tran2d(block_noisy_V);

	Mat kaiser = gen_kaiser(beta, kHard);

	int bn_r = row_idx.size();
	int bn_c = col_idx.size();

	vector<int> sim_num;//index number for the selected similar patch in the block vector
	vector<int> sim_idx_row;//index number for the selected similar patch in the original Mat
	vector<int> sim_idx_col;

	vector<Mat>data_Y;
	vector<Mat>data_U;
	vector<Mat>data_V;//store the data during transforming and shrinking

	float weight_hd = 1.0;//weights used for current relevent patch
	Mat denominator_hd(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_hd(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat denominator_hd_U(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_hd_U(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat denominator_hd_V(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_hd_V(image_noisy.size(), CV_32FC1, Scalar::all(0));

	for (int i = 0; i < bn_r; i++) {
		for (int j = 0; j < bn_c; j++) {
			sim_num.clear();
			sim_idx_col.clear();
			sim_idx_row.clear();
			data_Y.clear();
			data_U.clear();
			data_V.clear();

			getSimilarPatch(block_noisy_Y, block_noisy_V, block_noisy_U,data_Y, data_U, data_V, sim_num,
				i, j, bn_r, bn_c, int((nHard - kHard) / pHard) + 1, NHard, tao_hard);//block matching
			
			for (int k = 0; k < sim_num.size(); k++) {
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
			}

			tran1d(data_Y);
			weight_hd = shrink(data_Y, lambda3d * sigma, sigma);
			//weight_hd = calculate_weight_hd(data, sigma);
			inv_tran_3d(data_Y);//3-D inverse transforming

			aggregation(numerator_hd, denominator_hd, numerator_hd_U, denominator_hd_U, numerator_hd_V, denominator_hd_V, sim_idx_row, sim_idx_col, data_Y, data_U, data_V, weight_hd, kHard, kaiser);//aggregation using weigths

		}
	}
	Mat image_basic_Y(image_basic.size(), CV_32FC1);
	Mat image_basic_U(image_basic.size(), CV_32FC1);
	Mat image_basic_V(image_basic.size(), CV_32FC1);

	image_basic_Y = numerator_hd / denominator_hd;
	image_basic_U = numerator_hd_U / denominator_hd_U;
	image_basic_V = numerator_hd_V / denominator_hd_V;

	vector<Mat> hsv_vec_merge;
	hsv_vec_merge.push_back(image_basic_Y);
	hsv_vec_merge.push_back(image_basic_U);
	hsv_vec_merge.push_back(image_basic_V);

	merge(hsv_vec_merge,image_basic);
	cvtColor(image_basic, image_basic, CV_YUV2RGB);

	//step 2 wiena filtering
	vector<Mat> block_basic_Y;
	vector<Mat> block_basic_U;
	vector<Mat> block_basic_V;

	row_idx.clear();
	col_idx.clear();

	//��û�������ȥ��ͼ���
	getPatches(image_basic_Y, image_basic_U, image_basic_V, Width, Height, Channels,
		kHard, pHard, block_basic_Y, block_basic_U, block_basic_V, row_idx, col_idx);

	bn_r = row_idx.size();
	bn_c = col_idx.size();

	vector<Mat> data_noisy;
	vector<Mat> data_noisy_U;
	vector<Mat> data_noisy_V;

	float weight_wien = 1.0;//weights used for current relevent patch
	Mat denominator_wien(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_wien(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat denominator_wien_U(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_wien_U(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat denominator_wien_V(image_noisy.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_wien_V(image_noisy.size(), CV_32FC1, Scalar::all(0));

	for (int i = 0; i < bn_r; i++)
	{
		for (int j = 0; j < bn_c; j++)
		{
			//for each pack in the basic estimate
			sim_num.clear();
			sim_idx_row.clear();
			sim_idx_col.clear();

			data_Y.clear();
			data_U.clear();
			data_V.clear();

			data_noisy.clear();
			data_noisy_U.clear();
			data_noisy_V.clear();

			//���ڻ������Ƽ������������㣬���ں���ԭͼ��ÿ��Ŀ��ͼ�飬����ֱ���ö�Ӧ��������ͼ���ŷ�Ͼ���������Ƴ̶ȡ�
			//�������С���������ȡ���ǰMAXN1��������������ͼ�顢����ԭͼͼ��ֱ����������ά���顣
			getSimilarPatch(block_basic_Y, block_basic_V, block_basic_U, data_Y, data_U, data_V, sim_num,
				i, j, bn_r, bn_c, int((nHard - kHard) / pHard) + 1, NHard, tao_hard);//block matching

			for (int k = 0; k < sim_num.size(); k++)//calculate idx in the left-top corner
			{
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
				data_noisy.push_back(image_noisy_Y(Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
				data_noisy_U.push_back(image_noisy_U(Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
				data_noisy_V.push_back(image_noisy_V(Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
			}

			//�Ժ���������3D����ĵ���ά����ͼ���������ÿ��ͼ��ͬһ��λ�õ����ص㹹�ɵ����飬����DCT�任���������¹�ʽ�õ�ϵ��
			tran2d(data_Y);
			tran2d(data_noisy);
			tran1d(data_Y);
			tran1d(data_noisy);

			gen_wienFilter(data_Y, sigma);
			weight_wien = calculate_weight_wien(data_Y, sigma);

			//��ϵ���뺬��3Dͼ����˷Ż�ԭ�����������Ȩƽ���������ɵõ����չ���ͼ������ڻ�������ͼ����ԭ�˸���ԭͼ��ϸ�ڡ�
			wienFiltering(data_noisy, data_Y, kWien);

			inv_tran_3d(data_noisy);

			aggregation(numerator_wien, denominator_wien, numerator_wien_U, denominator_wien_U, numerator_wien_V, denominator_wien_V,
				sim_idx_row, sim_idx_col, data_noisy, data_noisy_U, data_noisy_V, weight_wien, kHard, kaiser);
		}
	}


	image_basic_Y = numerator_wien / denominator_wien;
	image_basic_U = numerator_wien_U / denominator_wien_U;
	image_basic_V = numerator_wien_V / denominator_wien_V;

	hsv_vec_merge.clear();
	hsv_vec_merge.push_back(image_basic_Y);
	hsv_vec_merge.push_back(image_basic_U);
	hsv_vec_merge.push_back(image_basic_V);

	merge(hsv_vec_merge, image_denoised);
	cvtColor(image_denoised, image_denoised, CV_YUV2BGR);

	return EXIT_SUCCESS;
}

//ͳ�Ʒ���ɷֵ�������Ϊ����Ȩ�صĲο�
float calculater_weight_hd(const vector<Mat> input, int sigma) {
	int num = 0;
	for (int k = 0; k < input.size(); k++) {
		for (int i = 0; i < input[k].rows; i++)
		{
			for (int j = 0; j < input[k].cols; j++)
			{
				if (input[k].at<float>(i, j) != 0)
				{
					num++;
				}
			}
		}
	}
	if (num == 0)
		return 1;
	else
		return 1.0 / (sigma * sigma * num);
}


//��ͼ��鼯��������ֵС����ֵ���������� //ͳ�Ʒ���ɷֵ�������Ϊ����Ȩ�صĲο�,�������Խ�࣬Ȩ��Խ�͡�
float shrink(vector<Mat>& input, float threshold, int sigma) {
	int num = 0;
	for (int k = 0; k < input.size(); k++) {
		for (int i = 0; i < input[k].rows; i++) {
			for (int j = 0; j < input[k].cols; j++) {
				//fabs������float��double�����ֵ
				if (fabs(input[k].at<float>(i, j))< threshold) {
					input[k].at<float>(i, j) = 0;
				}
				//ͳ�Ʒ���ɷֵ�������Ϊ����Ȩ�صĲο�
				if (input[k].at<float>(i, j) != 0)
				{
					num++;
				}
			}
		}
	}
	if (num == 0)
		return 1;
	else
		return 1.0 / (sigma * sigma * num);
}

//��3D����ĵ���ά����ͼ���������ÿ��ͼ��ͬһ��λ�õ����ص㹹�ɵ����飬����haarһάС���ֽ�
void tran1d(vector<Mat>& input) {
	int size = input.size();
	int layer = log2(size);
	float* data = new float[size];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			for (int k = 0; k < size; k++) {
				data[k] = input[k].at<float>(i, j);
			}
			wavedec(data, size);
			for (int k = 0; k < size; k++) {
				input[k].at<float>(i, j) = data[k];
			}
		}
	}
	delete[] data;
}

//���任������haarһάС���ֽⷴ�任��DCT���任��
void inv_tran_3d(vector<Mat>& input)
{
	int patchSize = 8;

	Mat tmp;
	int size = input.size();
	int layer = log2(size);
	float* data = new float[size];
	//С���ֽⷴ�任
	for (int i = 0; i < patchSize; i++)
		for (int j = 0; j < patchSize; j++)
		{
			for (int k = 0; k < size; k++)
			{
				data[k] = input[k].at<float>(i, j);
			}
			waverec(data, 2, layer);
			for (int k = 0; k < size; k++)
			{
				input[k].at<float>(i, j) = data[k];
			}
		}

	//DCT���任
	for (int k = 0; k < size; k++)
	{
		tmp = input[k].clone();
		dct(tmp, input[k], DCT_INVERSE);
	}
}

//С���ֽⷴ�任
void waverec(float* input, int length, int N)
{
	if (log2(length) > N) return;
	float* tmp = new float[length];
	for (int i = 0; i < length; i++) {
		tmp[i] = input[i];
	}
	for (int k = 0; k < length / 2; k++)
	{
		input[2 * k] = (tmp[k] + tmp[k + length / 2]) / sqrt(2);
		input[2 * k + 1] = (tmp[k] - tmp[k + length / 2]) / sqrt(2);
	}
	delete tmp;
	waverec(input, length * 2, N);
}

//haarһάС���ֽ�
void wavedec(float* input, int length)
{
	int N = log2(length);
	if (N == 0) return;
	float* tmp = new float[length];
	for (int i = 0; i < length; i++) {
		tmp[i] = input[i];
	}
	for (int k = 0; k < length / 2; k++)
	{
		input[k] = (tmp[2 * k] + tmp[2 * k + 1]) / sqrt(2);
		input[k + length / 2] = (tmp[2 * k] - tmp[2 * k + 1]) / sqrt(2);
	}
	delete[] tmp;
	wavedec(input, length / 2);
	return;
}


//�ֿ麯�����ڱ�������ʹ�õ���8size�Ŀ飬���Ϊ3�����Կ����֮�����ص����֡�
void getPatches(const Mat img, const Mat img_U, const Mat img_V, const int width, const int height, const int channels,
	const int patchSize, const int step, vector<Mat>& block, vector<Mat>& block_U, vector<Mat>& block_V, vector<int>& row_idx, vector<int>& col_idx) {
	Mat temp(patchSize, patchSize, CV_32FC1);

	//ÿ��step�����һ����ţ�����height - patchSizeλ�û��һ����š�
	for (int i = 0; i <= height - patchSize; i += step) {
		row_idx.push_back(i);
	}
	if ((height - patchSize) % step != 0)
	{
		row_idx.push_back(height - patchSize);
	}
	for (int j = 0; j <= width - patchSize; j += step)
	{
		col_idx.push_back(j);
	}
	if ((width - patchSize) % step != 0)
	{
		col_idx.push_back(width - patchSize);
	}

	//��ø�λ�õ�ͼ��顣
	for (int i = 0; i < row_idx.size(); i++) {
		for (int j = 0; j < col_idx.size(); j++) {
			temp = img(Rect(col_idx[j], row_idx[i], patchSize, patchSize));
			block.push_back(temp);
			temp = img_U(Rect(col_idx[j], row_idx[i], patchSize, patchSize));
			block_U.push_back(temp);
			temp = img_V(Rect(col_idx[j], row_idx[i], patchSize, patchSize));
			block_V.push_back(temp);
		}
	}
}

//����Щͼ����任��Ż�ԭλ�����÷���ɷ�����ͳ�Ƶ���Ȩ�أ���󽫵��ź��ͼ����ÿ�����Ȩ�ؾ͵õ��������Ƶ�ͼ��
void aggregation(Mat& numerator, Mat& denominator, Mat& numerator_U, Mat& denominator_U, Mat& numerator_V, Mat& denominator_V, vector<int>idx_r, vector<int>idx_c,
	const vector<Mat> input, const vector<Mat> input_U, const vector<Mat> input_V, float weight, int patchSize, Mat window)
{
	Rect rect;
	for (int k = 0; k < input.size(); k++)
	{
		rect.x = idx_c[k]; 
		rect.y = idx_r[k];
		rect.height = patchSize;
		rect.width = patchSize;

		numerator(rect) = numerator(rect) + weight * (input[k].mul(window));
		denominator(rect) = denominator(rect) + weight * window;

		numerator_U(rect) = numerator_U(rect) + weight * (input_U[k].mul(window));
		denominator_U(rect) = denominator_U(rect) + weight * window;

		numerator_V(rect) = numerator_V(rect) + weight * (input_V[k].mul(window));
		denominator_V(rect) = denominator_V(rect) + weight * window;
	}
}

struct distance_sort {
	float distance;
	int idx;
};

bool cmp(distance_sort a, distance_sort b) {
	return a.distance < b.distance;
}

void getSimilarPatch(const vector<Mat> block, const vector<Mat> block_U, const vector<Mat> block_V, 
	vector<Mat>& sim_patch, vector<Mat>& sim_patch_U, vector<Mat>& sim_patch_V, vector<int>& sim_num,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao) {

	//�ڸ����������Ҹ�5��ľ�������ʶ�飬�ܹ�121�顣
	int row_min = max(0, i - (area - 1) / 2);
	int row_max = min(bn_r - 1, i + (area - 1) / 2);
	int row_length = row_max - row_min + 1;

	int col_min = max(0, j - (area - 1) / 2);
	int col_max = min(bn_c - 1, j + (area - 1) / 2);
	int col_length = col_max - col_min + 1;

	const Mat relevence = block[i * bn_c + j];
	Mat temp;

	distance_sort* dis = new distance_sort[row_length * col_length];

	if (!dis) {
		cout << "allocation failure\n";
		system("pause");
	}

	//�����ٽ���ŷʽ���벢�洢
	for (int p = 0; p < row_length; p++)
	{
		for (int q = 0; q < col_length; q++)
		{
			temp = block[(p + row_min) * bn_c + (q + col_min)];
			dis[p * col_length + q].distance = cal_distance(relevence, temp);
			dis[p * col_length + q].idx = p * col_length + q;
		}
	}
	
	sort(dis, dis + row_length * col_length, cmp);

	int selectedNum = maxNum;
	while (row_length * col_length < selectedNum) //ѡ�����ҪС���ܸ���
	{
		selectedNum /= 2;//ȷ�����ƿ�ĸ���Ϊ2����
	}
	while (dis[selectedNum - 1].distance > tao) //��������Զ��ߵĻ�����ȡ���ٵĿ顣
	{
		selectedNum /= 2;
	}

	int Row, Col;
	for (int k = 0; k < selectedNum; k++)
	{
		Row = row_min + dis[k].idx / col_length;//p+row_min, patch����ԭͼ��λ�õ�row
		Col = col_min + dis[k].idx % col_length;//q+col_min��patch����ԭͼ��λ�õ�col

		temp = block[Row * bn_c + Col].clone();
		sim_patch.push_back(temp);

		temp = block_U[Row * bn_c + Col].clone();
		sim_patch_U.push_back(temp);

		temp = block_V[Row * bn_c + Col].clone();
		sim_patch_V.push_back(temp);

		sim_num.push_back(Row * bn_c + Col);//patch��ԭͼ�ı��
		//cout << dis[k].idx<<' '<< Row * bn_c + Col << endl;
	}

}

float cal_distance(const Mat relevence, Mat temp) {
	int rows = relevence.rows;
	int cols = relevence.cols;
	float sum = 0;
	for (int i = 0; i < rows; i++) {
		const float* M1 = relevence.ptr<float>(i);
		const float* M2 = temp.ptr<float>(i);
		for (int j = 0; j < cols; j++) {
			sum = (M1[j] - M2[j]) * (M1[j] - M2[j]);
		}
	}
	return sum/(rows*cols);
}

//DCT 2D�任
void tran2d(vector<Mat>& input) {
	Mat tmp;
	for (int i = 0; i < input.size(); i++) {
		dct(input[i], tmp);
		input[i] = tmp.clone();
	}
}

Mat gen_kaiser(int beta, int length)//How to do this?
{
	if ((beta == 2) && (length == 8))
	{
		Mat window(length, length, CV_32FC1);
		Mat kai1(length, 1, CV_32FC1);
		Mat kai1_T(1, length, CV_32FC1);
		kai1.at<float>(0, 0) = 0.4387;
		kai1.at<float>(1, 0) = 0.6813;
		kai1.at<float>(2, 0) = 0.8768;
		kai1.at<float>(3, 0) = 0.9858;
		for (int i = 0; i < 4; i++)
		{
			kai1.at<float>(7 - i, 0) = kai1.at<float>(i, 0);
			kai1_T.at<float>(0, i) = kai1.at<float>(i, 0);
			kai1_T.at<float>(0, 7 - i) = kai1.at<float>(i, 0);
		}
		window = kai1 * kai1_T;
		return window;
	}
}

//uchar ת��Ϊ float����������ת������ʹ�����鸳ֵ�����������ظ�ֵ����Ȼ�������й����лᱨ��
void uchar2float(const Mat tyuchar, Mat& tyfloat)
{
	for (int i = 0; i < tyuchar.rows; i++)
	{
		const uchar* ty1 = tyuchar.ptr<uchar>(i);
		float* ty2 = tyfloat.ptr<float>(i);
		for (int j = 0; j < tyuchar.cols; j++)
		{
			//ty2[j] = ty1[j];
			const uchar* dataWarpCol1 = ty1 + j * tyuchar.channels();
			float* dataWarpCol2 = ty2 + j * tyfloat.channels();
			dataWarpCol2[0] = dataWarpCol1[0];
			dataWarpCol2[1] = dataWarpCol1[1];
			dataWarpCol2[2] = dataWarpCol1[2];
		}
	}
}


//float ת��Ϊ uchar
void float2uchar(const Mat tyfloat, Mat& tyuchar)
{
	for (int i = 0; i < tyfloat.rows; i++) {
		const float* ty1 = tyfloat.ptr<float>(i);
		uchar* ty2 = tyuchar.ptr<uchar>(i);
		for (int j = 0; j < tyfloat.cols; j++) {
			uchar* dataWarpCol2 = ty2 + j * tyuchar.channels();
			const float* dataWarpCol1 = ty1 + j * tyfloat.channels();

			if (dataWarpCol1[0] < 0) dataWarpCol2[0] = 0;
			else if (dataWarpCol1[0] > 255) dataWarpCol2[0] = 255;
			else dataWarpCol2[0] = dataWarpCol1[0];

			if (dataWarpCol1[1] < 0) dataWarpCol2[1] = 0;
			else if (dataWarpCol1[1] > 255) dataWarpCol2[1] = 255;
			else dataWarpCol2[1] = dataWarpCol1[1];

			if (dataWarpCol1[2] < 0) dataWarpCol2[2] = 0;
			else if (dataWarpCol1[2] > 255) dataWarpCol2[2] = 255;
			else dataWarpCol2[2] = dataWarpCol1[2];
		}
	}
}


//ʵ��������������ʹ�õ�ʱ���ʹ��������Ӹ��Ӻ�ʱ���������ʹ��������ӡ�
void addNoise(const int sigma, const Mat origin, Mat& noisy) {
	Mat noise(origin.size(), CV_32FC3);
	//�����
	randn(noise, Scalar::all(0), Scalar::all(sigma));
	//�������
	noisy = noise + origin;
	//�������
	/*for (int i = 0; i < noise.rows; i++)
	{
		const float* Mx = origin.ptr<float>(i);
		float* Mn = noise.ptr<float>(i);
		float* My = noisy.ptr<float>(i);
		for (int j = 0; j < origin.cols; j++)
		{
			My[j] = Mx[j] + Mn[j];
		}
	}*/
}


//��ֵ����ȣ�PSNR����Խ��Խ��
float cal_psnr(const Mat x, const Mat y)
{
	float RMS = 0;
	for (int i = 0; i < x.rows; i++)
	{
		const float* Mx = x.ptr<float>(i);
		const float* My = y.ptr<float>(i);
		for (int j = 0; j < x.cols; j++)
		{
			RMS += (My[j] - Mx[j]) * (My[j] - Mx[j]);
		}
	}
	//���ΪMat��ʽ����ÿ��ͨ���ͣ����Բ���ֱ����sum;
	//RMS = sum((x - y) * (x - y));
	RMS = sqrtf(RMS / (x.rows * x.cols));
	return 20 * log10f(255.0 / RMS);
}

void gen_wienFilter(vector<Mat>& input, int sigma)
{
	Mat tmp;
	Mat Sigma(input[0].size(), CV_32FC1, Scalar::all(sigma * sigma));
	for (int k = 0; k < input.size(); k++)
	{
		tmp = input[k].mul(input[k]) + Sigma;
		input[k] = input[k].mul(input[k]) / (tmp.clone());
	}
}

void wienFiltering(vector<Mat>& input, const vector<Mat>wien, int patchSize)
{
	for (int k = 0; k < input.size(); k++)
	{
		input[k] = input[k].mul(wien[k]);
	}
}

float calculate_weight_wien(const vector<Mat>input, int sigma)
{
	float sum = 0;
	for (int k = 0; k < input.size(); k++)
		for (int i = 0; i < input[k].rows; i++)
			for (int j = 0; j < input[k].cols; j++)
			{
				sum += (input[k].at<float>(i, j)) * (input[k].at<float>(i, j));
			}
	return 1.0 / (sigma * sigma * sum);
}

//��ϵͳ�Դ��Ļ����,��֪��ʲôԭ��
int log2(const int N)
{
	int k = 1;
	int n = 0;
	while (k < N)
	{
		k *= 2;
		n++;
	}
	return n;
}
