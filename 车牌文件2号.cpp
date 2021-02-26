#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat OriginalImg;

	OriginalImg = imread("E:/����ʶ��2��/����1.jpg", IMREAD_COLOR);//��ȡԭʼ��ɫͼ��
	if (OriginalImg.empty())  //�ж�ͼ��Է��ȡ�ɹ�
	{
		cout << "����!��ȡͼ��ʧ��\n";
		return -1;
	}
	//imshow("ԭͼ", OriginalImg); //��ʾԭʼͼ��
	cout << "Width:" << OriginalImg.rows << "\tHeight:" << OriginalImg.cols << endl;//��ӡ����

	Mat ResizeImg;
	if (OriginalImg.cols > 640)
	resize(OriginalImg, ResizeImg, Size(640, 640 * OriginalImg.rows / OriginalImg.cols));
	imshow("�ߴ�任ͼ", ResizeImg);

	unsigned char pixelB, pixelG, pixelR;  //��¼��ͨ��ֵ
	unsigned char DifMax = 50;             //������ɫ���ֵ���ֵ����
	unsigned char B = 138, G = 63, R = 23; //��ͨ������ֵ�趨���������ɫ����
	Mat BinRGBImg = ResizeImg.clone();  //��ֵ��֮���ͼ��
	int i = 0, j = 0;
	for (i = 0; i < ResizeImg.rows; i++)   //ͨ����ɫ������ͼƬ���ж�ֵ������
	{
		for (j = 0; j < ResizeImg.cols; j++)
		{
			pixelB = ResizeImg.at<Vec3b>(i, j)[0]; //��ȡͼƬ����ͨ����ֵ
			pixelG = ResizeImg.at<Vec3b>(i, j)[1];
			pixelR = ResizeImg.at<Vec3b>(i, j)[2];

			if (abs(pixelB - B) < DifMax && abs(pixelG - G) < DifMax && abs(pixelR - R) < DifMax)
			{                                           //������ͨ����ֵ�͸���ͨ����ֵ���бȽ�
				BinRGBImg.at<Vec3b>(i, j)[0] = 255;     //������ɫ��ֵ��Χ�ڵ����óɰ�ɫ
				BinRGBImg.at<Vec3b>(i, j)[1] = 255;
				BinRGBImg.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				BinRGBImg.at<Vec3b>(i, j)[0] = 0;        //��������ɫ��ֵ��Χ�ڵ�����Ϊ��ɫ
				BinRGBImg.at<Vec3b>(i, j)[1] = 0;
				BinRGBImg.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	imshow("������ɫ��Ϣ��ֵ��", BinRGBImg);        //��ʾ��ֵ������֮���ͼ��

	Mat BinOriImg;     //��̬ѧ������ͼ��
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //������̬ѧ�����Ĵ�С
	dilate(BinRGBImg, BinOriImg, element);     //���ж�����Ͳ���
	dilate(BinOriImg, BinOriImg, element);
	dilate(BinOriImg, BinOriImg, element);

	erode(BinOriImg, BinOriImg, element);      //���ж�θ�ʴ����
	erode(BinOriImg, BinOriImg, element);
	erode(BinOriImg, BinOriImg, element);
	imshow("��̬ѧ�����", BinOriImg);        //��ʾ��̬ѧ����֮���ͼ��

	double length, area, rectArea;     //���������ܳ�����������������
	double rectDegree = 0.0;           //���ζ�=���������/�������
	double long2Short = 0.0;           //��̬��=����/�̱�
	CvRect rect;           //������
	CvBox2D box, boxTemp;  //��Ӿ���
	CvPoint2D32f pt[4];    //���ζ������
	double axisLong = 0.0, axisShort = 0.0;        //���εĳ��ߺͶ̱�
	double axisLongTemp = 0.0, axisShortTemp = 0.0;//���εĳ��ߺͶ̱�
	double LengthTemp;     //�м����
	float  angle = 0;      //��¼���Ƶ���б�Ƕ�
	float  angleTemp = 0;
	bool   TestPlantFlag = 0;  //���Ƽ��ɹ���־λ
	cvtColor(BinOriImg, BinOriImg, CV_BGR2GRAY);   //����̬ѧ����֮���ͼ��ת��Ϊ�Ҷ�ͼ��
	threshold(BinOriImg, BinOriImg, 100, 255, THRESH_BINARY); //�Ҷ�ͼ���ֵ��
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq * seq = 0;     //����һ������,CvSeq�������һ���������������У����ǹ̶�������
	CvSeq * tempSeq = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	int cnt = cvFindContours(&(IplImage(BinOriImg)), storage, &seq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//��һ��������IplImageָ�����ͣ���MATǿ��ת��ΪIplImageָ������
	//������������Ŀ 
	//��ȡ��ֵͼ���������ĸ���
	cout << "number of contours   " << cnt << endl;  //��ӡ��������
	for (tempSeq = seq; tempSeq != NULL; tempSeq = tempSeq->h_next)
	{
		length = cvArcLength(tempSeq);       //��ȡ�����ܳ�
		area = cvContourArea(tempSeq);       //��ȡ�������
		if (area > 800 && area < 50000)     //�������������С�ж�
		{
			rect = cvBoundingRect(tempSeq, 1);//������α߽�
			boxTemp = cvMinAreaRect2(tempSeq, 0);  //��ȡ�����ľ���
			cvBoxPoints(boxTemp, pt);              //��ȡ�����ĸ���������
			angleTemp = boxTemp.angle;                 //�õ�������б�Ƕ�

			axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //���㳤�ᣨ���ɶ���
			axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //������ᣨ���ɶ���

			if (axisShortTemp > axisLongTemp)   //������ڳ��ᣬ��������
			{
				LengthTemp = axisLongTemp;
				axisLongTemp = axisShortTemp;
				axisShortTemp = LengthTemp;
			}
			else
				angleTemp += 90;
			rectArea = axisLongTemp * axisShortTemp;  //������ε����
			rectDegree = area / rectArea;     //������ζȣ���ֵԽ�ӽ�1˵��Խ�ӽ����Σ�

			long2Short = axisLongTemp / axisShortTemp; //���㳤���
			if (long2Short > 1 && long2Short < 5.5 && rectDegree > 0.53 && rectDegree < 1.37 && rectArea > 1000 && rectArea < 50000)
			{
				Mat GuiRGBImg = ResizeImg.clone();
				TestPlantFlag = true;             //��⳵������ɹ�
				for (int i = 0; i < 4; ++i)       //���߿����������
					cvLine(&(IplImage(GuiRGBImg)), cvPointFrom32f(pt[i]), cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]), CV_RGB(255, 0, 0));
				imshow("��ȡ���ƽ��ͼ", GuiRGBImg);    //��ʾ���ս��ͼ

				box = boxTemp;
				angle = angleTemp;
				axisLong = axisLongTemp;
				axisShort = axisShortTemp;
				cout << "��б�Ƕȣ�" << angle << endl;
			}
		}
	}

	waitKey();
	return 0;

}


