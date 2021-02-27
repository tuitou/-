#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat OriginalImg;

	OriginalImg = imread("E:/车牌识别2号/车牌2.jpg", IMREAD_COLOR);//读取原始彩色图像
	if (OriginalImg.empty())  //判断图像对否读取成功
	{
		cout << "错误!读取图像失败\n";
		return -1;
	}
	//imshow("原图", OriginalImg); //显示原始图像
	cout << "Width:" << OriginalImg.rows << "\tHeight:" << OriginalImg.cols << endl;//打印长宽

	Mat ResizeImg;
	if (OriginalImg.cols > 640)
		resize(OriginalImg, ResizeImg, Size(640, 640 * OriginalImg.rows / OriginalImg.cols));
	imshow("尺寸变换图", ResizeImg);

	unsigned char pixelB, pixelG, pixelR;  //记录各通道值
	unsigned char DifMax = 50;             //基于颜色区分的阈值设置
	unsigned char B = 138, G = 63, R = 23; //各通道的阈值设定，针对与蓝色车牌
	Mat BinRGBImg = ResizeImg.clone();  //二值化之后的图像
	int i = 0, j = 0;
	for (i = 0; i < ResizeImg.rows; i++)   //通过颜色分量将图片进行二值化处理
	{
		for (j = 0; j < ResizeImg.cols; j++)
		{
			pixelB = ResizeImg.at<Vec3b>(i, j)[0]; //获取图片各个通道的值
			pixelG = ResizeImg.at<Vec3b>(i, j)[1];
			pixelR = ResizeImg.at<Vec3b>(i, j)[2];

			if (abs(pixelB - B) < DifMax && abs(pixelG - G) < DifMax && abs(pixelR - R) < DifMax)
			{                                           //将各个通道的值和各个通道阈值进行比较
				BinRGBImg.at<Vec3b>(i, j)[0] = 255;     //符合颜色阈值范围内的设置成白色
				BinRGBImg.at<Vec3b>(i, j)[1] = 255;
				BinRGBImg.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				BinRGBImg.at<Vec3b>(i, j)[0] = 0;        //不符合颜色阈值范围内的设置为黑色
				BinRGBImg.at<Vec3b>(i, j)[1] = 0;
				BinRGBImg.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	imshow("基于颜色信息二值化", BinRGBImg);        //显示二值化处理之后的图像

	Mat BinOriImg;     //形态学处理结果图像
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //设置形态学处理窗的大小
	dilate(BinRGBImg, BinOriImg, element);     //进行多次膨胀操作
	dilate(BinOriImg, BinOriImg, element);
	dilate(BinOriImg, BinOriImg, element);

	erode(BinOriImg, BinOriImg, element);      //进行多次腐蚀操作
	erode(BinOriImg, BinOriImg, element);
	erode(BinOriImg, BinOriImg, element);
	imshow("形态学处理后", BinOriImg);        //显示形态学处理之后的图像
	
	double length, area, rectArea;     //定义轮廓周长、面积、外界矩形面积
	double rectDegree = 0.0;           //矩形度=外界矩形面积/轮廓面积
	double long2Short = 0.0;           //体态比=长边/短边
	CvRect rect;           //外界矩形
	CvBox2D box, boxTemp;  //外接矩形
	CvPoint2D32f pt[4];    //矩形定点变量
	double axisLong = 0.0, axisShort = 0.0;        //矩形的长边和短边
	double axisLongTemp = 0.0, axisShortTemp = 0.0;//矩形的长边和短边
	double LengthTemp;     //中间变量
	float  angle = 0;      //记录车牌的倾斜角度
	float  angleTemp = 0;
	bool   TestPlantFlag = 0;  //车牌检测成功标志位
	cvtColor(BinOriImg, BinOriImg, CV_BGR2GRAY);   //将形态学处理之后的图像转化为灰度图像
	threshold(BinOriImg, BinOriImg, 100, 255, THRESH_BINARY); //灰度图像二值化
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq * seq = 0;     //创建一个序列,CvSeq本身就是一个可以增长的序列，不是固定的序列
	CvSeq * tempSeq = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	int cnt = cvFindContours(&(IplImage(BinOriImg)), storage, &seq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//第一个参数是IplImage指针类型，将MAT强制转换为IplImage指针类型
	//返回轮廓的数目 
	//获取二值图像中轮廓的个数
	cout << "number of contours   " << cnt << endl;  //打印轮廓个数
	Mat roi_image;
	for (tempSeq = seq; tempSeq != NULL; tempSeq = tempSeq->h_next)
	{
		length = cvArcLength(tempSeq);       //获取轮廓周长
		area = cvContourArea(tempSeq);       //获取轮廓面积

		if (area > 800 && area < 50000)     //矩形区域面积大小判断
		{
			rect = cvBoundingRect(tempSeq, 1);//计算矩形边界
			boxTemp = cvMinAreaRect2(tempSeq, 0);  //获取轮廓的矩形
			cvBoxPoints(boxTemp, pt);              //获取矩形四个顶点坐标
			angleTemp = boxTemp.angle;                 //得到车牌倾斜角度

			axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //计算长轴（勾股定理）
			axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //计算短轴（勾股定理）

			if (axisShortTemp > axisLongTemp)   //短轴大于长轴，交换数据
			{
				LengthTemp = axisLongTemp;
				axisLongTemp = axisShortTemp;
				axisShortTemp = LengthTemp;
			}
			else
				angleTemp += 90;
			rectArea = axisLongTemp * axisShortTemp;  //计算矩形的面积
			rectDegree = area / rectArea;     //计算矩形度（比值越接近1说明越接近矩形）

			long2Short = axisLongTemp / axisShortTemp; //计算长宽比
			if (long2Short > 1 && long2Short < 5.5 && rectDegree > 0.53 && rectDegree < 1.37 && rectArea > 1000 && rectArea < 50000)
			{
				Mat srcImage = ResizeImg.clone();
				TestPlantFlag = true;             //检测车牌区域成功
				for (int i = 0; i < 4; ++i)       //划线框出车牌区域
					cvLine(&(IplImage(srcImage)), cvPointFrom32f(pt[i]), cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]), CV_RGB(255, 0, 0));
				imshow("提取车牌结果图", srcImage);

				roi_image = srcImage(rect);
				imshow("提取车牌结果图", roi_image);    //显示最终结果图
				box = boxTemp;
				angle = -angleTemp;
				axisLong = axisLongTemp;
				axisShort = axisShortTemp;
				cout << "倾斜角度：" << angle << endl;


				Mat large_image;
				int col = roi_image.cols, row = roi_image.rows;
				resize(roi_image, large_image, Size(300, 300 * row / col));
				imshow("test", large_image);

				Mat gray_img;
				// 生成灰度图像
				cvtColor(large_image, gray_img, CV_BGR2GRAY);
				// 高斯模糊
				Mat img_gau;
				GaussianBlur(gray_img, img_gau, Size(3, 3), 0, 0);
				// 阈值分割
				Mat img_seg;
				threshold(img_gau, img_seg, 0, 255, THRESH_BINARY + THRESH_OTSU);
				// 边缘检测，提取轮廓
				Mat img_canny;
				Canny(img_seg, img_canny, 200, 100);
				imshow("test1", img_canny);

				vector<vector<Point>> contours;
				vector<Vec4i> hierarchy;
				findContours(img_canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());
				int size = (int)(contours.size());
				// 保存符号边框的序号
				vector<int> num_order;
				map<int, int> num_map;
				for (int i = 0; i < size; i++) {
					// 获取边框数据
					Rect number_rect = boundingRect(contours[i]);
					int width = number_rect.width;
					int height = number_rect.height;
					// 去除较小的干扰边框，筛选出合适的区域
					if (width > large_image.cols / 10 && height > large_image.rows / 2) {
						rectangle(img_seg, number_rect.tl(), number_rect.br(), Scalar(255, 255, 255), 1, 1, 0);
						num_order.push_back(number_rect.x);
						num_map[number_rect.x] = i;
					}
				}
				// 按符号顺序提取
				sort(num_order.begin(), num_order.end());
				for (int i = 0; i < num_order.size(); i++) {
					Rect number_rect = boundingRect(contours[num_map.find(num_order[i])->second]);
					Rect choose_rect(number_rect.x, 0, number_rect.width, gray_img.rows);
					Mat number_img = gray_img(choose_rect);
					imshow("number" + to_string(i), number_img);
					// imwrite("number" + to_string(i) + ".jpg", number_img);
				}
				imshow("添加方框", gray_img);
				waitKey(0);
				return 0;
			
			

				/*Mat gray_img;
				// 生成灰度图像
				cvtColor(large_image, gray_img, CV_BGR2GRAY);
				// 高斯模糊
				Mat img_gau;
				GaussianBlur(gray_img, img_gau, Size(3, 3), 0, 0);
				// 阈值分割
				Mat img_threadhold;
				threshold(img_gau, img_threadhold, 0, 255, THRESH_BINARY + THRESH_OTSU);
				// 判断字符水平位置
				int roi_col = img_threadhold.cols, roi_row = img_threadhold.rows, position1[50], position2[50], roi_width[50];
				uchar pix;
				// 确认为 1 的像素
				int pixrow[1000];
				for (int i = 0; i < roi_col - 1; i++) {
					for (int j = 0; j < roi_row - 1; j++) {
						pix = img_threadhold.at<uchar>(j, i);
						pixrow[i] = 0;
						if (pix > 0) {
							pixrow[i] = 1;
							break;
						}
					}
				}
				// 对数组进行滤波，减少突变概率
				for (int i = 2; i < roi_col - 1 - 2; i++) {
					if ((pixrow[i - 1] + pixrow[i - 2] + pixrow[i + 1] + pixrow[i + 2]) >= 3) {
						pixrow[i] = 1;
					}
					else if ((pixrow[i - 1] + pixrow[i - 2] + pixrow[i + 1] + pixrow[i + 2]) <= 1) {
						pixrow[i] = 0;
					}
				}
				// 确认字符位置
				int count = 0;
				bool flage = false;
				for (int i = 0; i < roi_col - 1; i++) {
					pix = pixrow[i];
					if (pix == 1 && !flage) {
						flage = true;
						position1[count] = i;
						continue;
					}
					if (pix == 0 && flage) {
						flage = false;
						position2[count] = i;
						count++;
					}
					if (i == (roi_col - 2) && flage) {
						flage = false;
						position2[count] = i;
						count++;
					}
				}
				// 记录所有字符宽度
				for (int n = 0; n < count; n++) {
					roi_width[n] = position2[n] - position1[n];
				}
				// 减去最大值、最小值，计算平均值用字符宽度来筛选
				int max = roi_width[0], max_index = 0;
				int min = roi_width[0], min_index = 0;
				for (int n = 1; n < count; n++) {
					if (max < roi_width[n]) {
						max = roi_width[n];
						max_index = n;
					}
					if (min > roi_width[n]) {
						min = roi_width[n];
						min_index = n;
					}
				}
				int index = 0;
				int new_roi_width[50];
				for (int i = 0; i < count; i++) {
					if (i == min_index || i == max_index) {}
					else {
						new_roi_width[index] = roi_width[i];
						index++;
					}
				}
				// 取后面三个值的平均值
				int avgre = (int)((new_roi_width[count - 3] + new_roi_width[count - 4] + new_roi_width[count - 5]) / 3.0);
				// 字母位置信息确认，用宽度来筛选
				int licenseX[10], licenseW[10], licenseNum = 0;
				int countX = 0;
				for (int i = 0; i < count; i++) {
					if (roi_width[i] > (avgre - 8) && roi_width[i] < (avgre + 8)) {
						licenseX[licenseNum] = position1[i];
						licenseW[licenseNum] = roi_width[i];
						licenseNum++;
						countX++;
						continue;
					}
					if (roi_width[i] > (avgre * 2 - 10) && roi_width[i] < (avgre * 2 + 10)) {
						licenseX[licenseNum] = position1[i];
						licenseW[licenseNum] = roi_width[i];
						licenseNum++;
					}
				}

				// 截取字符
				Mat number_img = Mat(Scalar(0));
				for (int i = 0; i < countX; i++) {
					Rect choose_rect(licenseX[i], 0, licenseW[i], gray_img.rows);
					number_img = gray_img(choose_rect);
					imshow("number" + to_string(i), number_img);
					//imwrite("number" + to_string(i) + ".jpg", number_img);
				}
				imshow("添加方框", gray_img);
				waitKey();
				return 0;*/




			}
		}
	}
}


