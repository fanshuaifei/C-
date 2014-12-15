// BoWClassifierImg.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "stdlib.h"
#include "stdio.h"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <windows.h>

using namespace std;
using namespace cv;



/***************************************************************
* ����������Ͳ�������
*****************************************************************/
struct FDMParams
{
	string featureDetectorType; //�������������
	string descriptorType; //��������������
	string matcherType; //����ƥ�䷽������
	FDMParams() :featureDetectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce")
	{
	}
	FDMParams(string _featureDetectorType, string _descriptorType, string _matcherType)
	{
		featureDetectorType = _featureDetectorType;
		descriptorType = _descriptorType;
		matcherType = _matcherType;
	}
	void printMessage()
	{
		cout << "feature detector type is : " << featureDetectorType << endl;
		cout << "descriptor extractor type is: " << descriptorType << endl;
		cout << "matcher type is : " << matcherType << endl<<endl;
	}
};



/***************************************************************
* �����ʵ��������
*****************************************************************/
struct constructVocabularyParams
{
	int clusterCount;//K��ֵ�������ĵĸ��������ʵ��е��ʵĸ���
	float descriptorFraction;//���ʹ��ÿ��ͼ�������ӵĸ�������
	constructVocabularyParams() :clusterCount(1000), descriptorFraction(0.5)
	{
	}
	constructVocabularyParams(int _clusterCount, float _descriptorFraction)
	{
		clusterCount = _clusterCount;
		descriptorFraction = _descriptorFraction;
	}
	void printMessage()
	{
		cout << "cluster count is : " << clusterCount << endl;
		cout << "descriptor fraction is : " << descriptorFraction << endl << endl;
	}
};



/***************************************************************
* ·����Ϣ����
*****************************************************************/
struct pathInfo
{
	string saveExtension; //����ʵ��svmѵ��������Ϣ�ļ���չ��
	string trainParentImgPath; //ѵ��ͼ��·��
	string testParentImgPath; //����ͼ��·��
	string filter; //ͼ����չ��
	string savePath; //����ʵ��svmѵ��������Ϣ�ļ�·��
	pathInfo()
	{
		saveExtension = ".xml";
		filter = "*.jpg";
		trainParentImgPath = "F:\\�о����׶�\\MyProgramming\\BoWClassifierImg\\ImageSetInfo\\TrainJPEGImages\\";
		testParentImgPath = "F:\\�о����׶�\\MyProgramming\\BoWClassifierImg\\ImageSetInfo\\TestJPEGImages\\";
		savePath = "F:\\�о����׶�\\MyProgramming\\BoWClassifierImg\\ImageSetInfo\\SaveInfo\\";
	}
	pathInfo(string& _saveExtension, string&  _filter, string& 
		_trainParentImgPath, string&  _testParentImgPath, string&  _savePath)
	{
		saveExtension = _saveExtension;
		filter = _filter;
		trainParentImgPath = _trainParentImgPath;
		testParentImgPath = _testParentImgPath;
		savePath = _savePath;
	}
	void printMessage()
	{
		cout << "save file extension name is : " << saveExtension  << endl;
		cout << "file filter type is : " << filter << endl;
		cout << "train image parent path is : " << trainParentImgPath << endl;
		cout << "test image parent path is: " << trainParentImgPath << endl;
		cout << "save file path is: " << savePath << endl << endl;
	}
};


/***************************************************************
* ����ָ���ļ�����ָ����ʽ�������ļ���ȫ·��
* imageDir �ļ������֣���ʽΪ : ����"C:\\Image\\"
* patternFilter ͼ����չ��, ��ʽΪ"*.jpg"
* objectClass  ·��imageDir�µ����ļ������֣�Ҳ��ͼ���ǩ����
* imageNameList �ļ�����ָ����ʽ�������ļ���ȫ·��
*****************************************************************/
void getImageNameList(string imageDir, string patternFilter, vector<string>& objectClass,
	vector<string>& imageNameList)
{
	size_t objectClassNum = objectClass.size();

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = nullptr;
	LPCTSTR lpszText;
	string imagePath;
	string separator = "\\";
	for (int i = 0; i < objectClassNum; i++)
	{
		imagePath = imageDir + objectClass[i] + separator + patternFilter;
		lpszText = imagePath.c_str(); //תΪC�����ַ���

		hFind = FindFirstFile(lpszText, &FindFileData);
		while (hFind != INVALID_HANDLE_VALUE)
		{
			imageNameList.push_back(imageDir + objectClass[i] + separator + (string)(FindFileData.cFileName));
			if (!FindNextFile(hFind, &FindFileData))
			{
				FindClose(hFind);
				hFind = INVALID_HANDLE_VALUE;
			}
		}
	}
}


/********************************************************************
* ����0��totalN֮�䲻�ظ���ramdomN������������������randomNumber
*********************************************************************/
void generateUnRepeatRondomNumber(int randomN, int totalN, vector<int>& randomNumber)
{
	RNG& rng = theRNG();
	vector<int> totalNumber;	

	int i = 0;
	for (i = 0; i < totalN; i++)
	{
		totalNumber.push_back(i);
	}
	size_t tepRandom = 0;
	for (i = 0; i < randomN; i++)
	{
		tepRandom = rng((unsigned int)(totalNumber.size()-1)); //�������һ��0-totalNumber.size()֮��������
		randomNumber.push_back(totalNumber[tepRandom]);
		totalNumber.erase(totalNumber.begin() + tepRandom); //ɾ���Ѿ�������������
	}
}


/********************************************************************
* K��ֵ���ɵ��ֵ�д��ָ���ļ�
*********************************************************************/
bool writeVocabulary(string& vocFileName, Mat& vocabulary)
{
	cout << "Saving vocabulary...Please waiting..." << endl;
	cout << "Vocabulary path: " << vocFileName << endl;
	FileStorage fs(vocFileName, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "vocabulary" << vocabulary;
		cout << "Vocabulary write over. " << endl<<endl;
		return true;
	}
	return false;
}


/********************************************************************
* ��ָ���ļ���ȡK��ֵ���ɵ��ֵ�
*********************************************************************/
bool readVocabulary(string& vocFileName, Mat& vocabulary)
{
	cout << "Reading vocabulary...";
	FileStorage fs(vocFileName, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
		cout << "Vocabulary read over. " << endl << endl;
		return true;
	}
	return false;
}


/********************************************************************
* �����ʵ�
* trainImageNameList �����ʵ�ͼ��ȫ·��
* featureDetector ����������
* descriptorExtractor ����������������
*********************************************************************/
void constructVocabulary(string& vocFilePath, string& trainImageDir, string& patternFilter, 
	vector<string>objectClass, constructVocabularyParams& conVocParams,
	Ptr<FeatureDetector>& featureDetector, Ptr<DescriptorExtractor>& descriptorExtractor, Mat& vocabulary)
{
	if (!readVocabulary(vocFilePath, vocabulary))//���ֵ��ļ�����ֱ�Ӷ�ȡ���������¹����ֵ�
	{
		vector<string> trainImageNameList;
		getImageNameList(trainImageDir, patternFilter, objectClass, trainImageNameList);
		size_t imageNum = trainImageNameList.size();

		TermCriteria termCriteria;
		termCriteria.epsilon = FLT_EPSILON;
		BOWKMeansTrainer bowKMeansTrainer = BOWKMeansTrainer(conVocParams.clusterCount, termCriteria, 3, KMEANS_PP_CENTERS);

		int i = 0;
		for (i = 0; i < imageNum; i++)
		{
			Mat image = imread(trainImageNameList[i]);
			vector<KeyPoint> keyPoint;
			featureDetector->detect(image, keyPoint);
			Mat descriptors;
			descriptorExtractor->compute(image, keyPoint, descriptors);//��������������

			//ʹ���������ӵ�conVocParams.descriptorFraction��������
			int usedDescNum = (int)(descriptors.rows * conVocParams.descriptorFraction);
			vector<int> randomNumber;
			generateUnRepeatRondomNumber(usedDescNum, descriptors.rows, randomNumber);

			int j = 0;
			for (j = 0; j < usedDescNum; j++)
			{
				bowKMeansTrainer.add(descriptors.row(randomNumber[j]));//��������ӵ�����������
			}
		}

		vocabulary = bowKMeansTrainer.cluster();//K��ֵ����
		writeVocabulary(vocFilePath, vocabulary);//�����ֵ�
	}
}


/********************************************************************
* ��ʼ������ͼ��������ǩ
*********************************************************************/
void initObjectClass(vector<string>& objectClass)
{
	objectClass.push_back("anchor");
	objectClass.push_back("brain");
	objectClass.push_back("camera");
	objectClass.push_back("cup");
	objectClass.push_back("Faces");
	objectClass.push_back("panda");
	//objectClass.push_back("diningtable");
	//objectClass.push_back("dog");

	cout << "Classifier has been labeled." << endl << endl;
}


/********************************************************************
* ����SVMѵ������CvSVMParams
*********************************************************************/
void setCvSVMparams(CvSVMParams& cvSVMparams)
{
	cvSVMparams.svm_type = SVM::C_SVC;
	cvSVMparams.C = 0.005;
	//cvSVMparams.kernel_type = SVM::LINEAR;
	cvSVMparams.kernel_type = CvSVM::RBF;
	cvSVMparams.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
}


/********************************************************************
* ����SVM�Զ�ѵ������CvParamGrid
*********************************************************************/
void setSVMAutoTrainCvParamGrid(CvParamGrid& c_grid, CvParamGrid& gamma_grid, CvParamGrid& p_grid, 
	CvParamGrid& nu_grid, CvParamGrid& coef_grid, CvParamGrid& degree_grid)
{
	c_grid = CvSVM::get_default_grid(CvSVM::C);

	gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

	p_grid = CvSVM::get_default_grid(CvSVM::P);
	p_grid.step = 0;

	nu_grid = CvSVM::get_default_grid(CvSVM::NU);
	nu_grid.step = 0;

	coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
	coef_grid.step = 0;

	degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
	degree_grid.step = 0;
}

/********************************************************************
* ����SVM�Զ�ѵ������response����Ϊ������ӷ����ǩ1��-1
*********************************************************************/
void getSVMResponses(int negativeClassNum, int positiveClassNum, Mat& responses)
{
	responses.rowRange(0, negativeClassNum).setTo(-1);
	responses.rowRange(negativeClassNum, negativeClassNum + positiveClassNum).setTo(1);
}

//��ȡSVM�����ǩ��1����-1
/*void getSVMResponses(int negativeClassNum, int positiveClassNum, Mat& responses)
{
	for (int i = 0; i < respLength; i++)
	{
		size_t position = trainImageNameList[i].find("_");
		string labelStr = trainImageNameList[i].substr(0, position);
		stringstream strStream;
		strStream << labelStr;
		int labelInt = 0;
		strStream >> labelInt;
		if (labelInt == label)
		{
			responses.rowRange(i, i + 1).setTo(1);
			//responses.at<int>(i) = 1;
		}
		else
		{
			responses.rowRange(i, i + 1).setTo(-1);
			//responses.at<int>(i) = -1;
		}
	}
}
*/


//void svmTrainClassifier(CvSVM& cvSVM, string svmSaveDir, string svmSaveExtension,string trainImageDir, 
//	string patternFilter,vector<string>objectClass, constructVocabularyParams conVocParams,
//	Ptr<FeatureDetector>& featureDetector, Ptr<BOWImgDescriptorExtractor> bowDesExtractor)
//{
//	
//	CvSVMParams cvSVMparams;
//	setCvSVMparams(cvSVMparams);
//	CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
//	setSVMAutoTrainCvParamGrid(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
//
//	size_t objClassNum = objectClass.size();
//	//vector<string> svmTrainImageList;
//	
//	for (int objClIndex = 0; objClIndex < objClassNum; objClIndex++)
//	{
//		vector<string> objectClassPositive;
//		vector<string> objectClassNegative;
//		objectClassNegative = objectClass;
//		objectClassPositive.push_back(objectClassNegative[objClIndex]);//��objClIndex����ǩ�µ�ͼ����Ϊһ��+1
//		objectClassNegative.erase(objectClassNegative.begin() + objClIndex);//�ų���objClIndex����ǩ���������б�ǩ�µ�ͼ����Ϊһ��-1
//		
//		//�ֱ��ȡ�����������ͼ��·��
//		vector<string> svmTrainClassNegativePath;
//		getImageNameList(trainImageDir, patternFilter, objectClassNegative, svmTrainClassNegativePath);
//		vector<string> svmTrainClassPositivePath;
//		getImageNameList(trainImageDir, patternFilter, objectClassPositive, svmTrainClassPositivePath);
//
//
//		Mat trainData(svmTrainClassNegativePath.size() + svmTrainClassPositivePath.size(), conVocParams.clusterCount, CV_32FC1);
//		Mat responses(svmTrainClassNegativePath.size() + svmTrainClassPositivePath.size(), 1, CV_32FC1);
//
//		int negativeClassNum = svmTrainClassNegativePath.size();
//		for (int i = 0; i < negativeClassNum; i++)
//		{
//			Mat image = imread(svmTrainClassNegativePath[i]);
//			cout << svmTrainClassNegativePath[i] << endl;
//			vector<KeyPoint> keyPiont;
//			featureDetector->detect(image, keyPiont);
//			Mat descriptors;
//			bowDesExtractor->compute(image, keyPiont, descriptors);
//			descriptors.copyTo(trainData.row(i));
//		}
//
//		int positiveClassNum = svmTrainClassPositivePath.size();
//		for (int i = 0; i < positiveClassNum; i++)
//		{
//			Mat image = imread(svmTrainClassPositivePath[i]);
//			cout << svmTrainClassPositivePath[i] << endl;
//			vector<KeyPoint> keyPiont;
//			featureDetector->detect(image, keyPiont);
//			Mat descriptors;
//			bowDesExtractor->compute(image, keyPiont, descriptors);
//			descriptors.copyTo(trainData.row(i + negativeClassNum));
//		}
//
//		getSVMResponses(negativeClassNum, positiveClassNum, responses);
//		for (int ii = 0; ii < negativeClassNum + positiveClassNum; ii++)
//		{
//			cout << responses.at<float>(ii) << endl;
//		}
//		
//
//		//cvSVM.train(trainData, responses, Mat(), Mat(), cvSVMparams);
//		cvSVM.train_auto(trainData, responses, Mat(), Mat(), cvSVMparams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
//		
//	/*	Mat image1 = imread("F:\\�о����׶�\\MyProgramming\\BOW\\VOC2010\\trainJPEGImages\\accordion\\1_0003.jpg");
//		vector<KeyPoint> keyPiont11;
//		featureDetector->detect(image1, keyPiont11);
//		Mat descriptors11;
//		bowDesExtractor->compute(image1, keyPiont11, descriptors11);
//		
//		descriptors11.copyTo(testData.row(0));*/
//		/*Mat testData(1, 100, CV_32FC1);
//		testData = trainData.row(7);
//		float confidence = 0.0;
//		confidence = cvSVM.predict(trainData.row(7), false);
//		confidence = cvSVM.predict(trainData.row(4), false);
//		confidence = cvSVM.predict(trainData.row(13), false);*/
//		string separator = "\\";
//		string svmSaveFullDir = svmSaveDir + objectClassPositive[0] + svmSaveExtension;
//		cvSVM.save(svmSaveFullDir.c_str());
//	}
//}


/************************************************************************ 
*SVM����
************************************************************************/
void svmTrainClassifier(CvSVM& cvSVM, string svmFileName, string trainImageDir,string patternFilter, 
	vector<string>objectClass,	Ptr<FeatureDetector>& featureDetector, 
	Ptr<BOWImgDescriptorExtractor> bowDesExtractor)
{
	FileStorage fs(svmFileName, FileStorage::READ);
	if (fs.isOpened())//��ѵ���õ�SVM���ļ�����ֱ�Ӷ�ȡ������ѵ��
	{
		cout << "SVM is trained." << endl << endl;
		cvSVM.load(svmFileName.c_str());
	}
	else
	{
		cout << "SVM is training..." << endl;
		//����SVM����
		CvSVMParams cvSVMparams;
		setCvSVMparams(cvSVMparams);
		CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
		setSVMAutoTrainCvParamGrid(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

		size_t objClassNum = objectClass.size();
		vector<string> svmTrainImageList;
		getImageNameList(trainImageDir, patternFilter, objectClass, svmTrainImageList);

		Mat trainData(svmTrainImageList.size(), bowDesExtractor->getVocabulary().rows, CV_32FC1);
		Mat responses(svmTrainImageList.size(), 1, CV_32FC1);
		int index = 0;
		for (int objClIndex = 0; objClIndex < objClassNum; objClIndex++)
		{
			vector<string> objectClassOne;
			objectClassOne.push_back(objectClass[objClIndex]);//��objClIndex����ǩ�µ�ͼ����Ϊһ��

			//��ȡ��objClIndex�����ͼ��·��
			vector<string> svmTrainClassPath;
			getImageNameList(trainImageDir, patternFilter, objectClassOne, svmTrainClassPath);

			int ClassNum = svmTrainClassPath.size();
			for (int i = 0; i < ClassNum; i++)
			{
				Mat image = imread(svmTrainClassPath[i]);
				//cout << svmTrainClassPath[i] << endl;
				vector<KeyPoint> keyPiont;
				featureDetector->detect(image, keyPiont);
				Mat descriptors;
				bowDesExtractor->compute(image, keyPiont, descriptors);

				descriptors.copyTo(trainData.row(i + index));
			}

			responses.rowRange(index, ClassNum + index).setTo(objClIndex);//��������ǩ
			index += ClassNum;
		}

		//cvSVM.train(trainData, responses, Mat(), Mat(), cvSVMparams);
		cvSVM.train_auto(trainData, responses, Mat(), Mat(), cvSVMparams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
		cvSVM.save(svmFileName.c_str());
		cout << "SVM is trained and saved." << endl << endl;
	}
}

//void experimentBoWClassifier(CvSVM& cvSVM, string svmSaveDir, string svmSaveExtension, string testImageDir,
//	string patternFilter, vector<string>objectClass,
//	Ptr<FeatureDetector>& featureDetector, Ptr<BOWImgDescriptorExtractor> bowDesExtractor)
//{
//	size_t objClassNum = objectClass.size();
//	//vector<string> svmTrainImageList;
//
//	for (int objClIndex = 0; objClIndex < objClassNum; objClIndex++)
//	{
//		vector<string> objectClass_objClIndex;
//		objectClass_objClIndex.push_back(objectClass[objClIndex]);
//		vector<string> objectClassPath;
//		getImageNameList(testImageDir, patternFilter, objectClass_objClIndex, objectClassPath);
//		
//		int imgLength = objectClassPath.size();
//		for (int i = 0; i < imgLength; i++)
//		{
//			Mat image = imread(objectClassPath[i]);
//			cout << "OBJpath:" << objectClassPath[i] << endl;
//			vector<KeyPoint> keyPiont;
//			featureDetector->detect(image, keyPiont);
//			Mat descriptors;
//			bowDesExtractor->compute(image, keyPiont, descriptors);
//			Mat testData(1, bowDesExtractor->getVocabulary().rows, CV_32FC1);
//			descriptors.copyTo(testData.row(0));
//			for (int j = 0; j < objectClass.size();j++)
//			{
//				string svmSaveFullDir = svmSaveDir + objectClass[j] + svmSaveExtension;
//				cout << "SVMpath:" << svmSaveFullDir << endl;
//				cvSVM.load(svmSaveFullDir.c_str());
//				float confidence = 0.0;
//				confidence = cvSVM.predict(testData, false);
//				if (confidence == 1.0)
//				{
//					int mmmm = 0;
//				}
//			}	
//		}
//	}
//}

/************************************************************************
*SVM����
************************************************************************/
void experimentBoWClassifier(CvSVM& cvSVM, string svmFileName, string testImageDir,
	string patternFilter, vector<string>objectClass,
	Ptr<FeatureDetector>& featureDetector, Ptr<BOWImgDescriptorExtractor> bowDesExtractor)
{
	size_t objClassNum = objectClass.size();

	cvSVM.load(svmFileName.c_str());
	int rightImgNum = 0;//������ȷ��ͼ�����
	int allImgNum = 0;//����ͼ���ܸ���

	for (int objClIndex = 0; objClIndex < objClassNum; objClIndex++)
	{
		vector<string> objectClass_objClIndex;
		objectClass_objClIndex.push_back(objectClass[objClIndex]);
		vector<string> objectClassPath;
		getImageNameList(testImageDir, patternFilter, objectClass_objClIndex, objectClassPath);

		int imgLength = objectClassPath.size();
		allImgNum += imgLength;
		int classRightNum = 0;
		for (int i = 0; i < imgLength; i++)
		{
			Mat image = imread(objectClassPath[i]);
			//cout << "OBJpath:" << objectClassPath[i] << endl;
			vector<KeyPoint> keyPiont;
			featureDetector->detect(image, keyPiont);
			Mat descriptors;
			bowDesExtractor->compute(image, keyPiont, descriptors);

		/*	int writedata[200];
			for (int i = 0; i < 200; i++)
			{
				writedata[i] = descriptors.at<int>(i);
			}
			FILE* fid = NULL;
			fopen_s(&fid, "F:\\�о����׶�\\MyProgramming\\BoWClassifierImg\\ImageSetInfo\\SaveInfo\\voc.dat", "wb+");
			int mm = fwrite(writedata, sizeof(int), 200, fid);
			fclose(fid);*/

			Mat testData(1, bowDesExtractor->getVocabulary().rows, CV_32FC1);
			descriptors.copyTo(testData.row(0));
			float confidence = 0.0;
			confidence = cvSVM.predict(testData, false);
			if (objClIndex == (int)(confidence))//��svmԤ����������֪�����Ԥ����ȷ
			{
				rightImgNum++;
				classRightNum++;
			}
		}
		cout << objectClass[objClIndex] << " total test number is " << imgLength << endl;
		cout << objectClass[objClIndex] << " right recognition number is " << classRightNum << endl;
		cout << objectClass[objClIndex] << " precsion is " << classRightNum*1.0/imgLength << endl <<endl;
	}
	
	float precsion = 0.0;
	precsion = rightImgNum * 1.0/ allImgNum;//ͼ�����׼ȷ��
	cout << "test image total number is " << allImgNum <<endl;
	cout << "right recognition image number is " << rightImgNum <<endl;
	cout << "The precsion is " << precsion << endl;
}


int _tmain(int argc, _TCHAR* argv[])
{
	////------------------------ ����·����Ϣ ------------------------------------------------////
	pathInfo pathinfo;
	pathinfo = pathInfo();
	pathinfo.printMessage();

	string vocFileName;//���ɵĴʵ䱣��·��
	string voc = "vocabulary";
	vocFileName = pathinfo.savePath + voc + pathinfo.saveExtension;
	cout << "vocabulary save path is: " << vocFileName << endl;
	string svmFileName;//SVMѵ������·��
	string svm = "svm";
	svmFileName = pathinfo.savePath + svm + pathinfo.saveExtension;
	cout << "svm save path is: " << vocFileName << endl << endl;

	////------------------------ ��ʼ��ģ�飬ʹ��SIFT��SURFʱ�õ� -----------------------------////
	initModule_nonfree();     


	////------------------------ ������ȡ���� --------------------------------------------------////
	FDMParams fdmParams;
	fdmParams = FDMParams();//ʹ��Ĭ�ϵ�������ȡ����"SURF"
	fdmParams.printMessage();
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create(fdmParams.featureDetectorType);
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(fdmParams.descriptorType);
	Ptr<DescriptorMatcher> descriptorMatch = DescriptorMatcher::create(fdmParams.matcherType);
	

    ////------------------------ ���ɴʵ� ----------------------------------------------------////
	constructVocabularyParams conVocParams;
	conVocParams = constructVocabularyParams();//ʹ��Ĭ�Ϲ��캯������������Ϊ1000��ʹ���������ӵ�0.5���ɴʵ�
	conVocParams.printMessage();
	
	vector<string>objectClass;
	initObjectClass(objectClass);

	Mat vocabulary;
	constructVocabulary(vocFileName, pathinfo.trainParentImgPath, pathinfo.filter, objectClass,
		conVocParams, featureDetector, descriptorExtractor, vocabulary);

	Ptr<BOWImgDescriptorExtractor> bowDesExtractor =
		new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatch);
	bowDesExtractor->setVocabulary(vocabulary);


	////------------------------SVMѵ�� ------------------------------------------------------////
	CvSVM cvSVM;
	svmTrainClassifier(cvSVM, svmFileName, pathinfo.trainParentImgPath, 
		pathinfo.filter, objectClass, featureDetector, bowDesExtractor);
	

	////------------------------SVMͼ�������ԣ�����׼ȷ��------------------------------------////
	experimentBoWClassifier(cvSVM, svmFileName, pathinfo.testParentImgPath,
		pathinfo.filter, objectClass, featureDetector, bowDesExtractor);

	return 0;
}