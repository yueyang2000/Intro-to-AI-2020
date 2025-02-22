#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include "Compete.h"
#include "Point.h"
#include "Data.h"
#include "Judge.h"

using namespace std;

typedef Point* (*GETPOINT)(const int M, const int N, const int* _top, const int* _board, const int lastX, const int lastY, const int noX, const int noY);
typedef void (*CLEARPOINT)(Point* p);

struct timeval now;
struct timespec stoptime;

//pthread_cond_t pcond;
//pthread_mutex_t pmutex;

struct Param{
	int M;
	int N;
	int* top;
	int* board;
	int lastX;
	int lastY;
	int noX;
	int noY;
	GETPOINT getPoint;
	Point* p;
	int bugOccurred;
};


void* callGetPoint(void* p_param){
    Param* param = (Param*) p_param;
	try{
		param->p = param->getPoint(param->M, param->N, param->top, param->board, param->lastX, param->lastY, param->noX, param->noY);
	}catch(...){
		param->bugOccurred = 1;
		return NULL;
	}
	param->bugOccurred = 0;
    
//    pthread_mutex_lock(&pmutex);
//    pthread_cond_signal(&pcond);
//    pthread_mutex_unlock(&pmutex);
    
	return NULL;
}

//传入getPointA 和 clearPointA
//returns : 0 - 平局结束 1 - A赢 2 - B赢 3 - A出错 4 - A给出非法落子 5 - B出错 6 - B给出非法落子 7 - A超时 8 - B超时 -1 - 游戏未结束
int AGo(GETPOINT getPoint, CLEARPOINT clearPoint, Data* data){
	int x, y;
	try{
		Param param;
		param.M = data->M;	param.N = data->N;	param.top = data->top;	param.board = data->boardA;
		param.lastX = data->lastX;	param.lastY = data->lastY;	param.noX = data->noX;	param.noY = data->noY;
		param.getPoint = getPoint;	param.p = NULL;	param.bugOccurred = 0;
		
        callGetPoint(&param);
		x = param.p->x;
		y = param.p->y;
		clearPoint(param.p);
	}
	catch(...){
		return 3;
	}
	
	if(!isLegal(x, y, data)){
		return 4;
	}
	data->lastX = x;
	data->lastY = y;
	data->boardA[x * data->N + y] = 2;
	data->boardB[x * data->N + y] = 1;
	data->top[y]--;
	//对不可落子点进行处理
	if(x == data->noX + 1 && y == data->noY){
		data->top[y]--;
	}
	
	if(AWin(x, y, data->M, data->N, data->boardA)){
		return 1;
	}
	if(isTie(data->N, data->top)){
		return 0;
	}
	return -1;
}

//传入getPointB 和 clearPointB
//returns : 0 - 平局结束 1 - A赢 2 - B赢 3 - A出错 4 - A给出非法落子 5 - B出错 6 - B给出非法落子 7 - A超时 8 - B超时 -1 - 游戏未结束
int BGo(GETPOINT getPoint, CLEARPOINT clearPoint, Data* data){
	int x, y;
	try{
		Param param;
		param.M = data->M;	param.N = data->N;	param.top = data->top;	param.board = data->boardB;
		param.lastX = data->lastX;	param.lastY = data->lastY;	param.noX = data->noX;	param.noY = data->noY;
		param.getPoint = getPoint;	param.p = NULL;	param.bugOccurred = 0;
		
        callGetPoint(&param);
		
		x = param.p->x;
		y = param.p->y;
		clearPoint(param.p);
	}
	catch(...){
		return 5;
	}
	
	if(!isLegal(x, y, data)){
		return 6;
	}
	data->lastX = x;
	data->lastY = y;
	data->boardA[x * data->N + y] = 1;
	data->boardB[x * data->N + y] = 2;
	data->top[y]--;
	//对不可落子点进行处理
	if(x == data->noX + 1 && y == data->noY){
		data->top[y]--;
	}
    
	if(BWin(x, y, data->M, data->N, data->boardB)){
		return 2;
	}
	if(isTie(data->N, data->top)){
		return 0;
	}
	return -1;
}

void printBoard(Data* data){
	int noPos = data->noX * data->N + data->noY;
	for(int i = 0; i < data->M; i++){
		for(int j = 0; j < data->N; j++){
			int pos = i * data->N + j;
			if(data->boardA[pos] == 0){
				if(pos == noPos){
					cout << "X ";
				}
				else{
					cout << ". ";
				}
			}
			else if(data->boardA[pos] == 2){
				cout << "A ";
			}
			else if(data->boardA[pos] == 1){
				cout << "B ";
			}
		}
		cout << endl;
	}
	return;
}

/*
 input:
 strategyA[] strategyB[] 两个策略文件的文件名
 Afirst: -true : A(前面的文件)先手 -false : B(后面的文件)先手
 reutrns:
 0 - 平局结束 1 - A赢 2 - B赢 3 - A出错 4 - A给出非法落子 5 - B出错 6 - B给出非法落子 7 - A超时 8 - B超时
 -1 - A文件无法载入 -2 - B文件无法载入 -3 - A文件中无法找到需要的接口函数 -4 - B文件中无法找到需要的接口函数
 */
int compete(char strategyA[], char strategyB[], bool Afirst, Data* data){
//    pthread_mutex_init(&pmutex, NULL);
//    pthread_cond_init(&pcond, NULL);
    
    void* hDLLA;
	GETPOINT getPointA;		// Function pointer
	CLEARPOINT clearPointA;
    
    void* hDLLB;
	GETPOINT getPointB;		// Function pointer
	CLEARPOINT clearPointB;

    hDLLA = dlopen(strategyA, RTLD_LOCAL|RTLD_NOW);
    hDLLB = dlopen(strategyB, RTLD_LOCAL|RTLD_NOW);

	if(!hDLLA){
		cout << "Load file A failed" << endl;
		return -1;
	}
	if(!hDLLB){
		cout << "Load file B failed" << endl;
		return -2;
	}
	
	getPointA = (GETPOINT)dlsym(hDLLA, "getPoint");
	clearPointA = (CLEARPOINT)dlsym(hDLLA, "clearPoint");
	getPointB = (GETPOINT)dlsym(hDLLB, "getPoint");
	clearPointB = (CLEARPOINT)dlsym(hDLLB, "clearPoint");
	
	if(getPointA == NULL || clearPointA == NULL){
		cout << "Can't find entrance of the wanted functions in the A DLL file" << endl;
		return -3;
	}
	if(getPointB == NULL || clearPointB == NULL){
		cout << "Can't find entrance of the wanted functions in the B DLL file" << endl;
		return -4;
	}
	
	//四个个函数已经拿到手，现在可以开始进行棋盘初始化和进行对抗了
	
	if(Afirst){
		int res = AGo(getPointA, clearPointA, data);
		if(res != -1){
			printBoard(data);
			return res;
		}
	}
	
	int res = -1;
	bool aGo = false;
	while(true){
		if(aGo){
			res = AGo(getPointA, clearPointA, data);
			aGo = false;
		}
		else{
			res = BGo(getPointB, clearPointB, data);
			aGo = true;
		}
		if(res != -1){
			printBoard(data);
			return res;
		}
	}
	
	return 0;//程序不应执行到这一步
}
