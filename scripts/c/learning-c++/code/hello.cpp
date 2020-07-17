#include<iostream>
using namespace std;

typedef int *new_int;

void do_something(int *something1, int *something2);    // A Way To Return Multiple Values
double getAverage(int *arr, int size);

int main(){
    // int * i;
    new_int i;
    int j;
    // *i = 1;     // Point The pointer to a constant , not allowed (before pointing to a vairiable)
    i = &j;
    *i = 1;         // Change The Varibe which is pointed to, allowed

    new_int i1,i2,i3;   //几个野指针
    int j1,j2,j3;       // 给野指针配的对象
    i1 = &j1;
    i2 = &j2;           // 他们不再是野指针了
    i3 = &j3;
    do_something(i1,i2);

    double average;
    int data[4] = {1,2,3,6};
    average = getAverage(data, 4);

    // printf("%d",*i);
    // cout << j1 << j2 <<endl;
    cout << average << endl;
    return 0;
}

void do_something(int* something1, int*something2){
    *something1 = 0x10;
    *something2 = 0x11;
}

double getAverage(int *arr, int size){
    int sum = 0;
    double average;

    for (int i=0;i<size;i++){
        sum = sum+arr[i];
    }

    average = sum/size;
    return average;
}