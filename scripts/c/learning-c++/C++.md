# C++
* 从来没有系统学习过C++，导致用起来很费劲，开始做一个积累记录
* C++是
    * 静态类型语言：在编译时进行类型检查，而不是在执行时
    * OOP：封装，抽象，继承，多态
    * 包含： 核心语言，标准库(string.h)，标准模板库(STL)


## etc
* ```typedef int new_int; new_int i;```
* 枚举
    * 默认第一个为0，之后1,2,3，也可以从中间开始指定，之后的比前面的多1
   ``` C
   enum color{
       red=1;
       green=2,
       blue=3
   };
   ```
* 作用域
    * 全局(Global)
    * 局部（函数或者是代码块中）
      * 定义局部变量时，系统不会自动初始化（全局会自动初始化，大部分值为0）
    * 形参，函数定义时
* 常量可以加后缀（u l）表示unisgned和long
    * 3.1415E-3L 科学计数法的常量
    * 定义常量
        * 可以采用 预处理器 *#define*
        * 或者是const
            * 这里const实际是一种修饰符
         
            |修饰符|意义|
            |--|--|
            |const|运行时不能被改变|
            |volatile|告诉编译器不优化该变量的存储（从内存中转移到寄存器中）程序可以直接从内存中读取变量）
            |restrict|具有该修饰的指针是唯一指向对象的方式（Only In C99）|      
    * 一般要把常量定义成大写
* 存储类型（决定了变量的生命周期）

|存储类|特点|
|--|--|
|register|局部变量存储在reg中而不是ram中（不能对他进行一元运算（因为他不不在内存中））|
|**static**|1. 局部变量：在程序的执行周期之内保持该局部变量的值，不用每次离开作用域的时候销毁，可以在函数调用之间保持值|
||2. 全局变量，作用域被限制为声明它的文件|
||3. 类数据成员： 仅存在一个该类别的副本，被所有对象共享|
|**extern**|用于不同文件之间共享相同的全局变量或*函数*（extern void func()）|
|thread_local|仅可在其当前线程访问|

* 杂项运算符

|||
|--|--|
|sizeof|返回变量占空间大小|
|（）？X：Y||
|. / ->|访问类或者结构体内部成员|
|强制类型转换||
|&/*|指针相关|

* 数学运算
* ```#include<cmath>```内置的一些数学运算
    * sin/con/tan
    * log (means ln)
    * pow(x,y)  x^y
    * sqrt()
    * abs
    * floor

* 字符串
    * C Style (Char String Style)
        * ```char a[] = "Hello"```
        * ```char a[] = {'a','b','c'}```
        * 在内存中，数组在初始化时，会自动将一个\0放在字符串的末尾
        * 方法
            * strcat
            * strcpy
            * strlen
            * strcmp
    * C++ ```include <string>```
        * 定义 ```string str=" "```
        * 复制 = 
        * cat +
        * s.size  
* 输入输出
    * 基本输入输出流 （cin/cout）
        * ```cin >> x```  

## 指针

* 定义 ```int *fp;    float *fp```
* 指针数组
    * 用法1： ```char * s[10]```来表示字符串
* 引用，其实和指针有着微妙的区别    ```int i;   int&r = i;  // r是i的引用```
    * 与指针的区别
        * 必须被链接到一块内存
        * 被初始化为一个对象之后不能指向另外一个对象
        * 智能在创建时初始化
    * 经常作为函数的参数或者是返回值
        * 参数  ```void swap(int &X,int &y)```
            * 相当于传了一个形参，与实际参数的区别在于实际参数传进来的是值，而这里传入的其实是实际变量的指针(引用)
* Example At [./code/hello.cpp]()

## 结构体
* ```
    typedef struct Person // 用了typedef之后所有的struct Person都可以简化为Person
    // 当然typedef也可以用来对其他类型别名    typedef unsigned short *uint16_t; uint16_t a;
    {
        char name[100];
        int age;
    } person1,person2,person3;
    ```
* 用"."访问属性
* 可以直接作为函数参数输入  ```void print(struct Person person1)```
* 也可以有指向结构体的指针  ```strcut Person *person_pointer;  person_pointer = person1;```
    * 使用结构体指针访问对象的时候，必须要 ```person_pointer -> age```

## OOP
### Class
* Template 
    * ```
      class Person{
          public:
            char* name;
            int age;
            // bool gender;  // Not Political Right   
            int gender;
        

      };

      Person Kujo_Jotaro;
      ```
    * 对于非private的属性，正常访问方式是通过"."
* **定义成员函数**，在类定义内部（某域内，比如Public） ```void doSomething(..)即可```
    * 或者是在类的外部用*范围解析运算符::*   ```double Person::getDouble(..)``` 
        * 当然在这种scenario下类的里面也是要定义下这个函数的名字的  
        * 类内的函数调用类内的变量，直接用类定义里面的变量名    
            * 上面Example中的age,而不是Person.age或者是person0.age
    * 两者调用都用"."，只有指针时候才用"->"
* **类访问修饰符** public, protected, private
    * 对public变量，可以直接在类的外部进行赋值  ```person0.age = 18```
        * 而private的变量，外部是不可以访问的，需要定义一个 ```void setAge()```(在Public Field)
    * protected与private比的区别是可以在其派生类当中访问
        * private类型的可以被类内部或者*友元*访问
    * 继承对雷访问修饰符的变化 优先度 private > protected > public
        * old level & inherit level = new level
    * *默认类型是private* 类默认是private继承，而struct默认是public继承
* **构造与析构**
    * 每次创建类的对象时候自动被调用，定义的时候不需要Return Type
        * ```People::People(char *_name, int _age): name(_name), age(_age){}``` 也可以用这种方式进行初始化
    * 析构 ```~Person```
        * 不带参数，没有返回值，作用是*在跳出程序之前释放内存
* [拷贝构造函数](https://www.runoob.com/cplusplus/cpp-copy-constructor.html) ```Person(const Person &obj){}```
    * 如果没有，系统会自己建立一个
    * Application：
        * 利用已有的对象初始化对象 ```Person person1 = person0; ``` (此时是*显示调用*了拷贝构造函数)
        * 当对象传入(as param)或者传出函数(return),*隐式调用*拷贝构造函数
    * *初始化方式* 
        * 拷贝初始化（调用拷贝构造函数）  ```Person p1 = p2;```
        * 直接初始化                    ```p1 = Person(..);```
* [友元函数](https://www.runoob.com/cplusplus/cpp-friend-functions.html) 定义在类的外部，但是可以访问protected&Private的函数 ```friend void setXXX()```
    * 可以是函数也可以是类  ```fiend Person StandUser```
    * 并不是成员函数，但是需要在类中定义
    * TODO: 为什么要产生这个东西？和用法
        * 就只是为了方便吗？那把这个方法放在public里面不是也可以起到效果？好像区别就在于友元函数不是类内部，是一种从类的外部模拟类的内部的形式) 
* This指针:
    * 每一个对象本质上都是通过this指针来访问自己的地址（是所有成员函数所隐含的 直接调用age相当于->age）
    * 友元没有this指针
    * this本质是一个隐式的常量指针 ```Person* get_address(){return this;}```
* 类的指针
    * **和一般的指针一样，在使用之前必须要进行初始化！**
    * 其访问成员需要用"->"
    * ```Person* person_pointer; person_pointer->getADDR()；```
* 静态成员
    * 可以将成员定义为static，创建的多个对象，静态成员都只会有一个副本
    * 初始化
        * 由于静态成员被所有对象共享，如果没有初始化语句，将会在第一个成员被初始化时被赋0
        * 不能将初始化放在*类的定义*中
        * 但是可以在外部利用::运算符通过重新声明对其初始化 ```int Person::object_cnt = 0;```
    * **静态成员函数**不依赖于对象（via ::）
        * 只能访问静态数据，没有this指针
### 继承 (Inherit)
* ```class SubClass : public Class``` 派生类
    * 可以访问基类中的所有非private成员
    * 继承了所有的基类的方法（除了构造，拷贝构造，析构，重载以及友元）
    * 一般只使用public继承
    * 多继承  ```class Subclass : public class1, public class2 ``` 
    * [继承的构造函数](https://www.cnblogs.com/jeavenwong/p/8094593.html) (会隐式调用基类的构造函数，如果不specify会调用default constructor)
        * *Constructors are not inherited. They are called implicitly or explicitly by the child constructor.* However In C++11, Could
            * A [Blog](https://www.learncpp.com/cpp-tutorial/114-constructors-and-initialization-of-derived-classes/) Clarifies That
        * 但是一般的类别都不太是default constructor（指没有输入参数，或者是输入参数全部确定的构造函数）
        * 方法一：```
                SubClass::SubClass(char * new_element):OldClass(A,B,C){
                    _stand_name = standname;
                }
                ```
        * 方法二：在Derived Class内部定义一个对象
    * 虚继承： 构造函数需要调用最终虚基类的构造函数 
        * 可以避免由于树状继承而产生多个对象的问题

## 重载 (Override)
* 函数重载 / 运算符重载 ： 在同一个作用域内对同一个变量名再次定义，但是参数列表与内部定义不一样
    * **函数重载**定义方式就和一般的函数相同，就是注意要在类的定义里面定义多个同名的成员函数 
    * **运算符重载** 由关键字```operator```加上运算符的名字，也带返回值以及参数
* 当调用重载函数的时候，编译器对使用的参数类型进行比较，选择合适的定义
* 不能改变优先级和操作目数

## 多态 （polymorphism）
* 当类之间存在层次的继承的时候，调用成员函数时，会因为函数对象的不同而执行不同的函数（可以认为是在不同对象之间的重载？虽然他们的参数也可能一样）
    * 当某一个指针指向A派生类的时候，调用的是A派生类的成员函数。
        * 指向不一样的对象，就会调用不一样的成员函数
    * 当要利用到多态的时候，基类中的方法必须定义为virtual不然定义的都会是基类的方法
* 虚函数Virtual的本意是告诉编译器不要动态链接
    * 这样当编译器先接触到基类的某个方法的时候，不会直接与他建立连接，而是只建立了一个动态软连接，便于指向新的类别的对象
    * *纯虚函数* 在类的定义的时候，如果你想要更好地在其派生类中实现功能，可在基类中定义一个纯虚函数：
        * ```virtual void something() = 0;```

## 接口（抽象类）
* 一个类中至少有一个函数被声明为虚函数，这个类就是一个抽象类
* 设计抽象类的目的只是为了给其派生类做一个模板，因而**抽象类不能被实例化为对象**，只能作为接口
* 如果一个派生类想要继承一个抽象类，必须要实现它的**所有虚函数**


## Qs
* A default constructor is a constructor that either has no parameters, or if it has parameters, all the parameters have default values.



        
