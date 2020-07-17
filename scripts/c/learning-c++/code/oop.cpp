#include<iostream>
using namespace std;

class Person{
    public:
        char* name;
        static int object_cnt;
        int age;
        int gender;
        void getAge();
        void setName(char* _name);
        virtual void whoami();
        Person operator+(const Person& p);
        Person* getADDR();  // 返回一个指向当前对象的指针
        Person(char* _name,int _age, int _gender);       // Constructor DONT Need a type
        ~Person();
};

Person Person::operator+(const Person& p){
    char tmp_name[] = " ";
    Person person2return(tmp_name,10,0);
    person2return.age = this->age + p.age;
    cout << "The Op + Overrided" << endl;
    return person2return;
}

Person* Person::getADDR(){
    cout << this << endl;
    return this;
}

Person::~Person(){
}

void Person::whoami(){
    cout << "I'm a person" << endl;
}

void Person::setName(char* _name){
    this->name = _name;
}

void Person::getAge(){
    cout << this->age << endl;        // 这里不用Person.age，也不是对象名 peron.age
}

Person::Person(char* _name, int _age, int _gender){
    name = _name;
    age = _age;
    gender = _gender;
}

class NormalPerson: public Person{
    public:
        NormalPerson();
        ~NormalPerson();
        void whoami();
};

NormalPerson::NormalPerson():Person(name,age,gender){
    cout << "Creating a Normal Person Object" << endl;
}

NormalPerson::~NormalPerson(){
}

void NormalPerson::whoami(){
    cout << "I'm Normal Person:" << this->name << endl;
}

// 继承自Person
class StandUser: public Person{
    public:
        // using Person::Person;  // This line is somehow useless
        char* stand_name;
        StandUser(char * _standname);
        ~StandUser();
        void getNames();
        void getNames(int i);  // define the override getNames
        void whoami();
};

StandUser::StandUser(char * _standname):Person(name,age,gender){
    stand_name = _standname;
}

StandUser::~StandUser(){
}

void StandUser::whoami(){
    cout << "I'm a Stand User:" << this->stand_name << endl;
}

void StandUser::getNames(){
    cout << "Stand User:" << this->name << endl;
    cout << "Stand:" << this->stand_name << endl; 
}

void StandUser::getNames(int i){
    cout << "Checkin' Overriding' The SatndUsers' getName" << i << endl;
}


char name0[] = "Josuke";
char standname0[] = "Crazy Diamond";
// Person person0("Josuke",19,0);  //如果构造函数的输入参数为char*,会给一个Warning"Josuke" is a string constant not char*,将输入参数类型改成string就没有问题
Person person0(name0,19,0);
Person person1(name0,20,0);
Person* person_pointer0;
Person* person_pointer1;
Person* person_pointer2;

NormalPerson normalperson0;     //注意即使是normalperon0即使构造函数的参数为空 不能写成 NormalPerson normalperon0()
StandUser standuser0(standname0);

int Person::object_cnt = 2;

int main(){

    // person0.age = 20;
    // cout << person0.age << endl;

    // person_pointer = &person0;
    // person_pointer -> getADDR();
    // person0.getAge();
    // person0.getADDR();
    // cout << person0.object_cnt << endl;
    Person person2(name0,0,0);
    // person2 = person1 + person0;     // Test Overrided


    person_pointer0 = &person2;
    // cout << person_pointer0 -> age << endl;
    // cout << person2.age << endl;

    // * Teting Polymorphism
    person_pointer1 = &standuser0;
    person_pointer1->whoami();
    person_pointer1 = &normalperson0;
    person_pointer1->name = (char*)("⚾");
    person_pointer1->whoami();


    // standuser0.setName(name0);      // Use The setName inherited from Person
    // standuser0.getNames(3);

    return 0;
}
