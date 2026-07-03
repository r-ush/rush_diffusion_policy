// For formal
#include <stdio.h>
#include <cmath>

// for TCP/IP
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#define BUF_SIZE 14


class Yoon_FT_sensor
{   
    /* Informations */

    // char ip; // ECAN IP (URE ECAN : 192.168.111.44)
    // int port; // ECAN PORT (URE ECAN : 4001)

    private:

        // parameters for setting
        char sendbuf[BUF_SIZE] = { 0x04,0x00,0x00,0x01,0x02,0x06,0x01,0x03,0x01,0x00,0x00,0x00,0x00,0x00 };
        char Contact_sendbuf[BUF_SIZE] = { 0x04,0x00,0x00,0x01,0x02,0x06,0x0b,0x03,0x01,0x00,0x00,0x00,0x00,0x00 };  
        bool init_flag = false;
        uint16_t return_out = 0;

        // parameters that do not touch
        int clnt_sock;
        int readstrlen = 0;

        struct sockaddr_in st_serv_addr; // sockaddr_in 구조체 변수 선언
        unsigned char recvmsg[BUF_SIZE]; // CAN to Ethernet must be set the buffer size 16

        // Calculation parameters
        double inter_force[3] = {0,}; // for force value transform
        double inter_moment[3] = {0,}; // for moment value transform
        double Cinter_force[3] = {0,}; // for force value transform
        double Cinter_moment[3] = {0,}; // for moment value transform
        double inter_posAcc[3] = {0,}; // for linear acceleration value transform
        double inter_angAcc[3] = {0,}; // for angular acceleration value transform

        
    public:
        Yoon_FT_sensor() {}
        ~Yoon_FT_sensor() {}


        double CAN_sampling = 1/50; // period(s)
        double Force_val[3], Moment_val[3], Contact_Force_val[3], Contact_Moment_val[3], Pos_acc_val[3], Ang_acc_val[3];
        
        int init_average_num = 50; // To use average force value for force initialization
        double init_Force[3] = {0,}; // initial values
        double init_Moment[3] = {0,}; // initial values
        double init_Contact_Force[3] = {0,}; // initial values
        double init_Contact_Moment[3] = {0,}; // initial values

        double Ang_vel_val[3] = {0,};
        double Ang_pvel_val[3] = {0,};
        
        int sensor_init_counter = 0;

        void TCP_init(char* IP, int port);
        uint16_t TCP_start();
        bool Sensor_value_init();
        void errhandle(const char *errmsg);



};
