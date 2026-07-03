#include "Yoon_FT_sensor.h"

void Yoon_FT_sensor::TCP_init(char* IP, int port)
{
    // 클라이언트 소켓 TCP/IP 프로토콜 생성
    this->clnt_sock = socket(PF_INET, SOCK_STREAM, 0);
    if(this->clnt_sock == -1) errhandle("socket() ERR!");

    // serv_sock에 bind로 주소 넣기 위한 밑작업
    memset(&this->st_serv_addr,0,sizeof(this->st_serv_addr));
    this->st_serv_addr.sin_family = AF_INET;
    this->st_serv_addr.sin_addr.s_addr = inet_addr(IP);
    this->st_serv_addr.sin_port = htons(port);

    // connect()으로 서버소켓에 연결요청
    int connret = connect(this->clnt_sock,(struct sockaddr*) &this->st_serv_addr, sizeof(this->st_serv_addr));
    if(connret == -1) errhandle("connect() ERR!");

    
    // To Hand-guiding sensor
    int iResult = send(this->clnt_sock, this->sendbuf, sizeof(this->sendbuf), 0);
    printf("Bytes Sent: %d\n", iResult);
    // To Contact force sensor
    iResult = send(this->clnt_sock, this->Contact_sendbuf, sizeof(this->Contact_sendbuf), 0);
    printf("Bytes Sent: %d\n", iResult);

    this->init_flag = true;
}

uint16_t Yoon_FT_sensor::TCP_start()
{
    if(this->init_flag == true)
    {
        this->readstrlen = read(this->clnt_sock, (char*)&this->recvmsg, sizeof(this->recvmsg));
        if(this->readstrlen == -1) errhandle("read() ERR!");

        // --------------- Sensor data calculation --------------- //
        if (this->recvmsg[4] == 0x01) // if ID is 1 -> Sensor1 force (Handle part)
        {
            
            #if 0 // To measure the sensor's sampling time
            gettimeofday(&this->tval_start, NULL);  // Get current time
            this->tval_microseconds = this->tval_end.tv_usec - this->tval_start.tv_usec;
            memcpy(&this->tval_end,&this->tval_start,sizeof(this->tval_start));
            printf("Force time interval: %d micros\n",this->tval_microseconds);
            #endif

            for (int i = 0; i < 3; i++)
            {
                this->inter_force[i] = (double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]) / 100 - 300;
                // inter_force[i] = FT_TCP_struct->Force_val[i];
            }

            // FT sensor transform according to its assembly configuration

            // original configuration
            #if 1
            this->Force_val[0] = this->inter_force[0] - this->init_Force[0]; 
            this->Force_val[1] = this->inter_force[1] - this->init_Force[1];
            this->Force_val[2] = this->inter_force[2] - this->init_Force[2]; 
            #endif

            /*
            // Handle force mv filtering
            FT_TCP_struct->Force_val[0] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Force_val[0], &FT_TCP_struct->FT_mv_fx);
            FT_TCP_struct->Force_val[1] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Force_val[1], &FT_TCP_struct->FT_mv_fy);
            FT_TCP_struct->Force_val[2] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Force_val[2], &FT_TCP_struct->FT_mv_fz);

            #if 1
            // Handle force Fixed BSF
            FT_TCP_struct->Force_val[0] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Force_val[0],&FT_TCP_struct->HGF_Fx_par);
            FT_TCP_struct->Force_val[1] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Force_val[1],&FT_TCP_struct->HGF_Fy_par);
            FT_TCP_struct->Force_val[2] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Force_val[2],&FT_TCP_struct->HGF_Fz_par);
            #endif

            #if 1 // al-BSF force switch
            // Hangle force al-BSF
            FT_TCP_struct->Force_val[0] = FT_TCP_struct->Fx_aBSF.al_BSF_operation(FT_TCP_struct->Force_val[0],FT_TCP_struct->joint2_vel);
            FT_TCP_struct->Force_val[1] = FT_TCP_struct->Fy_aBSF.al_BSF_operation(FT_TCP_struct->Force_val[1],FT_TCP_struct->joint2_vel);
            FT_TCP_struct->Force_val[2] = FT_TCP_struct->Fz_aBSF.al_BSF_operation(FT_TCP_struct->Force_val[2],FT_TCP_struct->joint2_vel);
            #endif
            */

            return_out = 1;
        }
        else if (this->recvmsg[4] == 0x02) // if ID is 2 -> Sensor1 moment (Handle part)
        {
            for (int i = 0; i < 3; i++)
            {
                this->inter_moment[i] = (double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]) / 500 - 50;
                // inter_moment[i] = FT_TCP_struct->Moment_val[i];
            }

            // FT sensor transform according to its assembly configuration

            // original configuration
            #if 1
            this->Moment_val[0] =  this->inter_moment[0] - this->init_Moment[0];
            this->Moment_val[1] =  this->inter_moment[1] - this->init_Moment[1];
            this->Moment_val[2] =  this->inter_moment[2] - this->init_Moment[2];
            #endif

            /*
            // Handle moment mv filtering
            FT_TCP_struct->Moment_val[0] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Moment_val[0], &FT_TCP_struct->FT_mv_mx);
            FT_TCP_struct->Moment_val[1] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Moment_val[1], &FT_TCP_struct->FT_mv_my);
            FT_TCP_struct->Moment_val[2] = yoon_indy7_filter::VR_mv_filter(FT_TCP_struct->Moment_val[2], &FT_TCP_struct->FT_mv_mz);

            #if 1
            // Handle moment Fixed BSF
            FT_TCP_struct->Moment_val[0] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Moment_val[0],&FT_TCP_struct->HGF_Mx_par);
            FT_TCP_struct->Moment_val[1] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Moment_val[1],&FT_TCP_struct->HGF_My_par);
            FT_TCP_struct->Moment_val[2] = yoon_indy7_filter::Yoon_BSF(FT_TCP_struct->Moment_val[2],&FT_TCP_struct->HGF_Mz_par);
            #endif

            #if 1 // al-BSF moment switch
            // Hangle moment al-BSF
            FT_TCP_struct->Moment_val[0] = FT_TCP_struct->Mx_aBSF.al_BSF_operation(FT_TCP_struct->Moment_val[0],FT_TCP_struct->joint2_vel);
            FT_TCP_struct->Moment_val[1] = FT_TCP_struct->My_aBSF.al_BSF_operation(FT_TCP_struct->Moment_val[1],FT_TCP_struct->joint2_vel);
            FT_TCP_struct->Moment_val[2] = FT_TCP_struct->Mz_aBSF.al_BSF_operation(FT_TCP_struct->Moment_val[2],FT_TCP_struct->joint2_vel);
            #endif
            */

            return_out = 2;
        }
        else if (this->recvmsg[4] == 0x03) // if ID is 3 -> Sensor1 linear acceleration (LAx,LAy,LAz)
        {
            for (int i = 0; i < 3; i++)
            {
                this->inter_posAcc[i] = ((double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]))*2-65535;
                this->inter_posAcc[i] = (this->inter_posAcc[i]/16384)*9.81; // m/s^2
            }
            
            // 45degree rotation
            this->Pos_acc_val[0] = -(cos(45*3.141592/180)*this->inter_posAcc[0] - sin(45*3.141592/180)*this->inter_posAcc[1]);
            this->Pos_acc_val[1] = -(sin(45*3.141592/180)*this->inter_posAcc[0] + cos(45*3.141592/180)*this->inter_posAcc[1]);
            this->Pos_acc_val[2] = -this->inter_posAcc[2];

            return_out = 3;

        }
        else if (this->recvmsg[4] == 0x04) // if ID is 4 -> Sensor1 angular acceleration (AAx,AAy,AAz)
        {
            for (int i = 0; i < 3; i++)
            {
                this->Ang_vel_val[i] = ((double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]))*2-65535;
                this->Ang_vel_val[i] = (this->Ang_vel_val[i]*250/32768)*(3.141592/180); // rad/s

                this->inter_angAcc[i] = (this->Ang_vel_val[i] - this->Ang_pvel_val[i])/this->CAN_sampling; // rad/s^2

                this->Ang_pvel_val[i] = this->Ang_vel_val[i];
            }

            // 45degree rotation
            this->Ang_acc_val[0] = -(cos(45*3.141592/180)*this->inter_angAcc[0] - sin(45*3.141592/180)*this->inter_angAcc[1]);
            this->Ang_acc_val[1] = -(sin(45*3.141592/180)*this->inter_angAcc[0] + cos(45*3.141592/180)*this->inter_angAcc[1]);
            this->Ang_acc_val[2] = -this->inter_angAcc[2];

            return_out = 4;
        }

        else if (this->recvmsg[4] == 0x0b) // if ID is 11 -> Sensor2 force (Contact part)
        {
            for (int i = 0; i < 3; i++)
            {
                this->Cinter_force[i] = (double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]) / 100 - 300;
            }

            // FT sensor transform according to its assembly configuration

            // original configuration
            #if 0
            this->Contact_Force_val[0] = -this->Cinter_force[1] + this->init_Contact_Force[1]; 
            this->Contact_Force_val[1] = -this->Cinter_force[0] + this->init_Contact_Force[0]; 
            this->Contact_Force_val[2] = -this->Cinter_force[2] + this->init_Contact_Force[2]; 
            #endif

            // vertical configuration
            #if 1
            this->Contact_Force_val[0] =  this->Cinter_force[1] - this->init_Contact_Force[1]; 
            this->Contact_Force_val[1] = -this->Cinter_force[2] + this->init_Contact_Force[2]; 
            this->Contact_Force_val[2] = -this->Cinter_force[0] + this->init_Contact_Force[0]; 
            #endif



            /*
            // Contact force filtering
            this->Contact_Force_val[0] = yoon_indy7_filter::VR_mv_filter(this->Contact_Force_val[0], &this->FT_mv_Cfx);
            this->Contact_Force_val[1] = yoon_indy7_filter::VR_mv_filter(this->Contact_Force_val[1], &this->FT_mv_Cfy);
            this->Contact_Force_val[2] = yoon_indy7_filter::VR_mv_filter(this->Contact_Force_val[2], &this->FT_mv_Cfz);
            */

            return_out = 11;
        }
        else if (this->recvmsg[4] == 0x0c) // if ID is 12 -> Sensor2 moment (Contact part)
        {
            for (int i = 0; i < 3; i++)
            {
                this->Cinter_moment[i] = (double)((int)this->recvmsg[6 + 2 * i] * 256 + (int)this->recvmsg[7 + 2 * i]) / 500 - 50;
            }

            // FT sensor transform according to its assembly configuration

            // original configuration
            #if 0
            this->Contact_Moment_val[0] = -this->Cinter_moment[1] + this->init_Contact_Moment[1]; // -y value -> x value
            this->Contact_Moment_val[1] = -this->Cinter_moment[0] + this->init_Contact_Moment[0]; // -x value -> y value
            this->Contact_Moment_val[2] = -this->Cinter_moment[2] + this->init_Contact_Moment[2]; // -z value -> z value
            #endif

            // vertical configuration
            #if 1
            this->Contact_Moment_val[0] =  this->Cinter_moment[1] - this->init_Contact_Moment[1]; 
            this->Contact_Moment_val[1] = -this->Cinter_moment[2] + this->init_Contact_Moment[2]; 
            this->Contact_Moment_val[2] = -this->Cinter_moment[0] + this->init_Contact_Moment[0]; 
            #endif

            /*
            // Contact moment filtering
            this->Contact_Moment_val[0] = yoon_indy7_filter::VR_mv_filter(this->Contact_Moment_val[0], &this->FT_mv_Cmx);
            this->Contact_Moment_val[1] = yoon_indy7_filter::VR_mv_filter(this->Contact_Moment_val[1], &this->FT_mv_Cmy);
            this->Contact_Moment_val[2] = yoon_indy7_filter::VR_mv_filter(this->Contact_Moment_val[2], &this->FT_mv_Cmz);
            */
            return_out = 12;
        }
        else
        {
            return_out = 0;
        }
    }
    else
    {
        errhandle("Connection initialization was failed, please run it again");
        return_out = 0;
    }

    return return_out;

}
bool Yoon_FT_sensor::Sensor_value_init()
{
    while(this->sensor_init_counter < this->init_average_num)
    {
        for(int i=0;i<3;i++)
        {
            this->init_Force[i] += this->inter_force[i]/this->init_average_num;
            this->init_Moment[i] += this->inter_moment[i]/this->init_average_num;

            this->init_Contact_Force[i] += this->Cinter_force[i]/this->init_average_num;
            this->init_Contact_Moment[i] += this->Cinter_moment[i]/this->init_average_num;
        }

        this->sensor_init_counter++;
    }

    if(this->sensor_init_counter < this->init_average_num) return false;
    else return true;
}

void Yoon_FT_sensor::errhandle(const char *errmsg){ // for FT_communication
  fputs(errmsg, stderr);
  fputc('\n', stderr);
  close(this->clnt_sock);
  exit(1);
}