#include <stdio.h>
#include "Yoon_FT_sensor.h"
#include "Yoon_filters.h"
#include "ros/ros.h"
#include "ft_data_acquisit/ftsensorMsg.h"
#include "std_msgs/UInt8.h"
#include <signal.h>
#include <ctime>
//---------------------------- Yaml file headers ---------------------------------//
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iostream>

#define Force_recording 0 // 0: Do not record, 1: record


Yoon_FT_sensor FT_sensor;
int value_init_counter = 0;
char key_MODE='0';

FILE *Data1_txt;
double time_counter = 0;
double time_step = 0.01; // second

#if 1 // Moving average filter
Yoon_filters MV_filter_Fx,MV_filter_Fy,MV_filter_Fz; // Moving average filter instance
Yoon_filters MV_filter_Mx,MV_filter_My,MV_filter_Mz; // Moving average filter instance

Yoon_filters MV_filter_CFx,MV_filter_CFy,MV_filter_CFz; // Moving average filter instance
Yoon_filters MV_filter_CMx,MV_filter_CMy,MV_filter_CMz; // Moving average filter instance
#endif

#if 1 // Low pass filter
Yoon_filters LPF_Fx,LPF_Fy,LPF_Fz; //Low pass filter instance
Yoon_filters LPF_Mx,LPF_My,LPF_Mz; //Low pass filter instance

Yoon_filters LPF_CFx,LPF_CFy,LPF_CFz; //Low pass filter instance
Yoon_filters LPF_CMx,LPF_CMy,LPF_CMz; //Low pass filter instance
#endif

#if 1 // Band stop filter1
Yoon_filters BSF_Fx, BSF_Fy, BSF_Fz; //Band stop filter instance
Yoon_filters BSF_Mx, BSF_My, BSF_Mz; //Band stop filter instance

Yoon_filters BSF_CFx, BSF_CFy, BSF_CFz; //Band stop filter instance
Yoon_filters BSF_CMx, BSF_CMy, BSF_CMz; //Band stop filter instance
#endif

#if 1 // Band stop filter2
Yoon_filters BSF2_Fx, BSF2_Fy, BSF2_Fz; //Band stop filter instance
Yoon_filters BSF2_Mx, BSF2_My, BSF2_Mz; //Band stop filter instance

Yoon_filters BSF2_CFx, BSF2_CFy, BSF2_CFz; //Band stop filter instance
Yoon_filters BSF2_CMx, BSF2_CMy, BSF2_CMz; //Band stop filter instance
#endif

// ROS message instance
ft_data_acquisit::ftsensorMsg msg;

//---------------------------- Yaml file load ---------------------------------//
std::ifstream fin1("/home/gene/catkin_ws/src/ft_data_acquisit/NRS_yaml/NRS_UR10_IP.yaml");
std::ifstream fin2("/home/gene/catkin_ws/src/ft_data_acquisit/NRS_yaml/NRS_Record_Printing.yaml");
YAML::Node NRS_IP = YAML::Load(fin1);
YAML::Node NRS_recording = YAML::Load(fin2);

int External_zeroset_flag = 0;
void ft_callback_func(const std_msgs::UInt8::ConstPtr& msg)
{
	if(msg->data == 1)
	{
		External_zeroset_flag = msg->data;
		printf("Right zero set message was received \n");
	}
	else
	{
		printf("Wrong zero set message was received \n");
	}
	
}

void catch_signal(int sig)
{
    exit(1);
}


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "Yoon_FT_acquisit");
    ros::NodeHandle nh;

    ros::Publisher ft_pub = nh.advertise<ft_data_acquisit::ftsensorMsg>("/ftsensor", 10);
	ros::Subscriber ft_sub = nh.subscribe("zeroset",1000, ft_callback_func);

    signal(SIGTERM, catch_signal);// Termination
	signal(SIGINT, catch_signal);// Active

	// TCP init using yaml file
	auto YamlString_IP = NRS_IP["AFT80IP"].as<std::string>();
	char* AFT80_IP = const_cast<char*>(YamlString_IP.c_str());
	FT_sensor.TCP_init(AFT80_IP,4001);

	// Data1 recording init using yaml file
	auto YamlData1_path = NRS_recording["Data1_path"].as<std::string>();
	auto YamlData1_switch = NRS_recording["Data1_switch"].as<int>();
	Data1_txt = fopen(YamlData1_path.c_str(),"wt"); // Force/Torque recording
	
	#if 1 // Moving average filter
	// Moving average parameter setting
	MV_filter_Fx.MV_par.mv_num = 3;
	MV_filter_Fy.MV_par.mv_num = 3;
	MV_filter_Fz.MV_par.mv_num = 3;

	MV_filter_Mx.MV_par.mv_num = 3;
	MV_filter_My.MV_par.mv_num = 3;
	MV_filter_Mz.MV_par.mv_num = 3;

	// For contact sensor
	MV_filter_CFx.MV_par.mv_num = 3;
	MV_filter_CFy.MV_par.mv_num = 3;
	MV_filter_CFz.MV_par.mv_num = 3;

	MV_filter_CMx.MV_par.mv_num = 3;
	MV_filter_CMy.MV_par.mv_num = 3;
	MV_filter_CMz.MV_par.mv_num = 3;
	#endif

	#if 1 // Low pass filter
	LPF_Fx.LPF_par.CutOffFrequency = 10; // Hz
	LPF_Fy.LPF_par.CutOffFrequency = 10; // Hz
	LPF_Fz.LPF_par.CutOffFrequency = 10; // Hz

	LPF_Mx.LPF_par.CutOffFrequency = 10; // Hz
	LPF_My.LPF_par.CutOffFrequency = 10; // Hz
	LPF_Mz.LPF_par.CutOffFrequency = 10; // Hz

	LPF_Fx.LPF_par.SamplingFrequency = 100; //Hz
	LPF_Fy.LPF_par.SamplingFrequency = 100; //Hz
	LPF_Fz.LPF_par.SamplingFrequency = 100; //Hz

	LPF_Mx.LPF_par.SamplingFrequency = 100; //Hz
	LPF_My.LPF_par.SamplingFrequency = 100; //Hz
	LPF_Mz.LPF_par.SamplingFrequency = 100; //Hz

	// For contact sensor
	LPF_CFx.LPF_par.CutOffFrequency = 10; // Hz
	LPF_CFy.LPF_par.CutOffFrequency = 10; // Hz
	LPF_CFz.LPF_par.CutOffFrequency = 10; // Hz

	LPF_CMx.LPF_par.CutOffFrequency = 10; // Hz
	LPF_CMy.LPF_par.CutOffFrequency = 10; // Hz
	LPF_CMz.LPF_par.CutOffFrequency = 10; // Hz

	LPF_CFx.LPF_par.SamplingFrequency = 100; //Hz
	LPF_CFy.LPF_par.SamplingFrequency = 100; //Hz
	LPF_CFz.LPF_par.SamplingFrequency = 100; //Hz

	LPF_CMx.LPF_par.SamplingFrequency = 100; //Hz
	LPF_CMy.LPF_par.SamplingFrequency = 100; //Hz
	LPF_CMz.LPF_par.SamplingFrequency = 100; //Hz
	#endif

	#if 1 // Band stop filter 1
	double BSF_f_peak = 15; // stop frequency(Hz)
	double BSF_bandWidth = 5; // stop frequency width(Hz)
	double BSF_ts = 0.01; // sampling time(s)

	// Peak frequency
	BSF_Fx.BSF_par.f_peak = BSF_f_peak;
	BSF_Fy.BSF_par.f_peak = BSF_f_peak;
	BSF_Fz.BSF_par.f_peak = BSF_f_peak;
	BSF_Mx.BSF_par.f_peak = BSF_f_peak;
	BSF_My.BSF_par.f_peak = BSF_f_peak;
	BSF_Mz.BSF_par.f_peak = BSF_f_peak;

	BSF_CFx.BSF_par.f_peak = BSF_f_peak;
	BSF_CFy.BSF_par.f_peak = BSF_f_peak;
	BSF_CFz.BSF_par.f_peak = BSF_f_peak;
	BSF_CMx.BSF_par.f_peak = BSF_f_peak;
	BSF_CMy.BSF_par.f_peak = BSF_f_peak;
	BSF_CMz.BSF_par.f_peak = BSF_f_peak;
	// Bandwidth
	BSF_Fx.BSF_par.bandWidth = BSF_bandWidth;
	BSF_Fy.BSF_par.bandWidth = BSF_bandWidth;
	BSF_Fz.BSF_par.bandWidth = BSF_bandWidth;
	BSF_Mx.BSF_par.bandWidth = BSF_bandWidth;
	BSF_My.BSF_par.bandWidth = BSF_bandWidth;
	BSF_Mz.BSF_par.bandWidth = BSF_bandWidth;

	BSF_CFx.BSF_par.bandWidth = BSF_bandWidth;
	BSF_CFy.BSF_par.bandWidth = BSF_bandWidth;
	BSF_CFz.BSF_par.bandWidth = BSF_bandWidth;
	BSF_CMx.BSF_par.bandWidth = BSF_bandWidth;
	BSF_CMy.BSF_par.bandWidth = BSF_bandWidth;
	BSF_CMz.BSF_par.bandWidth = BSF_bandWidth;
	//BSF_ts
	BSF_Fx.BSF_par.ts = BSF_ts;
	BSF_Fy.BSF_par.ts = BSF_ts;
	BSF_Fz.BSF_par.ts = BSF_ts;
	BSF_Mx.BSF_par.ts = BSF_ts;
	BSF_My.BSF_par.ts = BSF_ts;
	BSF_Mz.BSF_par.ts = BSF_ts;

	BSF_CFx.BSF_par.ts = BSF_ts;
	BSF_CFy.BSF_par.ts = BSF_ts;
	BSF_CFz.BSF_par.ts = BSF_ts;
	BSF_CMx.BSF_par.ts = BSF_ts;
	BSF_CMy.BSF_par.ts = BSF_ts;
	BSF_CMz.BSF_par.ts = BSF_ts;

	#endif

	#if 1 // Band stop filter 1
	double BSF2_f_peak = 2.4; // stop frequency(Hz)
	double BSF2_bandWidth = 2; // stop frequency width(Hz)
	double BSF2_ts = 0.01; // sampling time(s)

	// Peak frequency
	BSF2_Fx.BSF_par.f_peak = BSF2_f_peak;
	BSF2_Fy.BSF_par.f_peak = BSF2_f_peak;
	BSF2_Fz.BSF_par.f_peak = BSF2_f_peak;
	BSF2_Mx.BSF_par.f_peak = BSF2_f_peak;
	BSF2_My.BSF_par.f_peak = BSF2_f_peak;
	BSF2_Mz.BSF_par.f_peak = BSF2_f_peak;

	BSF2_CFx.BSF_par.f_peak = BSF2_f_peak;
	BSF2_CFy.BSF_par.f_peak = BSF2_f_peak;
	BSF2_CFz.BSF_par.f_peak = BSF2_f_peak;
	BSF2_CMx.BSF_par.f_peak = BSF2_f_peak;
	BSF2_CMy.BSF_par.f_peak = BSF2_f_peak;
	BSF2_CMz.BSF_par.f_peak = BSF2_f_peak;
	// Bandwidth
	BSF2_Fx.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_Fy.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_Fz.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_Mx.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_My.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_Mz.BSF_par.bandWidth = BSF2_bandWidth;

	BSF2_CFx.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_CFy.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_CFz.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_CMx.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_CMy.BSF_par.bandWidth = BSF2_bandWidth;
	BSF2_CMz.BSF_par.bandWidth = BSF2_bandWidth;
	//BSF_ts
	BSF2_Fx.BSF_par.ts = BSF2_ts;
	BSF2_Fy.BSF_par.ts = BSF2_ts;
	BSF2_Fz.BSF_par.ts = BSF2_ts;
	BSF2_Mx.BSF_par.ts = BSF2_ts;
	BSF2_My.BSF_par.ts = BSF2_ts;
	BSF2_Mz.BSF_par.ts = BSF2_ts;

	BSF2_CFx.BSF_par.ts = BSF2_ts;
	BSF2_CFy.BSF_par.ts = BSF2_ts;
	BSF2_CFz.BSF_par.ts = BSF2_ts;
	BSF2_CMx.BSF_par.ts = BSF2_ts;
	BSF2_CMy.BSF_par.ts = BSF2_ts;
	BSF2_CMz.BSF_par.ts = BSF2_ts;

	#endif

	printf("While loop start\n");

    printf("MODE SELECT: printing mode\n");
    printf("1 : Printing FT data,  2 : Non-Printing FT data\n");

    key_MODE=getchar();
	
	while(1)
	{
		if(FT_sensor.TCP_start() == 2) // if the contact sensor added, must be changed
		{
			// sensor value initialization
			
			if(External_zeroset_flag == 1)
			{
				FT_sensor.sensor_init_counter = 0;
				for(int i=0;i<3;i++)
				{
					FT_sensor.init_Force[i] = 0;
					FT_sensor.init_Moment[i] = 0;

					FT_sensor.init_Contact_Force[i] = 0;
					FT_sensor.init_Contact_Moment[i] = 0;
				}
				External_zeroset_flag = 2;

			}
			else if(External_zeroset_flag == 2 && ~FT_sensor.Sensor_value_init()) 
			{
				External_zeroset_flag = 0;
			}
			
            #if 1 // Moving average filter
            // Moving average filter test
            FT_sensor.Force_val[0] = MV_filter_Fx.MovingAvgFilter(FT_sensor.Force_val[0]);
            FT_sensor.Force_val[1] = MV_filter_Fy.MovingAvgFilter(FT_sensor.Force_val[1]);
            FT_sensor.Force_val[2] = MV_filter_Fz.MovingAvgFilter(FT_sensor.Force_val[2]);
            
            FT_sensor.Moment_val[0] = MV_filter_Mx.MovingAvgFilter(FT_sensor.Moment_val[0]);
            FT_sensor.Moment_val[1] = MV_filter_My.MovingAvgFilter(FT_sensor.Moment_val[1]);
            FT_sensor.Moment_val[2] = MV_filter_Mz.MovingAvgFilter(FT_sensor.Moment_val[2]);

			FT_sensor.Contact_Force_val[0] = MV_filter_CFx.MovingAvgFilter(FT_sensor.Contact_Force_val[0]);
            FT_sensor.Contact_Force_val[1] = MV_filter_CFy.MovingAvgFilter(FT_sensor.Contact_Force_val[1]);
            FT_sensor.Contact_Force_val[2] = MV_filter_CFz.MovingAvgFilter(FT_sensor.Contact_Force_val[2]);
            
            FT_sensor.Contact_Moment_val[0] = MV_filter_CMx.MovingAvgFilter(FT_sensor.Contact_Moment_val[0]);
            FT_sensor.Contact_Moment_val[1] = MV_filter_CMy.MovingAvgFilter(FT_sensor.Contact_Moment_val[1]);
            FT_sensor.Contact_Moment_val[2] = MV_filter_CMz.MovingAvgFilter(FT_sensor.Contact_Moment_val[2]);
            #endif

            #if 1 // Low pass filter
            // Low pass filter test
            FT_sensor.Force_val[0] = LPF_Fx.LowPassFilter(FT_sensor.Force_val[0]);
            FT_sensor.Force_val[1] = LPF_Fy.LowPassFilter(FT_sensor.Force_val[1]);
            FT_sensor.Force_val[2] = LPF_Fz.LowPassFilter(FT_sensor.Force_val[2]);
            
            FT_sensor.Moment_val[0] = LPF_Mx.LowPassFilter(FT_sensor.Moment_val[0]);
            FT_sensor.Moment_val[1] = LPF_My.LowPassFilter(FT_sensor.Moment_val[1]);
            FT_sensor.Moment_val[2] = LPF_Mz.LowPassFilter(FT_sensor.Moment_val[2]);

			FT_sensor.Contact_Force_val[0] = LPF_CFx.LowPassFilter(FT_sensor.Contact_Force_val[0]);
            FT_sensor.Contact_Force_val[1] = LPF_CFy.LowPassFilter(FT_sensor.Contact_Force_val[1]);
            FT_sensor.Contact_Force_val[2] = LPF_CFz.LowPassFilter(FT_sensor.Contact_Force_val[2]);
            
            FT_sensor.Contact_Moment_val[0] = LPF_CMx.LowPassFilter(FT_sensor.Contact_Moment_val[0]);
            FT_sensor.Contact_Moment_val[1] = LPF_CMy.LowPassFilter(FT_sensor.Contact_Moment_val[1]);
            FT_sensor.Contact_Moment_val[2] = LPF_CMz.LowPassFilter(FT_sensor.Contact_Moment_val[2]);
            #endif

            #if 1 // BSF filter1
            FT_sensor.Force_val[0] = BSF_Fx.BandStopFilter(FT_sensor.Force_val[0]);
            FT_sensor.Force_val[1] = BSF_Fy.BandStopFilter(FT_sensor.Force_val[1]);
            FT_sensor.Force_val[2] = BSF_Fz.BandStopFilter(FT_sensor.Force_val[2]);
            
            FT_sensor.Moment_val[0] = BSF_Mx.BandStopFilter(FT_sensor.Moment_val[0]);
            FT_sensor.Moment_val[1] = BSF_My.BandStopFilter(FT_sensor.Moment_val[1]);
            FT_sensor.Moment_val[2] = BSF_Mz.BandStopFilter(FT_sensor.Moment_val[2]);

			FT_sensor.Contact_Force_val[0] = BSF_CFx.BandStopFilter(FT_sensor.Contact_Force_val[0]);
            FT_sensor.Contact_Force_val[1] = BSF_CFy.BandStopFilter(FT_sensor.Contact_Force_val[1]);
            FT_sensor.Contact_Force_val[2] = BSF_CFz.BandStopFilter(FT_sensor.Contact_Force_val[2]);
            
            FT_sensor.Contact_Moment_val[0] = BSF_CMx.BandStopFilter(FT_sensor.Contact_Moment_val[0]);
            FT_sensor.Contact_Moment_val[1] = BSF_CMy.BandStopFilter(FT_sensor.Contact_Moment_val[1]);
            FT_sensor.Contact_Moment_val[2] = BSF_CMz.BandStopFilter(FT_sensor.Contact_Moment_val[2]);
            #endif

			#if 0 // BSF filter2
            FT_sensor.Force_val[0] = BSF2_Fx.BandStopFilter(FT_sensor.Force_val[0]);
            FT_sensor.Force_val[1] = BSF2_Fy.BandStopFilter(FT_sensor.Force_val[1]);
            FT_sensor.Force_val[2] = BSF2_Fz.BandStopFilter(FT_sensor.Force_val[2]);
            
            FT_sensor.Moment_val[0] = BSF2_Mx.BandStopFilter(FT_sensor.Moment_val[0]);
            FT_sensor.Moment_val[1] = BSF2_My.BandStopFilter(FT_sensor.Moment_val[1]);
            FT_sensor.Moment_val[2] = BSF2_Mz.BandStopFilter(FT_sensor.Moment_val[2]);

			FT_sensor.Contact_Force_val[0] = BSF2_CFx.BandStopFilter(FT_sensor.Contact_Force_val[0]);
            FT_sensor.Contact_Force_val[1] = BSF2_CFy.BandStopFilter(FT_sensor.Contact_Force_val[1]);
            FT_sensor.Contact_Force_val[2] = BSF2_CFz.BandStopFilter(FT_sensor.Contact_Force_val[2]);
            
            FT_sensor.Contact_Moment_val[0] = BSF2_CMx.BandStopFilter(FT_sensor.Contact_Moment_val[0]);
            FT_sensor.Contact_Moment_val[1] = BSF2_CMy.BandStopFilter(FT_sensor.Contact_Moment_val[1]);
            FT_sensor.Contact_Moment_val[2] = BSF2_CMz.BandStopFilter(FT_sensor.Contact_Moment_val[2]);
            #endif


			// sensor value printing
			if(key_MODE == '1')
			{
				printf("Fx:%10f, Fy:%10f, Fz:%10f \n",FT_sensor.Force_val[0],FT_sensor.Force_val[1],FT_sensor.Force_val[2]);
				printf("Mx:%10f, My:%10f, Mz:%10f \n",FT_sensor.Moment_val[0],FT_sensor.Moment_val[1],FT_sensor.Moment_val[2]);
				printf("CFx:%10f, CFy:%10f, CFz:%10f \n",FT_sensor.Contact_Force_val[0],FT_sensor.Contact_Force_val[1],FT_sensor.Contact_Force_val[2]);
				printf("CMx:%10f, CMy:%10f, CMz:%10f \n",FT_sensor.Contact_Moment_val[0],FT_sensor.Contact_Moment_val[1],FT_sensor.Contact_Moment_val[2]);
			}

			if(YamlData1_switch == 1){
				if(Data1_txt!=NULL)
				{
					// data recording
					fprintf(Data1_txt, "%10f %10f %10f %10f %10f %10f %10f\n",time_counter*time_step,FT_sensor.Force_val[0],FT_sensor.Force_val[1],FT_sensor.Force_val[2],
					FT_sensor.Moment_val[0],FT_sensor.Moment_val[1],FT_sensor.Moment_val[2]);
				}
				else
				{
					printf("Data1 does not open : warnning !!");
					exit(1);
				}
			}

            // Send the Force/Torque data with ROS message
            msg.Fx = FT_sensor.Force_val[0];
            msg.Fy = FT_sensor.Force_val[1];
            msg.Fz = FT_sensor.Force_val[2];

            msg.Mx = FT_sensor.Moment_val[0];
            msg.My = FT_sensor.Moment_val[1];
            msg.Mz = FT_sensor.Moment_val[2];

			msg.CFx = FT_sensor.Contact_Force_val[0];
            msg.CFy = FT_sensor.Contact_Force_val[1];
            msg.CFz = FT_sensor.Contact_Force_val[2];

            msg.CMx = FT_sensor.Contact_Moment_val[0];
            msg.CMy = FT_sensor.Contact_Moment_val[1];
            msg.CMz = FT_sensor.Contact_Moment_val[2];


            ft_pub.publish(msg);

			time_counter = time_counter + 1.0;
			ros::spinOnce();
		}
		
	}
	fclose(Data1_txt);
	return 0;	
}

